"""
Model definitions for Autoformer with Channel-Spatial Attention (CSA).

Implements AutoformerCSA, a time-series forecasting model that augments the
standard Autoformer architecture with a convolutional Channel-Spatial Attention
Module (CSAM) inserted within each encoder and decoder layer.  The CSA module
reweights features both across channels (frequency bins) and across spatial
positions (time steps), helping the model focus on spectrally salient regions.

Also provides a vanilla Autoformer variant (AutoformerVanilla) that can be
imported from the upstream Autoformer repository when available.
"""

import sys, os, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

_AUTOFORMER_REPO = os.environ.get(
    "AUTOFORMER_REPO",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "extern", "Autoformer"),
)
if os.path.isdir(_AUTOFORMER_REPO) and _AUTOFORMER_REPO not in sys.path:
    sys.path.insert(0, _AUTOFORMER_REPO)

from layers.Autoformer_EncDec import (
    series_decomp, moving_avg, my_Layernorm,
    Encoder as UpstreamEncoder, Decoder as UpstreamDecoder,
    EncoderLayer, DecoderLayer,
)
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Embed import DataEmbedding_wo_pos


class ChannelAttention(nn.Module):
    """Channel-wise attention module (1D variant of CBAM).

    Squeezes each channel to a scalar via average and max pooling, then
    learns per-channel importance weights through a shared bottleneck MLP.
    The two pooled descriptors are fused element-wise and passed through a
    sigmoid to produce a channel-gating mask.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        # Global pooling reduces the spatial dimension (time) to 1 per channel
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # Bottleneck MLP: compress then expand channels
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Fuse both pooled descriptors before the gate
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial (temporal) attention module along the time dimension.

    Stacks the channel-wise average and max into a 2-channel descriptor,
    applies a 1D convolution to learn local temporal importance, and gates
    the input with a sigmoid mask.  The kernel size controls the receptive
    field over neighbouring time steps.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        # 1D conv operating over time; padding preserves length
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CSAM(nn.Module):
    """Channel-Spatial Attention Module.

    Sequentially applies channel attention then spatial attention, followed
    by a pointwise convolution for channel mixing and dropout for
    regularisation.  The mask is applied multiplicatively so that the network
    learns to suppress irrelevant frequency bands and time steps.
    """

    def __init__(self, d_model, kernel_size=7, reduction=16, dropout=0.1):
        super().__init__()
        self.channel_attn = ChannelAttention(d_model, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x * self.channel_attn(x)
        out = out * self.spatial_attn(out)
        out = self.dropout(self.conv(out))
        return out


class EncoderLayerCSA(nn.Module):
    """Single encoder layer combining Autoformer series decomposition with CSAM.

    After the AutoCorrelation attention and residual addition, the signal
    is decomposed into seasonal and trend components via moving-average.
    The seasonal part is further refined by the CSAM module to highlight
    informative channel/spatial patterns, then recombined with the original
    seasonal path before a second decomposition.  The final trend component
    is discarded inside the encoder (only seasonal is passed upward).
    """

    def __init__(self, attention, d_model, moving_avg=25, dropout=0.1,
                 activation="relu", csam_kernel_size=7, csam_reduction=16):
        super().__init__()
        self.attention = attention
        self.csam = CSAM(d_model, csam_kernel_size, csam_reduction, dropout)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # 1) AutoCorrelation self-attention with residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        # 2) Series decomposition: isolate seasonal and trend
        x, _ = self.decomp1(x)
        # 3) CSAM refinement on the seasonal component
        y = x
        y = y.transpose(-1, 1)          # [B, D, T] for Conv1d
        y = self.csam(y)
        y = y.transpose(-1, 1)          # back to [B, T, D]
        # 4) Combine CSAM output with original seasonal, then decompose again
        res, _ = self.decomp2(x + y)
        return res, attn


class DecoderLayerCSA(nn.Module):
    """Single decoder layer with self-attention, cross-attention, and CSAM.

    The decoder processes the seasonal component through two attention blocks
    (self then cross-attending to encoder outputs), each followed by series
    decomposition.  The seasonal stream is then augmented with CSAM before a
    third decomposition.  Accumulated trend components from all three stages
    are projected to the output channel dimension.
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out,
                 moving_avg=25, dropout=0.1, activation="relu",
                 csam_kernel_size=7, csam_reduction=16):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.csam = CSAM(d_model, csam_kernel_size, csam_reduction, dropout)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        # Project accumulated trend to c_out channels
        self.projection = nn.Conv1d(
            in_channels=d_model, out_channels=c_out,
            kernel_size=3, stride=1, padding=1, padding_mode="circular", bias=False,
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 1) Self-attention + series decomposition
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        # 2) Cross-attention to encoder outputs + decomposition
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        # 3) CSAM refinement on seasonal + third decomposition
        y = x
        y = y.transpose(-1, 1)
        y = self.csam(y)
        y = y.transpose(-1, 1)
        x, trend3 = self.decomp3(x + y)

        # Sum all extracted trends and project to output channels
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class AutoformerCSA(nn.Module):
    """Autoformer with Channel-Spatial Attention Module (CSAM).

    Extends the standard Autoformer architecture by inserting CSAM blocks
    inside each encoder/decoder layer.  The model uses series decomposition
    (moving-average based) to separate seasonal and trend components, and
    leverages AutoCorrelation for discovering period-based dependencies.

    Configs (DotDict) must include:
        seq_len, label_len, pred_len, enc_in, dec_in, c_out, d_model,
        n_heads, e_layers, d_layers, moving_avg, dropout, factor,
        activation, output_attention, embed, freq,
        csam_kernel_size, csam_reduction.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embeddings without learned positional encoding (Autoformer style)
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
        )

        # Stacked encoder layers with CSAM
        self.encoder = UpstreamEncoder(
            [
                EncoderLayerCSA(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    csam_kernel_size=configs.csam_kernel_size,
                    csam_reduction=configs.csam_reduction,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )

        # Stacked decoder layers with CSAM (uses masked self-attention
        # and cross-attention to encoder outputs)
        self.decoder = UpstreamDecoder(
            [
                DecoderLayerCSA(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True, configs.factor,
                            attention_dropout=configs.dropout, output_attention=False,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False, configs.factor,
                            attention_dropout=configs.dropout, output_attention=False,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model,
                    configs.c_out,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    csam_kernel_size=configs.csam_kernel_size,
                    csam_reduction=configs.csam_reduction,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Prepare decoder trend init: repeat the mean of encoder input
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        # Decompose encoder input into seasonal and trend
        seasonal_init, trend_init = self.decomp(x_enc)

        # Concatenate last label_len steps with zeros/mean for decoder starting point
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # Encode the input sequence
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decode using seasonal_init and cross-attending to enc_out
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out,
            x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init,
        )

        dec_out = trend_part + seasonal_part

        # Slice only the prediction horizon and optionally return attention
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
