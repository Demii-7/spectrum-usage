"""Autoformer-CSA model implementation.

The implementation follows the Autoformer decomposition and auto-correlation
architecture.  CSAM replaces the feed-forward sublayer in the encoder and
decoder with channel and temporal-position attention over the embedded series.
The upstream primitives live here so the integrated runner has no external
checkout requirement.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotConfig(dict[str, Any]):
    """Small attribute-access wrapper for the model configuration."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class MovingAverage(nn.Module):
    """Centered moving average with edge replication."""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("moving_avg must be a positive odd integer")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        front = x[:, :1, :].expand(-1, padding, -1)
        end = x[:, -1:, :].expand(-1, padding, -1)
        padded = torch.cat([front, x, end], dim=1)
        return self.avg(padded.transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        return x - trend, trend


class SeasonalLayerNorm(nn.Module):
    """LayerNorm that removes the mean bias from the seasonal component."""

    def __init__(self, channels: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.layernorm(x)
        bias = normalized.mean(dim=1, keepdim=True).expand_as(normalized)
        return normalized - bias


class TokenEmbedding(nn.Module):
    def __init__(self, channels_in: int, d_model: int):
        super().__init__()
        self.token_conv = nn.Conv1d(
            channels_in,
            d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.transpose(1, 2)).transpose(1, 2)


class TimeFeatureEmbedding(nn.Module):
    """Linear embedding for the four hourly calendar features."""

    def __init__(self, d_model: int):
        super().__init__()
        self.embed = nn.Linear(4, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class DataEmbeddingWithoutPosition(nn.Module):
    """Autoformer's value plus time embedding without positional encoding."""

    def __init__(self, channels_in: int, d_model: int, dropout: float):
        super().__init__()
        self.value_embedding = TokenEmbedding(channels_in, d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.temporal_embedding(marks))


class AutoCorrelation(nn.Module):
    """FFT-based period discovery and time-delay aggregation."""

    def __init__(
        self,
        mask_flag: bool,
        factor: int,
        attention_dropout: float,
        output_attention: bool = False,
    ):
        super().__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _top_k(self, length: int) -> int:
        return max(1, min(length, int(self.factor * math.log(max(length, 2)))))

    def _aggregate_training(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        _, heads, channels, length = values.shape
        top_k = self._top_k(length)
        mean_value = corr.mean(dim=1).mean(dim=1)
        index = torch.topk(mean_value.mean(dim=0), top_k, dim=-1).indices
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        weights = torch.softmax(weights, dim=-1)
        output = torch.zeros_like(values)
        for i in range(top_k):
            delayed = torch.roll(values, -int(index[i]), dims=-1)
            output = output + delayed * weights[:, i].view(-1, 1, 1, 1)
        return output

    def _aggregate_inference(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        batch, heads, channels, length = values.shape
        top_k = self._top_k(length)
        mean_value = corr.mean(dim=1).mean(dim=1)
        weights, delays = torch.topk(mean_value, top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        indices = torch.arange(length, device=values.device).view(1, 1, 1, length)
        indices = indices.expand(batch, heads, channels, -1)
        doubled = values.repeat(1, 1, 1, 2)
        output = torch.zeros_like(values)
        for i in range(top_k):
            gather_index = indices + delays[:, i].view(batch, 1, 1, 1)
            delayed = torch.gather(doubled, -1, gather_index)
            output = output + delayed * weights[:, i].view(batch, 1, 1, 1)
        return output

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del attn_mask
        batch, length, _, _ = queries.shape
        value_length = values.shape[1]
        if length > value_length:
            padding = torch.zeros_like(queries[:, : length - value_length])
            values = torch.cat([values, padding], dim=1)
            keys = torch.cat([keys, padding], dim=1)
        else:
            values = values[:, :length]
            keys = keys[:, :length]

        query_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        key_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        correlation = torch.fft.irfft(query_fft * torch.conj(key_fft), n=length, dim=-1)
        values_by_delay = values.permute(0, 2, 3, 1).contiguous()
        if self.training:
            aggregated = self._aggregate_training(values_by_delay, correlation)
        else:
            aggregated = self._aggregate_inference(values_by_delay, correlation)
        output = aggregated.permute(0, 3, 1, 2).contiguous()
        return output, correlation.permute(0, 3, 1, 2) if self.output_attention else None


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation: AutoCorrelation, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, head_dim * n_heads)
        self.key_projection = nn.Linear(d_model, head_dim * n_heads)
        self.value_projection = nn.Linear(d_model, head_dim * n_heads)
        self.out_projection = nn.Linear(head_dim * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, query_length, _ = queries.shape
        key_length = keys.shape[1]
        queries = self.query_projection(queries).view(batch, query_length, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch, key_length, self.n_heads, -1)
        values = self.value_projection(values).view(batch, key_length, self.n_heads, -1)
        output, attention = self.inner_correlation(queries, keys, values, attn_mask)
        return self.out_projection(output.view(batch, query_length, -1)), attention


class ChannelAttention(nn.Module):
    """Paper CSAM channel attention over the embedded feature channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_projection = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.max_projection = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = self.avg_projection(self.avg_pool(x))
        maximum = self.max_projection(self.max_pool(x))
        return torch.sigmoid(average + maximum)


class TemporalPositionAttention(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("csam_kernel_size must be a positive odd integer")
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))


class CSAM(nn.Module):
    """Series Channel-Spatial Attention Module from Autoformer-CSA."""

    def __init__(self, d_model: int, d_ff: int, kernel_size: int, dropout: float):
        super().__init__()
        # The paper retains the first Autoformer point-wise projection before
        # channel attention, then projects the refined features back to d_model.
        self.feed_forward = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.channel_attention = ChannelAttention(d_ff)
        self.position_attention = TemporalPositionAttention(kernel_size)
        self.projection = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feed_forward_dropout(F.relu(self.feed_forward(x)))
        x = x * self.channel_attention(x)
        x = x * self.position_attention(x)
        return self.dropout(self.projection(x))


class EncoderLayerCSA(nn.Module):
    def __init__(self, attention: AutoCorrelationLayer, config: DotConfig):
        super().__init__()
        self.attention = attention
        self.csam = CSAM(
            config.d_model,
            config.d_ff,
            config.csam_kernel_size,
            config.dropout,
        )
        self.decomp1 = SeriesDecomposition(config.moving_avg)
        self.decomp2 = SeriesDecomposition(config.moving_avg)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, attention = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(attended)
        x, _ = self.decomp1(x)
        csam_out = self.csam(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.decomp2(x + csam_out)
        return output, attention


class DecoderLayerCSA(nn.Module):
    def __init__(
        self,
        self_attention: AutoCorrelationLayer,
        cross_attention: AutoCorrelationLayer,
        config: DotConfig,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.csam = CSAM(
            config.d_model,
            config.d_ff,
            config.csam_kernel_size,
            config.dropout,
        )
        self.decomp1 = SeriesDecomposition(config.moving_avg)
        self.decomp2 = SeriesDecomposition(config.moving_avg)
        self.decomp3 = SeriesDecomposition(config.moving_avg)
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Conv1d(
            config.d_model,
            config.c_out,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self_out, _ = self.self_attention(x, x, x, x_mask)
        x = x + self.dropout(self_out)
        x, trend1 = self.decomp1(x)
        cross_out, _ = self.cross_attention(x, cross, cross, cross_mask)
        x = x + self.dropout(cross_out)
        x, trend2 = self.decomp2(x)
        csam_out = self.csam(x.transpose(1, 2)).transpose(1, 2)
        x, trend3 = self.decomp3(x + csam_out)
        trend = trend1 + trend2 + trend3
        trend = self.projection(trend.transpose(1, 2)).transpose(1, 2)
        return x, trend


class Encoder(nn.Module):
    def __init__(self, layers: list[EncoderLayerCSA], norm: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        attentions = []
        for layer in self.layers:
            x, attention = layer(x, attn_mask)
            attentions.append(attention)
        return self.norm(x), attentions


class Decoder(nn.Module):
    def __init__(self, layers: list[DecoderLayerCSA], norm: nn.Module, projection: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.projection = projection

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        trend: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask, cross_mask)
            trend = trend + residual_trend
        return self.projection(self.norm(x)), trend


class AutoformerCSA(nn.Module):
    """Autoformer with CSAM replacing the encoder and decoder FFNs."""

    def __init__(self, config: DotConfig):
        super().__init__()
        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.output_attention = config.output_attention
        self.decomp = SeriesDecomposition(config.moving_avg)
        self.enc_embedding = DataEmbeddingWithoutPosition(config.enc_in, config.d_model, config.dropout)
        self.dec_embedding = DataEmbeddingWithoutPosition(config.dec_in, config.d_model, config.dropout)

        def correlation(mask_flag: bool, output_attention: bool = False) -> AutoCorrelationLayer:
            return AutoCorrelationLayer(
                AutoCorrelation(
                    mask_flag,
                    config.factor,
                    config.dropout,
                    output_attention,
                ),
                config.d_model,
                config.n_heads,
            )

        self.encoder = Encoder(
            [EncoderLayerCSA(correlation(False, config.output_attention), config) for _ in range(config.e_layers)],
            SeasonalLayerNorm(config.d_model),
        )
        self.decoder = Decoder(
            [
                DecoderLayerCSA(correlation(True), correlation(False), config)
                for _ in range(config.d_layers)
            ],
            SeasonalLayerNorm(config.d_model),
            nn.Linear(config.d_model, config.c_out),
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        dec_enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        mean = x_enc.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        zeros = torch.zeros_like(x_dec[:, : self.pred_len])
        seasonal, trend = self.decomp(x_enc)
        trend = torch.cat([trend[:, -self.label_len :], mean], dim=1)
        seasonal = torch.cat([seasonal[:, -self.label_len :], zeros], dim=1)

        encoded = self.enc_embedding(x_enc, x_mark_enc)
        encoded, attentions = self.encoder(encoded, enc_self_mask)
        decoded = self.dec_embedding(seasonal, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            decoded,
            encoded,
            trend,
            dec_self_mask,
            dec_enc_mask,
        )
        output = (trend_part + seasonal_part)[:, -self.pred_len :]
        return (output, attentions) if self.output_attention else output


class AutoformerCSAForecaster(nn.Module):
    """Shared-pipeline adapter exposing ``(x) -> forecast``."""

    def __init__(self, config: DotConfig):
        super().__init__()
        self.model = AutoformerCSA(config)
        self.prediction_horizon = config.pred_len
        self.label_len = config.label_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = x.shape
        decoder_length = self.label_len + self.prediction_horizon
        decoder_input = torch.zeros(
            batch,
            decoder_length,
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        decoder_input[:, : self.label_len] = x[:, -self.label_len :]
        encoder_marks = torch.zeros(batch, sequence_length, 4, device=x.device, dtype=x.dtype)
        decoder_marks = torch.zeros(batch, decoder_length, 4, device=x.device, dtype=x.dtype)
        output = self.model(x, encoder_marks, decoder_input, decoder_marks)
        return output[0] if isinstance(output, tuple) else output
