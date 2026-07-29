"""
TSS-LCD model definitions.

Implements the three core components of the Time-Series Spectrum Latent
Conditional Diffusion framework:
  1. TSS Condition Constructor (TSS-CC) — a transformer-based encoder that
     produces latent condition vectors from spatiotemporal spectrogram input.
  2. Latent Space Encoder/Decoder (LSE/LSD) — a Conv2D autoencoder that maps
     future spectrogram slices to/from a compact latent space.
  3. Diffusion Model — a denoising probabilistic model that operates in the
     latent space, conditioned on the TSS-CC output.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#  Activation & Normalization Factories
# =====================================================================

def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def _get_norm1d(name: str, channels: int) -> nn.Module:
    name = name.lower()
    if name == "batchnorm":
        return nn.BatchNorm1d(channels)
    elif name == "layernorm":
        return nn.LayerNorm(channels)
    elif name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown normalization: {name}")


# =====================================================================
#  Common Building Blocks (from repo Context2CondNew.py)
# =====================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding injected into transformer tokens.

    Provides position-aware information by adding fixed sine/cosine
    signals so the attention mechanism can exploit sequence order.
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Compute the division term for each frequency pair
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention.

    Projects input to Q, K, V via a single linear layer, splits into
    `num_heads`, computes attention scores, and concatenates outputs.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.W_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.W_qkv(x).chunk(3, dim=-1)
        # Reshape to (B, num_heads, T, d_k) for batched attention
        Q, K, V = [t.view(B, T, self.num_heads, self.d_k).transpose(1, 2) for t in qkv]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """One transformer encoder block: self-attention + feed-forward with residual connections."""

    def __init__(self, dim: int, num_heads: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual architecture: add & norm after each sub-layer
        x = self.norm1(x + self.dropout(self.self_attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, dim: int, num_heads: int, num_layers: int,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention block: queries attend to key-value from another sequence.

    Projects key-value to query dimension if needed, applies multi-head
    cross-attention, then a feed-forward block — both with residual
    connections and layer norm.
    """

    def __init__(self, dim_q: int, dim_kv: int, num_heads: int,
                 dropout: float = 0.1, ffn_dim: int = 2048):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_q, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.kv_proj = nn.Linear(dim_kv, dim_q) if dim_kv != dim_q else nn.Identity()
        self.norm1 = nn.LayerNorm(dim_q)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim_q, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim_q),
        )
        self.norm2 = nn.LayerNorm(dim_q)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        kv = self.kv_proj(x_kv)
        attn_out, _ = self.cross_attn(query=x_q, key=kv, value=kv)
        x = self.norm1(x_q + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# =====================================================================
#  TSS-CC Branches
# =====================================================================

class TemporalFE(nn.Module):
    """Temporal feature extractor with T tokens and L*F features per token."""

    def __init__(self, T_in: int, L: int, F: int, hidden_dim: int,
                 num_heads: int, num_layers: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.L, self.F = L, F
        self.token_dim = L * F
        self.num_tokens = T_in
        self.proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=self.num_tokens)
        self.encoder = TransformerEncoder(hidden_dim, num_heads, num_layers, ffn_dim, dropout)
        self.out_proj = nn.Linear(hidden_dim, self.token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, L, F = x.shape
        x = x.reshape(B, T_in, L * F)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.out_proj(x)


class SpectralFE(nn.Module):
    """Spectral feature extractor with F tokens and L*T features per token."""

    def __init__(self, T_in: int, L: int, F: int, hidden_dim: int,
                 num_heads: int, num_layers: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.T_in, self.L, self.F = T_in, L, F
        self.token_dim = L * T_in
        self.num_tokens = F
        self.proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=F)
        self.encoder = TransformerEncoder(hidden_dim, num_heads, num_layers, ffn_dim, dropout)
        self.out_proj = nn.Linear(hidden_dim, self.token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, L, F = x.shape
        # Keep locations grouped within each frequency token.
        x = x.permute(0, 3, 2, 1).reshape(B, F, L * T_in)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.out_proj(x)


class SpatialFE(nn.Module):
    """Spatial feature extractor with L tokens and F*T features per token."""

    def __init__(self, T_in: int, L: int, F: int, hidden_dim: int,
                 num_heads: int, num_layers: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.T_in, self.L, self.F = T_in, L, F
        self.token_dim = F * T_in
        self.num_tokens = L
        self.proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=L)
        self.encoder = TransformerEncoder(hidden_dim, num_heads, num_layers, ffn_dim, dropout)
        self.out_proj = nn.Linear(hidden_dim, self.token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, L, F = x.shape
        # Frequency-major features make the paper's later (L,F*T)->(F,T*L)
        # value reshape an exact rearrangement rather than a projection.
        x = x.permute(0, 2, 3, 1).reshape(B, L, F * T_in)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.out_proj(x)


class FeatureFusionModule(nn.Module):
    """Paper FFM: temporal Q, spectral K, reshaped spatial V."""

    def __init__(self, T_in: int, L: int, F: int, hidden_dim: int,
                 num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.T_in, self.L, self.F = T_in, L, F
        # The paper uses d=L*F. These adapters retain the configured hidden_dim
        # API when d differs, while preserving all native branch features.
        self.q_proj = nn.Linear(L * F, hidden_dim)
        self.k_proj = nn.Linear(L * T_in, hidden_dim)
        self.v_proj = nn.Linear(L * T_in, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def _spatial_value(self, spatial: torch.Tensor) -> torch.Tensor:
        B = spatial.shape[0]
        spatial_value = spatial.reshape(B, self.L, self.F, self.T_in)
        return spatial_value.permute(0, 2, 3, 1).reshape(
            B, self.F, self.T_in * self.L
        )

    def forward(self, temporal: torch.Tensor | None, spectral: torch.Tensor | None,
                spatial: torch.Tensor | None) -> torch.Tensor:
        spatial_value = self._spatial_value(spatial) if spatial is not None else None
        projected_spectral = self.k_proj(spectral) if spectral is not None else None
        projected_spatial = self.v_proj(spatial_value) if spatial_value is not None else None

        if temporal is not None:
            q = self.q_proj(temporal)
        else:
            source = projected_spectral if projected_spectral is not None else projected_spatial
            q = F.interpolate(
                source.transpose(1, 2), size=self.T_in, mode="linear", align_corners=False
            ).transpose(1, 2)
        k = projected_spectral if projected_spectral is not None else projected_spatial
        v = projected_spatial if projected_spatial is not None else projected_spectral
        if k is None:
            k = v = q
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        fused = self.norm1(q + self.dropout1(attn_out))
        return self.norm2(fused + self.dropout2(self.ffn(fused)))


class ConditionToLatentProjection(nn.Module):
    """Projects the fused condition tokens into a latent vector.

    The complete fixed-length fused temporal sequence is projected at once,
    avoiding the information loss of token averaging.
    """

    def __init__(self, hidden_dim: int, latent_dim: int, num_tokens: int = 1):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * num_tokens, latent_dim)

    def forward(self, H_fusion: torch.Tensor) -> torch.Tensor:
        z_pred = self.fc(H_fusion.flatten(start_dim=1))
        return z_pred


# =====================================================================
#  TSS-CC (full conditioner)
# =====================================================================

class TSSConditionConstructor(nn.Module):
    """TSS Condition Constructor — the core conditioner of the framework.

    Processes the input spectrogram through up to three parallel
    transformer branches (Temporal, Spectral, Spatial), fuses them via
    cross-attention, and projects the result into a latent condition
    vector that guides the diffusion denoising process.
    """

    def __init__(
        self,
        T_in: int,
        L: int,
        F: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        latent_dim: int = 32,
        use_temporal: bool = True,
        use_spectral: bool = True,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.T_in = T_in
        self.L = L
        self.F = F
        self.use_temporal = use_temporal
        self.use_spectral = use_spectral
        self.use_spatial = use_spatial

        active = sum([use_temporal, use_spectral, use_spatial])
        assert active > 0, "At least one TSS branch must be active"

        if use_temporal:
            self.temporal_fe = TemporalFE(T_in, L, F, hidden_dim, num_heads, num_layers, ffn_dim, dropout)
        if use_spectral:
            self.spectral_fe = SpectralFE(T_in, L, F, hidden_dim, num_heads, num_layers, ffn_dim, dropout)
        if use_spatial:
            self.spatial_fe = SpatialFE(T_in, L, F, hidden_dim, num_heads, num_layers, ffn_dim, dropout)

        self.ffm = FeatureFusionModule(T_in, L, F, hidden_dim, num_heads, dropout)

        self.to_latent = ConditionToLatentProjection(hidden_dim, latent_dim, T_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.T_in, self.L, self.F):
            raise ValueError(f"TSS-CC expected (B,{self.T_in},{self.L},{self.F}), got {tuple(x.shape)}")
        x_4d = x
        H_temporal = self.temporal_fe(x_4d) if self.use_temporal else None
        H_spectral = self.spectral_fe(x_4d) if self.use_spectral else None
        H_spatial = self.spatial_fe(x_4d) if self.use_spatial else None

        H_fusion = self.ffm(H_temporal, H_spectral, H_spatial)
        z_pred = self.to_latent(H_fusion)
        return z_pred


# =====================================================================
#  Latent Space Encoder / Decoder (Conv2D)
# =====================================================================

class LatentSpaceEncoder(nn.Module):
    """Conv2D encoder that maps a future spectrogram slice into a latent vector.

    The input (B, T_out, L*F) is reshaped into a 2D image-like tensor
    (B, 1, T_out, L*F), processed by convolutional blocks with max-pooling,
    globally pooled, and projected to `latent_dim`.
    """

    def __init__(self, T_out: int, L: int, F: int, latent_dim: int,
                 num_blocks: int = 3, init_channels: int = 32,
                 kernel_size: int = 3, pool_kernel: int = 2,
                 pool_stride: int = 2, activation: str = "relu"):
        super().__init__()
        # Exponentially grow channel count per block: 1 -> 32 -> 64 -> 128
        channels = [1] + [init_channels * (2 ** i) for i in range(num_blocks)]
        act_fn = _get_activation(activation)
        blocks = []
        padding = kernel_size // 2
        for i in range(num_blocks):
            blocks.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(channels[i + 1]),
                act_fn,
                nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            ])
        self.encoder = nn.Sequential(*blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_channels = channels[-1]
        self.fc = nn.Linear(channels[-1], latent_dim)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        if Y.ndim != 4:
            raise ValueError(f"Latent encoder expected (B,T,L,F), got {tuple(Y.shape)}")
        B, T_out, L, Freq = Y.shape
        D = L * Freq
        # Treat (T_out, D) as a 2D map with 1 channel
        x = Y.reshape(B, 1, T_out, D)
        x = self.encoder(x)
        x = self.adaptive_pool(x).view(B, self.final_channels)
        z = self.fc(x)
        return z


class LatentSpaceDecoder(nn.Module):
    """Conv2D decoder that reconstructs a spectrogram slice from a latent vector.

    The latent vector is projected to a 4x4 feature map, upsampled via
    transposed convolutions, and bilinearly resized to (T_out, L*F).
    """

    def __init__(self, T_out: int, L: int, F: int, latent_dim: int,
                 num_blocks: int = 3, init_channels: int = 32,
                 kernel_size: int = 3, activation: str = "relu"):
        super().__init__()
        self.T_out = T_out
        self.L = L
        self.F = F

        # Descending channel counts: reverse of the encoder's exponential growth
        channels = [init_channels * (2 ** i) for i in range(num_blocks)]
        channels = channels[::-1]
        self.init_C = channels[0]

        self.fc = nn.Linear(latent_dim, self.init_C * 4 * 4)

        act_fn = _get_activation(activation)
        blocks = []
        for i in range(num_blocks):
            in_c = channels[i]
            out_c = channels[i + 1] if i + 1 < num_blocks else 1
            blocks.extend([
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c) if out_c > 1 else nn.Identity(),
                act_fn if out_c > 1 else nn.Identity(),
            ])
        self.decoder = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.fc(z).view(B, self.init_C, 4, 4)
        x = self.decoder(x)
        H = x.shape[2]
        W = x.shape[3]
        # Resize if output spatial dimensions don't match target
        if H != self.T_out or W != self.L * self.F:
            x = F.interpolate(x, size=(self.T_out, self.L * self.F), mode="bilinear", align_corners=False)
        Y_hat = x.reshape(B, self.T_out, self.L, self.F)
        return Y_hat


# =====================================================================
#  Sinusoidal Time Embedding (from repo NoiseNet.py)
# =====================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of diffusion timesteps (similar to positional encoding).

    Maps each scalar timestep t to a vector of `dim` sine/cosine features,
    allowing the noise network to condition on the noise level.
    """

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * (-math.log(10000.0) / half_dim))
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# =====================================================================
#  Noise Estimation Network (config-driven NEN)
# =====================================================================

class _EncBlock(nn.Module):
    """One encoder block for the noise estimation network.

    Conv1D → Norm → Activation → MaxPool1d. Returns both the pooled output
    and the pre-pooled feature map (used as a skip connection).
    """

    def __init__(self, in_c: int, out_c: int, kernel_size: int, padding: int,
                 activation: str = "relu", normalization: str = "batchnorm"):
        super().__init__()
        norm_fn = _get_norm1d(normalization, out_c)
        act_fn_a = _get_activation(activation)
        # Workaround: layernorm expects (N, C, L) but BatchNorm1d also expects (N, C, L);
        # For layernorm we need to permute or use a wrapper
        if normalization == "layernorm":
            norm_fn = nn.Sequential(
                nn.LayerNorm(out_c),
            )
        # Conv1d outputs (B, C, L); BatchNorm1d/LayerNorm both work on C dim,
        # but LayerNorm in PyTorch normalizes the last dim by default.
        # We'll handle layernorm via a small wrapper that permutes.
        class _LN1d(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.ln = nn.LayerNorm(channels)
            def forward(self, x):
                return self.ln(x.transpose(1, 2)).transpose(1, 2)
        if normalization == "layernorm":
            norm_fn = _LN1d(out_c)
        else:
            norm_fn = _get_norm1d(normalization, out_c)
        self.conv_norm_act = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            norm_fn,
            act_fn_a,
        )
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        pre_pool = self.conv_norm_act(x)
        pooled = self.pool(pre_pool)
        return pooled, pre_pool


class _DecBlock(nn.Module):
    """One decoder block for the noise estimation network.

    Transposed Conv1D upsamples by 2×, then the result is concatenated
    with the corresponding skip connection from the encoder path
    (U-Net style), followed by Conv1D → Norm → Activation.
    """

    def __init__(self, in_c: int, out_c: int, skip_c: int, kernel_size: int, padding: int,
                 activation: str = "relu", normalization: str = "batchnorm"):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2)
        class _LN1d(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.ln = nn.LayerNorm(channels)
            def forward(self, x):
                return self.ln(x.transpose(1, 2)).transpose(1, 2)
        if normalization == "layernorm":
            norm_fn = _LN1d(out_c)
        else:
            norm_fn = _get_norm1d(normalization, out_c)
        act_fn = _get_activation(activation)
        self.conv_norm_act = nn.Sequential(
            nn.Conv1d(out_c + skip_c, out_c, kernel_size=kernel_size, padding=padding),
            norm_fn,
            act_fn,
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_norm_act(x)


class EnhancedNoiseNet(nn.Module):
    """1D U-Net style noise estimation network for the diffusion model.

    Takes the concatenation of [noisy latent, conditioned latent, time
    embedding] as a 1D signal and predicts the added noise. Uses
    skip-connected encoder-decoder with transposed convolutions.
    """

    def __init__(self, latent_dim: int, time_embed_dim: int = 32,
                 encoder_channels: list[int] | None = None,
                 bottleneck_channels: int = 256,
                 decoder_channels: list[int] | None = None,
                 kernel_size: int = 3,
                 activation: str = "relu",
                 normalization: str = "batchnorm"):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128]
        if decoder_channels is None:
            decoder_channels = [128, 64]
        self.input_dim = latent_dim * 2 + time_embed_dim
        self.num_blocks = len(encoder_channels)
        padding = kernel_size // 2
        act_fn = _get_activation(activation)

        self.enc_blocks = nn.ModuleList()
        prev_c = 1
        for c in encoder_channels:
            self.enc_blocks.append(_EncBlock(prev_c, c, kernel_size, padding, activation, normalization))
            prev_c = c

        self.bottleneck = nn.Sequential(
            nn.Conv1d(encoder_channels[-1], bottleneck_channels, kernel_size=kernel_size, padding=padding),
            act_fn,
        )

        self.dec_blocks = nn.ModuleList()
        prev_c = bottleneck_channels
        for i, c in enumerate(decoder_channels):
            skip_c = encoder_channels[self.num_blocks - 1 - i]
            self.dec_blocks.append(_DecBlock(prev_c, c, skip_c, kernel_size, padding, activation, normalization))
            prev_c = c

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.final = nn.Linear(decoder_channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # Add channel dimension: (B, 1, input_dim)
        x = x.unsqueeze(1)
        pre_pool = []
        h = x
        for block in self.enc_blocks:
            h, pre = block(h)
            pre_pool.append(pre)

        h = self.bottleneck(h)

        # Decode with skip connections (reverse order of pre_pool)
        for i, block in enumerate(self.dec_blocks):
            skip = pre_pool[self.num_blocks - 1 - i]
            h = block(h, skip)

        h = self.adaptive_pool(h).view(B, -1)
        return self.final(h)


# =====================================================================
#  Diffusion Model (from repo NoiseNet.py)
# =====================================================================

def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """Cosine noise schedule (cosine² ᾱ from Nichol & Dhariwal).

    Smoother than linear schedule; avoids adding too much noise too
    early in the diffusion process.
    """
    steps = T + 1
    x = np.linspace(0, T, steps)
    alpha_bar = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(beta, 1e-8, 0.999)


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    """Simple linearly increasing β schedule (Ho et al. DDPM)."""
    return np.linspace(beta_start, beta_end, T)


class DiffusionModel(nn.Module):
    """Denoising diffusion probabilistic model operating in latent space.

    Performs forward diffusion (q_sample) and reverse denoising
    (p_sample / p_sample_loop). The reverse process is conditioned on
    the latent vector `cond_z` produced by TSS-CC and uses a 1D U-Net
    (EnhancedNoiseNet) to predict the added noise.
    """

    def __init__(self, latent_dim: int, n_timestep: int, device: torch.device,
                 noise_schedule: str = "cosine",
                 nen_encoder_channels: list[int] | None = None,
                 nen_bottleneck_channels: int = 256,
                 nen_decoder_channels: list[int] | None = None,
                 nen_kernel_size: int = 3,
                 time_embed_dim: int = 32,
                 condition_proj_dim: int | None = None,
                 condition_strategy: str = "concat",
                 nen_activation: str = "relu",
                 nen_normalization: str = "batchnorm"):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_timestep = n_timestep
        self.device = device
        self.condition_strategy = condition_strategy

        self.time_embedding = SinusoidalTimeEmbedding(dim=time_embed_dim)
        # Project condition to same space before concatenation
        proj_dim = condition_proj_dim if condition_proj_dim is not None else latent_dim
        self.cond_proj = nn.Linear(latent_dim, proj_dim)
        self.noise_net = EnhancedNoiseNet(
            latent_dim=latent_dim,
            time_embed_dim=time_embed_dim,
            encoder_channels=nen_encoder_channels or [64, 128],
            bottleneck_channels=nen_bottleneck_channels,
            decoder_channels=nen_decoder_channels or [128, 64],
            kernel_size=nen_kernel_size,
            activation=nen_activation,
            normalization=nen_normalization,
        )

        # Precompute β, α, ᾱ for all timesteps
        if noise_schedule == "cosine":
            betas = cosine_beta_schedule(n_timestep)
        elif noise_schedule == "linear":
            betas = linear_beta_schedule(n_timestep)
        else:
            raise ValueError(f"Unknown noise_schedule: {noise_schedule}")
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_cumprod", torch.cumprod(self.alphas, dim=0))
        # ᾱ_{t-1} with ᾱ_0 = 1 for t=0
        self.register_buffer("alpha_cumprod_prev",
                             torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]]))

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: sample z_t = √ᾱ_t · z0 + √(1-ᾱ_t) · ε."""
        a_bar = self.alpha_cumprod[t].view(-1, 1)
        om = 1 - a_bar
        return torch.sqrt(a_bar) * z0 + torch.sqrt(om) * noise

    def forward(self, zt: torch.Tensor, cond_z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise ε from (z_t, cond_z, t)."""
        t_emb = self.time_embedding(t)
        if self.condition_strategy == "concat":
            inp = torch.cat([zt, self.cond_proj(cond_z), t_emb], dim=1)
        elif self.condition_strategy == "film":
            raise NotImplementedError("FiLM conditioning not yet implemented")
        elif self.condition_strategy == "adaln":
            raise NotImplementedError("AdaLN conditioning not yet implemented")
        else:
            raise ValueError(f"Unknown condition_strategy: {self.condition_strategy}")
        return self.noise_net(inp)

    def p_sample(self, zt: torch.Tensor, cond_z: torch.Tensor, t: torch.Tensor,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Single reverse denoising step using the analytic DDPM posterior.

        Computes the mean (A_t * z_t - B_t * ε_θ) and adds stochastic
        noise σ_t, clipping noise for t=0.
        """
        beta_t = self.betas[t].view(-1, 1)
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alpha_cumprod[t].view(-1, 1)
        alpha_bar_tm1 = self.alpha_cumprod_prev[t].view(-1, 1)
        e_pred = self.forward(zt, cond_z, t)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_alpha_bar_tm1 = torch.sqrt(alpha_bar_tm1)
        denom = 1 - alpha_bar_t
        # Posterior mean coefficients (from DDPM derivation)
        A_t = (1 / sqrt_alpha_t) * (sqrt_alpha_bar_tm1 * beta_t / denom) + (sqrt_alpha_t * (1 - alpha_bar_tm1) / denom)
        B_t = (torch.sqrt(1 - alpha_bar_t) / sqrt_alpha_t) * (sqrt_alpha_bar_tm1 * beta_t / denom)
        sigma_t = torch.sqrt((1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t)
        noise = torch.randn(zt.shape, device=zt.device, dtype=zt.dtype, generator=generator)
        mask = (t > 0).float().view(-1, 1)
        return A_t * zt - B_t * e_pred + mask * sigma_t * noise

    @torch.no_grad()
    def p_sample_loop(self, cond_z: torch.Tensor,
                      generator: torch.Generator | None = None) -> torch.Tensor:
        """Full reverse denoising chain: z_T → z_0 conditioned on cond_z."""
        B = cond_z.size(0)
        z = torch.randn(B, self.latent_dim, device=cond_z.device, dtype=cond_z.dtype,
                        generator=generator)
        for step in reversed(range(self.n_timestep)):
            t = torch.full((B,), step, device=cond_z.device, dtype=torch.long)
            z = self.p_sample(z, cond_z, t, generator=generator)
        return z

    @torch.no_grad()
    def ddim_sample_loop(self, cond_z: torch.Tensor, steps: int,
                         generator: torch.Generator | None = None) -> torch.Tensor:
        """Deterministic DDIM sampling over an evenly spaced subset of timesteps."""
        if not 1 <= steps <= self.n_timestep:
            raise ValueError(f"DDIM steps must be in [1, {self.n_timestep}], got {steps}")
        device = cond_z.device
        z = torch.randn(cond_z.size(0), self.latent_dim, device=device, dtype=cond_z.dtype,
                        generator=generator)
        schedule = np.linspace(0, self.n_timestep - 1, steps, dtype=np.int64)[::-1]
        for index, step in enumerate(schedule):
            t = torch.full((cond_z.size(0),), int(step), device=device, dtype=torch.long)
            alpha_bar = self.alpha_cumprod[t].view(-1, 1)
            eps = self.forward(z, cond_z, t)
            z0 = (z - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            previous = int(schedule[index + 1]) if index + 1 < len(schedule) else -1
            alpha_bar_previous = (
                self.alpha_cumprod[previous]
                if previous >= 0
                else torch.ones((), device=device, dtype=z.dtype)
            )
            z = torch.sqrt(alpha_bar_previous) * z0 + torch.sqrt(1 - alpha_bar_previous) * eps
        return z
