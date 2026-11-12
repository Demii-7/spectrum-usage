"""iTransformer (Liu et al., ICLR 2024) for the direct-model path: (B, T, C) -> (B, H, C).

A self-contained port of the Time-Series-Library implementation (pinned commit 4e938a1, ``models/iTransformer.py`` with
``DataEmbedding_inverted``, ``Encoder``/``EncoderLayer`` and ``FullAttention``), keeping its parameter names so a state_dict loads into the
library's ``Model``. Each channel (here a frequency bin) is one token: its whole lookback is embedded by ``Linear(T, d_model)``, attention
runs across the channels, and ``Linear(d_model, H)`` projects each token to its forecast. Normalization is the library's non-stationary
one (per-channel mean and standard deviation over the lookback, restored afterwards). No time-mark features are used.

Config keys (``config["model"]``): ``input_sequence_length``, ``prediction_horizon``, ``d_model`` (128), ``n_heads`` (8), ``e_layers`` (2),
``d_ff`` (256), ``dropout`` (0.1). ``input_size`` is not needed by the network (any number of channels works). Activation is GELU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_EPS = 1e-5


class DataEmbeddingInverted(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # (B, T, C) -> (B, C, d_model)
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))


class FullAttention(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):                       # (B, L, H, E)
        scale = 1.0 / math.sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attn, values).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int) -> None:
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        S, H = keys.shape[1], self.n_heads
        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        return self.out_projection(self.inner_attention(q, k, v).view(B, L, -1))


class EncoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(x, x, x))
        y = x = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers: list[nn.Module], norm_layer: nn.Module) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.attn_layers:
            x = layer(x)
        return self.norm(x)


class ITransformerForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mc = config["model"]
        self.seq_len = int(mc["input_sequence_length"])
        self.pred_len = int(mc["prediction_horizon"])
        d_model, n_heads = int(mc.get("d_model", 128)), int(mc.get("n_heads", 8))
        d_ff, e_layers, dropout = int(mc.get("d_ff", 256)), int(mc.get("e_layers", 2)), float(mc.get("dropout", 0.1))
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.enc_embedding = DataEmbeddingInverted(self.seq_len, d_model, dropout)
        self.encoder = Encoder([EncoderLayer(AttentionLayer(FullAttention(dropout), d_model, n_heads), d_model, d_ff, dropout)
                                for _ in range(e_layers)], nn.LayerNorm(d_model))
        self.projection = nn.Linear(d_model, self.pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + NORM_EPS)
        out = self.projection(self.encoder(self.enc_embedding(x / std)))   # (B, C, H)
        return out.permute(0, 2, 1) * std + mean
