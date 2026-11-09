"""PatchTST (Nie et al., ICLR 2023), channel-independent, in the repo's model interface.

Each channel (here one frequency bin of one receiver) is a separate univariate series that shares all weights: the series is
instance-normalized (RevIN without affine terms), cut into overlapping patches, embedded, passed through a transformer encoder,
flattened and mapped to the forecast by one linear head, then de-normalized.

Config keys (``config["model"]``): ``input_sequence_length``, ``prediction_horizon``, ``patch_length`` (16), ``patch_stride`` (8),
``d_model`` (128), ``n_heads`` (4), ``e_layers`` (2), ``d_ff`` (256), ``dropout`` (0.1). Input ``(B, T, C)``, output ``(B, H, C)``.

Ablation switches (defaults give the standard model): ``instance_norm`` (True; False feeds the series unnormalized) and ``channel_mode``
("independent", the default; "joint" makes every patch token span all channels and needs ``input_size``, like the paper's all-channel
arrangement). ``patch_length`` 1 with ``patch_stride`` 1 removes patching (one token per time step).
"""

from __future__ import annotations

import torch
import torch.nn as nn

NORM_EPS = 1e-5


class PatchTSTForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mc = config["model"]
        self.seq_len = int(mc["input_sequence_length"])
        self.pred_len = int(mc["prediction_horizon"])
        self.patch_len = int(mc.get("patch_length", 16))
        self.stride = int(mc.get("patch_stride", 8))
        d_model = int(mc.get("d_model", 128))
        n_heads = int(mc.get("n_heads", 4))
        layers = int(mc.get("e_layers", 2))
        d_ff = int(mc.get("d_ff", 256))
        dropout = float(mc.get("dropout", 0.1))
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        if self.patch_len > self.seq_len:
            raise ValueError("patch_length exceeds the lookback")
        self.instance_norm = bool(mc.get("instance_norm", True))
        self.channel_mode = str(mc.get("channel_mode", "independent")).lower()
        if self.channel_mode not in ("independent", "joint"):
            raise ValueError(f"channel_mode must be independent|joint, got {self.channel_mode!r}")
        self.channels = int(mc.get("input_size", 1)) if self.channel_mode == "joint" else 1
        self.pad = nn.ReplicationPad1d((0, self.stride))          # as in the reference: pad the end by one stride
        self.num_patches = (self.seq_len + self.stride - self.patch_len) // self.stride + 1
        self.embed = nn.Linear(self.patch_len * self.channels, d_model)
        self.position = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(self.num_patches * d_model, self.pred_len * self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        series = x.permute(0, 2, 1).reshape(batch * channels, length)          # (B*C, T)
        mean = series.mean(dim=1, keepdim=True)
        std = (series.var(dim=1, keepdim=True, unbiased=False) + NORM_EPS).sqrt()
        if self.instance_norm:
            series = (series - mean) / std
        patches = self.pad(series.unsqueeze(1)).squeeze(1).unfold(-1, self.patch_len, self.stride)   # (B*C, N, P)
        if self.channel_mode == "joint":
            if channels != self.channels:
                raise ValueError(f"joint mode was built for {self.channels} channels, got {channels}")
            patches = patches.reshape(batch, channels, self.num_patches, self.patch_len).permute(0, 2, 3, 1)
            patches = patches.reshape(batch, self.num_patches, self.patch_len * channels)                # (B, N, P*C)
            hidden = self.encoder(self.dropout(self.embed(patches) + self.position))
            forecast = self.head(hidden.flatten(1)).reshape(batch, self.pred_len, channels)              # (B, H, C)
            if self.instance_norm:
                forecast = forecast * std.reshape(batch, channels).unsqueeze(1) + mean.reshape(batch, channels).unsqueeze(1)
            return forecast
        hidden = self.encoder(self.dropout(self.embed(patches) + self.position))                      # (B*C, N, d)
        forecast = self.head(hidden.flatten(1))                                                      # (B*C, H)
        if self.instance_norm:
            forecast = forecast * std + mean
        return forecast.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
