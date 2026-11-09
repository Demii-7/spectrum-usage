"""RevIN + MLP: the simplest nonlinear channel-independent forecaster.

Each channel's window is instance-normalized (mean and standard deviation over the lookback, RevIN without affine terms), one MLP
(weights shared by every channel) maps the normalized window to the normalized forecast, and the statistics are put back. It has the
two ingredients PatchTST and OFA share (per-window normalization and weights shared across channels) without patching or attention.

Config keys (``config["model"]``): ``input_sequence_length``, ``prediction_horizon``, ``mlp_hidden`` (128), ``mlp_layers`` (2 hidden
layers), ``mlp_dropout`` (0.1). Input ``(B, T, C)``, output ``(B, H, C)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NORM_EPS = 1e-5


class RevINMLPForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mc = config["model"]
        self.seq_len = int(mc["input_sequence_length"])
        self.pred_len = int(mc["prediction_horizon"])
        hidden, layers, dropout = int(mc.get("mlp_hidden", 128)), int(mc.get("mlp_layers", 2)), float(mc.get("mlp_dropout", 0.1))
        blocks: list[nn.Module] = []
        width = self.seq_len
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(width, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series = x.permute(0, 2, 1)                                      # (B, C, T)
        mean = series.mean(dim=-1, keepdim=True)
        std = (series.var(dim=-1, keepdim=True, unbiased=False) + NORM_EPS).sqrt()
        forecast = self.head(self.body((series - mean) / std)) * std + mean
        return forecast.permute(0, 2, 1)
