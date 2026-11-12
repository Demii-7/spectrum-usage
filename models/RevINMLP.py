"""RevIN + MLP: the simplest nonlinear channel-independent forecaster.

Each channel's window is instance-normalized (mean and standard deviation over the lookback, RevIN without affine terms), one MLP
(weights shared by every channel) maps the normalized window to the normalized forecast, and the statistics are put back. It has the
two ingredients PatchTST and OFA share (per-window normalization and weights shared across channels) without patching or attention.

With ``channel_mixing`` true, one residual linear layer across the C channels (``input_size``, e.g. the 200 bins) is applied to the
hidden representation after the per-channel MLP body (zero-initialised, so training starts from the channel-independent model). With it
off no parameter is created and the state_dict is the original one.

Config keys (``config["model"]``): ``input_sequence_length``, ``prediction_horizon``, ``mlp_hidden`` (128), ``mlp_layers`` (2 hidden
layers), ``mlp_dropout`` (0.1), ``channel_mixing`` (false). Input ``(B, T, C)``, output ``(B, H, C)``.
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
        self.mixer = None
        if bool(mc.get("channel_mixing", False)):
            channels = int(mc["input_size"])
            self.mixer = nn.Linear(channels, channels)
            nn.init.zeros_(self.mixer.weight)
            nn.init.zeros_(self.mixer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series = x.permute(0, 2, 1)                                      # (B, C, T)
        mean = series.mean(dim=-1, keepdim=True)
        std = (series.var(dim=-1, keepdim=True, unbiased=False) + NORM_EPS).sqrt()
        hidden = self.body((series - mean) / std)                        # (B, C, hidden)
        if self.mixer is not None:                                       # mix across channels, residually
            hidden = hidden + self.mixer(hidden.transpose(1, 2)).transpose(1, 2)
        forecast = self.head(hidden) * std + mean
        return forecast.permute(0, 2, 1)
