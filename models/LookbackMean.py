"""
Lookback-mean baseline for spectrum prediction.

A parameter-free baseline that predicts the mean of the lookback window.
Works for any input shape — the mean is taken over the time dimension
(dim=1), leaving all other dimensions intact.

Expected input shapes:
    1D: (batch, time, 1)         single frequency bin
    2D: (batch, time, features)  time x frequency
    4D: (batch, time, C, H, W)   spatial map

Output shapes:
    (batch, 1, ...)   single-step prediction (prediction_horizon=1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LookbackMeanForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        self.prediction_horizon = int(model_config["prediction_horizon"])
        self.input_sequence_length = int(model_config["input_sequence_length"])
        # Optional variants; the defaults (mean of the whole window) are the original model.
        self.statistic = str(model_config.get("statistic", "mean")).lower()
        if self.statistic not in {"mean", "median"}:
            raise ValueError(f"statistic must be 'mean' or 'median', got {self.statistic!r}")
        self.window = int(model_config.get("window") or self.input_sequence_length)
        if not 1 <= self.window <= self.input_sequence_length:
            raise ValueError(f"window must be in [1, {self.input_sequence_length}], got {self.window}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # statistic over the newest ``window`` steps of the time axis, keep dim for broadcasting
        recent = x[:, -self.window:]
        if self.statistic == "median":
            # lower median for an even count (torch.median), so the prediction is always an observed value
            return recent.median(dim=1, keepdim=True).values
        return recent.mean(dim=1, keepdim=True)
