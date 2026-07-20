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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean over time axis, keep dim for broadcasting
        return x.mean(dim=1, keepdim=True)
