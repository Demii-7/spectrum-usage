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
        if self.statistic not in {"mean", "median", "train_mean", "ewma"}:
            raise ValueError(f"statistic must be 'mean', 'median', 'train_mean' or 'ewma', got {self.statistic!r}")
        # "train_mean": the training-set mean per bin (inputs are z-scored with training statistics, so it is zero in model space).
        # "ewma": exponentially weighted mean of the window, newest weight ``gain`` (a scalar, or one value per bin: the steady-state
        # gain of a per-bin local-level Kalman filter fitted on the training data), weights normalized over the window.
        gain = model_config.get("gain")
        self.register_buffer("gain", None if gain is None else torch.as_tensor(gain, dtype=torch.float32), persistent=False)
        if self.statistic == "ewma" and self.gain is None:
            raise ValueError("statistic 'ewma' needs model.gain (a scalar or one gain per bin)")
        self.window = int(model_config.get("window") or self.input_sequence_length)
        if not 1 <= self.window <= self.input_sequence_length:
            raise ValueError(f"window must be in [1, {self.input_sequence_length}], got {self.window}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # statistic over the newest ``window`` steps of the time axis, keep dim for broadcasting
        recent = x[:, -self.window:]
        if self.statistic == "train_mean":
            return torch.zeros_like(x[:, :1])
        if self.statistic == "ewma":
            gain = self.gain.to(x.device)
            if gain.dim() == 1 and gain.shape[0] != x.shape[-1]:
                raise ValueError(f"per-bin gain has {gain.shape[0]} values but the input has {x.shape[-1]} features")
            ages = torch.arange(recent.shape[1] - 1, -1, -1, device=x.device, dtype=x.dtype).view(-1, *([1] * (x.dim() - 2)))
            weights = gain * (1.0 - gain) ** ages                          # (T, ...) broadcast against (B, T, ...)
            weights = weights / weights.sum(dim=0, keepdim=True)
            return (recent * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        if self.statistic == "median":
            # lower median for an even count (torch.median), so the prediction is always an observed value
            return recent.median(dim=1, keepdim=True).values
        return recent.mean(dim=1, keepdim=True)
