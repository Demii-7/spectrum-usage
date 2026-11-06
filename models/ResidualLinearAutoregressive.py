"""Residual linear autoregressive model for spectrum prediction.

The model predicts a correction to a baseline from recent deviations around the lookback mean. Its residual head
starts at zero, so the initial prediction is exactly the baseline.

With ``prediction_horizon > 1`` the model is direct multi-step: the baseline is broadcast (or computed) over all
horizon steps and the residual head predicts one correction per step.

Optional variants (model config keys; the defaults reproduce the original model bit for bit):

``baseline``      "mean" (default): the lookback mean.
                  "seasonal": for output step h the mean of the window values at the same time of day, i.e. at
                  positions L + h - 1 - k * seasonal_period inside the window (k = 1, 2, ...); steps with no such
                  position fall back to the lookback mean.
                  "last": persistence, the last observation of the window repeated over the horizon. The correction
                  head then sees the window centred on that last observation (x - x_last) instead of on the mean, so
                  the model is still level-shift equivariant and starts as exactly persistence.
``window_norm``   "none" (default), or "std": the deviations are divided by the window's standard deviation and the
                  head's output is multiplied back by it (instance normalization of the correction, as in RevIN
                  without the affine terms). The model is then equivariant to affine rescaling of the window.
``seasonal_period``  steps per day for the seasonal baseline (default 96, i.e. 15-minute data).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.LinearAutoregressive import LinearAutoregressiveForecaster
from models.LookbackMean import LookbackMeanForecaster

NORM_EPS = 1e-3


def seasonal_matrix(length: int, horizon: int, period: int) -> torch.Tensor:
    """(horizon, length) averaging weights: row h-1 averages the window positions L + h - 1 - k * period."""
    matrix = torch.zeros(horizon, length)
    for h in range(1, horizon + 1):
        positions = [length + h - 1 - period * k for k in range(1, length // period + 2)]
        positions = [p for p in positions if 0 <= p < length]
        if positions:
            matrix[h - 1, positions] = 1.0 / len(positions)
        else:                                  # no same-time-of-day value in the window: use the lookback mean
            matrix[h - 1, :] = 1.0 / length
    return matrix


class ResidualLinearAutoregressiveForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        self.baseline_kind = str(model_config.get("baseline", "mean")).lower()
        self.window_norm = str(model_config.get("window_norm", "none")).lower()
        if self.baseline_kind not in {"mean", "seasonal", "last"}:
            raise ValueError(f"baseline must be 'mean', 'seasonal' or 'last', got {self.baseline_kind!r}")
        if self.window_norm not in {"none", "std"}:
            raise ValueError(f"window_norm must be 'none' or 'std', got {self.window_norm!r}")
        self.baseline = LookbackMeanForecaster(config)
        self.residual = LinearAutoregressiveForecaster(config)

        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

        length = int(model_config["input_sequence_length"])
        horizon = int(model_config["prediction_horizon"])
        if self.baseline_kind == "seasonal":
            period = int(model_config.get("seasonal_period", 96))
            self.register_buffer("seasonal_weights", seasonal_matrix(length, horizon, period), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.baseline(x)                                          # (B, 1, ...), the lookback mean
        if self.baseline_kind == "seasonal":
            batch, length = x.shape[0], x.shape[1]
            flat = x.reshape(batch, length, -1)
            base = torch.einsum("hl,bld->bhd", self.seasonal_weights, flat)
            base = base.reshape(batch, self.seasonal_weights.shape[0], *x.shape[2:])
        elif self.baseline_kind == "last":
            anchor = x[:, -1:]                                           # (B, 1, ...), the last observation
            base = anchor.expand(-1, self.residual.prediction_horizon, *([-1] * (x.dim() - 2)))
            mean = anchor                                                # centre the correction input on it
        else:
            base = mean
        deviations = x - mean
        if self.window_norm == "std":
            scale = (x.var(dim=1, keepdim=True, unbiased=False) + NORM_EPS ** 2).sqrt()
            return base + scale * self.residual(deviations / scale)
        return base + self.residual(deviations)

    def ridge_penalty(self) -> torch.Tensor:
        return self.residual.ridge_penalty()
