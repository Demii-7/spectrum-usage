"""Residual linear autoregressive model for spectrum prediction.

The model predicts a correction to the lookback-mean baseline from recent
deviations around that mean. Its residual head starts at zero, so the initial
prediction is exactly the lookback-mean prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.LinearAutoregressive import LinearAutoregressiveForecaster
from models.LookbackMean import LookbackMeanForecaster


class ResidualLinearAutoregressiveForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.baseline = LookbackMeanForecaster(config)
        self.residual = LinearAutoregressiveForecaster(config)

        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline = self.baseline(x)
        deviations = x - baseline
        return baseline + self.residual(deviations)

    def ridge_penalty(self) -> torch.Tensor:
        return self.residual.ridge_penalty()
