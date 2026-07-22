"""Residual VanillaLSTM model for spectrum prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.LookbackMean import LookbackMeanForecaster
from models.VanillaLSTM import VanillaLSTMForecaster


class ResidualVanillaLSTMForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.baseline = LookbackMeanForecaster(config)
        self.residual = VanillaLSTMForecaster(config)

        if self.residual.num_layers != 1:
            raise ValueError("ResidualVanillaLSTM requires num_layers=1")

        nn.init.zeros_(self.residual.output_head.weight)
        nn.init.zeros_(self.residual.output_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline = self.baseline(x)
        centered = x - baseline
        return baseline + self.residual(centered)
