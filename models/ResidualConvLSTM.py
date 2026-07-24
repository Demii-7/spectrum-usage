"""Residual ConvLSTM model for spatiotemporal spectrum prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.ConvLSTM import ConvLSTMForecaster
from models.LookbackMean import LookbackMeanForecaster


class ResidualConvLSTMForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.baseline = LookbackMeanForecaster(config)
        self.residual = ConvLSTMForecaster(config)

        output_layers = [
            module
            for module in self.residual.output_head.modules()
            if isinstance(module, nn.Conv2d)
        ]
        output_layer = output_layers[-1]
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lookback_mean = self.baseline(x)
        centered = x - lookback_mean
        baseline = lookback_mean.expand(
            -1, self.residual.prediction_horizon, -1, -1, -1
        )
        return baseline + self.residual(centered)
