"""ResidualLinearAR with a small nonlinear correction head.

The model is ``ResidualLinearAutoregressiveForecaster`` plus an MLP that sees each channel's deviation from the lookback mean (one
series at a time, weights shared across channels, like the patch models) and predicts an extra correction for every horizon step.
The MLP's last layer starts at zero, so the initial prediction is exactly the linear model's. It exists to test whether a nonlinear
head on top of the best linear model explains the gap that nonlinear patch models show over ResidualLinearAR on some datasets.

Extra config keys (``config["model"]``, besides everything ``ResidualLinearAutoregressiveForecaster`` reads): ``mlp_hidden`` (64),
``mlp_layers`` (2, the number of hidden layers), ``mlp_dropout`` (0.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.ResidualLinearAutoregressive import ResidualLinearAutoregressiveForecaster


class ResidualNonlinearAutoregressiveForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mc = config["model"]
        self.linear = ResidualLinearAutoregressiveForecaster(config)
        length, horizon = int(mc["input_sequence_length"]), int(mc["prediction_horizon"])
        hidden, layers, dropout = int(mc.get("mlp_hidden", 64)), int(mc.get("mlp_layers", 2)), float(mc.get("mlp_dropout", 0.0))
        blocks: list[nn.Module] = []
        width = length
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        self.mlp = nn.Sequential(*blocks)
        self.out = nn.Linear(width, horizon)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)                                               # (B, H, C)
        deviations = x - x.mean(dim=1, keepdim=True)                         # (B, T, C)
        series = deviations.permute(0, 2, 1)                                 # (B, C, T): one series per channel, shared weights
        correction = self.out(self.mlp(series)).permute(0, 2, 1)             # (B, H, C)
        return base + correction

    def ridge_penalty(self) -> torch.Tensor:
        return self.linear.ridge_penalty()
