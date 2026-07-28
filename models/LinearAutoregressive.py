"""
Linear autoregressive baseline for spectrum prediction.

A simple linear model that maps the lookback window to the next timestep
independently for each feature/spatial position.  Weights are NOT shared
across positions — each frequency bin (2D) or grid cell (4D) has its own
linear mapping.

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


class LinearAutoregressiveForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        self.input_sequence_length = int(model_config["input_sequence_length"])
        self.prediction_horizon = int(model_config["prediction_horizon"])
        self.input_size = int(model_config["input_size"])
        self.ridge_alpha = float(model_config.get("ridge_alpha", 0.0))
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")

        self.weight = nn.Parameter(
            torch.empty(self.input_size, self.input_sequence_length)
        )
        self.bias = nn.Parameter(torch.zeros(self.input_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x_flat = x.reshape(B, self.input_sequence_length, self.input_size)

        # Per-feature linear map: out[b,d] = sum_t x[b,t,d] * weight[d,t] + bias[d]
        # Optimized as element-wise multiply + sum (avoids slow einsum for large D)
        out = torch.mul(x_flat, self.weight.T.unsqueeze(0)).sum(dim=1) + self.bias

        return out.reshape(B, self.prediction_horizon, *x.shape[2:])

    def ridge_penalty(self) -> torch.Tensor:
        """Return L2 regularization for AR weights without penalizing the bias."""
        return self.ridge_alpha * self.weight.square().sum(dim=1).mean()
