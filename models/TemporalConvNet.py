"""Causal temporal convolution forecaster based on Sen et al. (NeurIPS 2019)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Conv1d):
    """One-dimensional convolution padded only with past values."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(x, (self.left_padding, 0)))


class TemporalConvNetForecaster(nn.Module):
    """Forecast with either independent-series or joint-feature temporal convolutions."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        self.input_size = int(model_config["input_size"])
        self.input_sequence_length = int(model_config["input_sequence_length"])
        self.prediction_horizon = int(model_config.get("prediction_horizon", 1))
        self.feature_mode = str(model_config.get("feature_mode", "independent")).lower()
        if self.feature_mode not in {"independent", "joint"}:
            raise ValueError("feature_mode must be either 'independent' or 'joint'")
        if self.prediction_horizon != 1:
            raise ValueError("TemporalConvNet predicts one step; use shared autoregressive rollout")

        hidden_channels = [int(value) for value in model_config.get("hidden_channels", [32] * 6)]
        kernel_size = int(model_config.get("kernel_size", 2))
        dropout = float(model_config.get("dropout", 0.0))
        if not hidden_channels or any(value <= 0 for value in hidden_channels):
            raise ValueError("hidden_channels must contain positive integers")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")

        input_channels = 1 if self.feature_mode == "independent" else self.input_size
        output_channels = 1 if self.feature_mode == "independent" else self.input_size
        channels = [input_channels, *hidden_channels]
        layers: list[nn.Module] = []
        for index, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            dilation = 2**index
            layers.append(CausalConv1d(in_channels, out_channels, kernel_size, dilation))
            layers.extend((nn.ReLU(), nn.Dropout(dropout)))
        layers.append(nn.Conv1d(hidden_channels[-1], output_channels, kernel_size=1))
        self.network = nn.Sequential(*layers)

        if bool(model_config.get("leveled_init", True)):
            self._leveled_init(kernel_size)

    def _leveled_init(self, kernel_size: int) -> None:
        convolutions = [module for module in self.network if isinstance(module, nn.Conv1d)]
        with torch.no_grad():
            for convolution in convolutions:
                convolution.bias.zero_()
                convolution.weight.fill_(1.0 / (kernel_size * convolution.in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TemporalConvNet expects (batch, time, features), got {tuple(x.shape)}")
        if x.shape[1:] != (self.input_sequence_length, self.input_size):
            raise ValueError(
                "TemporalConvNet input dimensions do not match configuration: "
                f"got {tuple(x.shape[1:])}, expected "
                f"({self.input_sequence_length}, {self.input_size})"
            )

        batch_size = x.shape[0]
        series = x.transpose(1, 2)
        if self.feature_mode == "independent":
            series = series.reshape(batch_size * self.input_size, 1, -1)
            prediction = self.network(series)[:, :, -1]
            return prediction.reshape(batch_size, 1, self.input_size)

        prediction = self.network(series)[:, :, -1]
        return prediction.unsqueeze(1)
