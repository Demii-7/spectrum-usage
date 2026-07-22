"""Dependency-free per-series ARIMA baseline using conditional least squares."""

from __future__ import annotations

import torch
import torch.nn as nn


class ARIMAForecaster(nn.Module):
    """Fit an ARIMA(p,d,0) model independently to every lookback window."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        self.input_size = int(model_config["input_size"])
        self.input_sequence_length = int(model_config["input_sequence_length"])
        self.prediction_horizon = int(model_config["prediction_horizon"])
        self.p = int(model_config.get("p", 1))
        self.d = int(model_config.get("d", 1))
        self.q = int(model_config.get("q", 0))
        self.ridge = float(model_config.get("ridge", 1e-6))
        if self.p < 0 or self.d < 0 or self.q < 0:
            raise ValueError("ARIMA orders p, d, and q must be non-negative")
        if self.q != 0:
            raise ValueError("This dependency-free ARIMA baseline currently supports q=0")
        if self.p == 0 and self.d == 0:
            raise ValueError("At least one of p or d must be positive")
        if self.input_sequence_length - self.d <= self.p:
            raise ValueError("Lookback is too short for the configured ARIMA order")

    @staticmethod
    def _difference(values: torch.Tensor, order: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        anchors = []
        differenced = values
        for _ in range(order):
            anchors.append(differenced[-1])
            differenced = torch.diff(differenced)
        return differenced, anchors

    @staticmethod
    def _integrate(value: torch.Tensor, anchors: list[torch.Tensor]) -> torch.Tensor:
        for index in range(len(anchors) - 1, -1, -1):
            value = anchors[index] + value
            anchors[index] = value
        return value

    def _forecast_series(self, values: torch.Tensor) -> torch.Tensor:
        history, anchors = self._difference(values, self.d)
        if self.p:
            rows = torch.stack(
                [history[index - self.p:index].flip(0) for index in range(self.p, history.numel())]
            )
            target = history[self.p:]
            design = torch.cat((torch.ones_like(target).unsqueeze(1), rows), dim=1)
        else:
            target = history
            design = torch.ones_like(target).unsqueeze(1)
        if self.ridge > 0:
            identity = torch.eye(design.shape[1], dtype=values.dtype, device=values.device)
            design = torch.cat((design, self.ridge**0.5 * identity), dim=0)
            target = torch.cat((target, target.new_zeros(design.shape[1])))
        coefficients = torch.linalg.lstsq(design, target).solution

        forecasts = []
        ar_history = list(history.unbind())
        for _ in range(self.prediction_horizon):
            prediction = coefficients[0]
            if self.p:
                lags = torch.stack(ar_history[-self.p:]).flip(0)
                prediction = prediction + coefficients[1:] @ lags
            ar_history.append(prediction)
            forecasts.append(self._integrate(prediction, anchors))
        return torch.stack(forecasts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ARIMA expects (batch, time, features), got {tuple(x.shape)}")
        if x.shape[1:] != (self.input_sequence_length, self.input_size):
            raise ValueError(
                f"ARIMA got dimensions {tuple(x.shape[1:])}; expected "
                f"({self.input_sequence_length}, {self.input_size})"
            )
        forecasts = [
            [self._forecast_series(x[batch, :, feature]) for feature in range(self.input_size)]
            for batch in range(x.shape[0])
        ]
        return torch.stack([torch.stack(item, dim=1) for item in forecasts])
