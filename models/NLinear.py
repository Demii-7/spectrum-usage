"""NLinear: a linear forecaster with level normalization (Zeng et al., "Are transformers effective for time series forecasting?").

Each channel's window has its last value subtracted, one linear map over time (weights shared by every channel) predicts the future
deviations, and the last value is added back. The simplest model that is channel-independent and level-aware at the same time.

Config keys (``config["model"]``): ``input_sequence_length``, ``prediction_horizon``. Input ``(B, T, C)``, output ``(B, H, C)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NLinearForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mc = config["model"]
        self.seq_len = int(mc["input_sequence_length"])
        self.pred_len = int(mc["prediction_horizon"])
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        # start as persistence of the last value: zero deviation forecast
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, :]                                              # (B, 1, C)
        deviation = self.linear((x - last).permute(0, 2, 1)).permute(0, 2, 1)
        return deviation + last
