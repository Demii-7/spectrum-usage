from __future__ import annotations

import torch
import torch.nn as nn


class VanillaLSTMForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        window_config = config["windowing"]

        self.input_size = int(model_config["input_size"])
        self.hidden_size = int(model_config["hidden_size"])
        self.num_layers = int(model_config.get("num_layers", 1))
        self.bidirectional = bool(model_config.get("bidirectional", False))
        self.output_strategy = str(model_config.get("output_strategy", "final_hidden"))
        self.input_sequence_length = int(window_config["input_sequence_length"])
        self.prediction_horizon = int(window_config["prediction_horizon"])
        self.num_directions = 2 if self.bidirectional else 1

        dropout = float(model_config.get("dropout", 0.0)) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        if self.output_strategy == "final_hidden":
            head_input_dim = self.hidden_size * self.num_directions
        elif self.output_strategy == "all_hidden":
            head_input_dim = self.input_sequence_length * self.hidden_size * self.num_directions
        else:
            raise ValueError(
                f"Unsupported output_strategy {self.output_strategy!r}; use 'final_hidden' or 'all_hidden'."
            )

        self.output_head = nn.Linear(head_input_dim, self.prediction_horizon * self.input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, (hidden_state, _) = self.lstm(x)
        if self.output_strategy == "final_hidden":
            if self.bidirectional:
                final_features = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
            else:
                final_features = hidden_state[-1]
        else:
            final_features = outputs.reshape(outputs.size(0), -1)

        projected = self.output_head(final_features)
        return projected.view(x.size(0), self.prediction_horizon, self.input_size)
