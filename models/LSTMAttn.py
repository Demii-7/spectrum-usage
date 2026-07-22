"""Joint multi-channel, multi-step LSTM forecaster with additive attention."""

from __future__ import annotations

import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_size: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(hidden_size, attention_size, bias=False)
        self.decoder_projection = nn.Linear(hidden_size, attention_size, bias=False)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(
        self, encoder_states: torch.Tensor, decoder_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energies = self.score(
            torch.tanh(
                self.encoder_projection(encoder_states)
                + self.decoder_projection(decoder_state).unsqueeze(1)
            )
        ).squeeze(-1)
        weights = torch.softmax(energies, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_states).squeeze(1)
        return context, weights


class LSTMAttnForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        self.input_size = int(model_config["input_size"])
        self.input_sequence_length = int(model_config["input_sequence_length"])
        self.prediction_horizon = int(model_config["prediction_horizon"])
        hidden_size = int(model_config.get("hidden_size", 128))
        num_layers = int(model_config.get("num_layers", 1))
        dropout = float(model_config.get("dropout", 0.0))
        attention_size = int(model_config.get("attention_size", hidden_size))

        self.encoder = nn.LSTM(
            self.input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = AdditiveAttention(hidden_size, attention_size)
        self.decoder = nn.LSTMCell(self.input_size + hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size + hidden_size, self.input_size)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"LSTMAttn expects (batch, time, features), got {tuple(x.shape)}")
        if x.shape[1:] != (self.input_sequence_length, self.input_size):
            raise ValueError(
                f"LSTMAttn got dimensions {tuple(x.shape[1:])}; expected "
                f"({self.input_sequence_length}, {self.input_size})"
            )

        encoder_states, (hidden, cell) = self.encoder(x)
        decoder_hidden = hidden[-1]
        decoder_cell = cell[-1]
        previous = x[:, -1]
        predictions = []
        attention_weights = []
        for _ in range(self.prediction_horizon):
            context, weights = self.attention(encoder_states, decoder_hidden)
            decoder_hidden, decoder_cell = self.decoder(
                torch.cat((previous, context), dim=-1),
                (decoder_hidden, decoder_cell),
            )
            prediction = self.output_projection(
                torch.cat((self.dropout(decoder_hidden), context), dim=-1)
            )
            predictions.append(prediction)
            attention_weights.append(weights)
            previous = prediction

        self.last_attention_weights = torch.stack(attention_weights, dim=1)
        return torch.stack(predictions, dim=1)
