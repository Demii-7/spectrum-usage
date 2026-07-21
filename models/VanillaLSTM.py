"""
VanillaLSTM model definitions for spectrum prediction.

Architecture overview:
1. Configurable single-layer or stacked LSTM
2. Configurable hidden-state output strategy
3. Linear projection to the requested prediction horizon

The model receives input shaped as:

    (batch, time, features)

and returns:

    (batch, prediction_horizon, features)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VanillaLSTMForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        
        #Load configuration file
        model_config = config["model"]

        #Read configuration file and extract critical variables
        self.input_size = int(model_config["input_size"])
        self.hidden_size = int(model_config["hidden_size"])
        self.num_layers = int(model_config.get("num_layers", 1))
        self.bidirectional = bool(model_config.get("bidirectional", False))
        self.output_strategy = str(model_config.get("output_strategy", "final_hidden"))
        self.input_sequence_length = int(model_config["input_sequence_length"])
        self.prediction_horizon = int(model_config["prediction_horizon"])
        self.num_directions = 2 if self.bidirectional else 1
        self.dropout = float(model_config.get("dropout", 0.0))

        # Set dropout value from config based on number of stacked LSTM layers
        #dropout = float(model_config.get("dropout", 0.0)) if self.num_layers > 1 else 0.0


        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Determine which  hidden state representation to use for forecasting: the one for the last timesteps or all timesteps together
        if self.output_strategy == "final_hidden":
            head_input_dim = self.hidden_size * self.num_directions
        elif self.output_strategy == "all_hidden":
            head_input_dim = self.input_sequence_length * self.hidden_size * self.num_directions
        else:
            raise ValueError(
                f"Unsupported output_strategy {self.output_strategy!r}; use 'final_hidden' or 'all_hidden'."
            )

        # Dropout Layer between LSTM and fc
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Set up Linear network forcaster hidden_state and output dimensions 
        self.output_head = nn.Linear(head_input_dim, self.prediction_horizon * self.input_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define forward pass prediction from input data x
        
        # Ensure the incoming data is exactly 3-dimensional (Batch, Time, Features)
        if x.dim() != 3:
            raise ValueError(
                "Error! VanillaLSTM expects input shaped "
                f"(batch, time, features), got {tuple(x.shape)}"
            )
        # Verify that the sequence length matches what the model was constructed to receive
        if x.shape[1] != self.input_sequence_length:
            raise ValueError(
                f"Error! Input sequence length {x.shape[1]} != "
                f"configured length {self.input_sequence_length}"
            )
        
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"Error! Input feature count {x.shape[2]} != "
                f"configured input_size {self.input_size}"
            )
        
        # Store all hidden states for each timesteps(outputs) and the final hidden state separately
        outputs, (hidden_state, _) = self.lstm(x)

        #Choose hidden state representation based on stratetgy
        if self.output_strategy == "final_hidden":
            if self.bidirectional:
                final_features = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
            else:
                final_features = hidden_state[-1]
        else:
            final_features = outputs.reshape(outputs.size(0), -1)

        # Apply Dropout Layer
        final_features = self.dropout_layer(final_features)
        
        # Predict outputs and rehape
        projected = self.output_head(final_features)
        return projected.view(x.size(0), self.prediction_horizon, self.input_size)
