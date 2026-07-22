"""
TimeRAN forecasting model.

This module wraps the pretrained MOMENT forecasting pipeline so that it follows
the common interface used by the integrated training and evaluation scripts.

Integrated pipeline format:
    input:  (batch, lookback, features)
    output: (batch, prediction_horizon, features)

MOMENT format:
    input:  (batch, channels, sequence_length)
    output: (batch, channels, forecast_horizon)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch
import torch.nn as nn

from momentfm import MOMENTPipeline

# Dictionary mapping friendly size names to their official Hugging Face repository identifiers
VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


class TimeRANForecaster(nn.Module):
    """
    MOMENT-based forecasting model used by TimeRAN.

    The pretrained MOMENT encoder and embedder are frozen by default while the
    forecasting head remains trainable.
    """
    
    def __init__(self, config: dict[str, Any]):
        # Initialize the parent torch.nn.Module class
        super().__init__()
        
        # Extract the model-specific settings sub-dictionary from the configuration
        model_cfg = config["model"]
        
        # Parse and store the historical window length (lookback) as an integer
        self.input_sequence_length = int(
            model_cfg["input_sequence_length"]
        )
        # Parse and store the number of future timesteps to predict as an integer
        self.prediction_horizon = int(
            model_cfg["prediction_horizon"]
        )
        # Parse and store the number of input time-series features (channels) as an integer
        self.input_size = int(
            model_cfg["input_size"]
        )
        # Get the chosen model size, fallback to "base" if not provided, and convert to lowercase
        checkpoint_size = str(
            model_cfg.get("checkpoint_size", "base")
        ).lower()
        
        # Retrieve the corresponding Hugging Face model repository path
        pretrained_model_name = VARIANT_TO_MODEL.get(
            checkpoint_size
        )
        
        # Raise an informative error if the user provided an unsupported model size string
        if pretrained_model_name is None:
            raise ValueError(
                "Unknown TimeRAN checkpoint_size "
                f"{checkpoint_size!r}. Expected one of "
                f"{sorted(VARIANT_TO_MODEL)}."
            )
            
        # Download and instantiate the pretrained MOMENT pipeline with configuration parameters
        self.moment = MOMENTPipeline.from_pretrained(
            pretrained_model_name,
            model_kwargs={
                "task_name": "forecasting",                            # Instruct MOMENT to configure itself for forecasting tasks
                "forecast_horizon": self.prediction_horizon,           # Set target forecasting steps
                "seq_len": self.input_sequence_length,                 # Set historical context window length
                "freeze_encoder": bool(                                # Determine whether to freeze core representation weights
                    model_cfg.get("freeze_encoder", True)
                ),
                "freeze_embedder": bool(                               # Determine whether to freeze patch tokenization weights
                    model_cfg.get("freeze_embedder", True)
                ),
                "freeze_head": bool(                                   # Determine whether to keep the linear projection head trainable
                    model_cfg.get("freeze_head", False)
                ),
            },
        )

        # Explicitly initialize the inner weights and configurations of the MOMENT pipeline
        self.moment.init()

        if bool(model_cfg.get("use_timeran_checkpoint", True)):
            configured_path = model_cfg.get("timeran_checkpoint_path")
            checkpoint_path = (
                Path(configured_path).expanduser()
                if configured_path
                else Path(__file__).resolve().parents[1]
                / "training"
                / "TimeRAN"
                / "checkpoints"
                / checkpoint_size
                / f"TimeRAN_{checkpoint_size}.pth"
            )
            if checkpoint_path.is_file():
                state_dict = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(state_dict, dict):
                    raise ValueError(
                        "TimeRAN checkpoint must contain a state dictionary, "
                        f"got {type(state_dict).__name__} from {checkpoint_path}."
                    )
                if any(key.startswith("module.") for key in state_dict):
                    state_dict = {
                        key.removeprefix("module."): value
                        for key, value in state_dict.items()
                    }

                # The MOMENT forecasting head depends on this run's horizon.
                state_dict.pop("head.linear.weight", None)
                state_dict.pop("head.linear.bias", None)
                self.moment.load_state_dict(state_dict, strict=False)
            else:
                warnings.warn(
                    f"TimeRAN checkpoint not found at {checkpoint_path}; "
                    "using raw MOMENT weights. Set use_timeran_checkpoint: false "
                    "to request raw MOMENT explicitly.",
                    stacklevel=2,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forecast all configured future timesteps.

        Parameters
        ----------
        x:
            Input tensor shaped:

                (batch, lookback, features)

        Returns
        -------
        torch.Tensor
            Forecast tensor shaped:

                (batch, prediction_horizon, features)
        """
        # Ensure the incoming data is exactly 3-dimensional (Batch, Time, Features)
        if x.ndim != 3:
            raise ValueError(
                "TimeRAN expects input shaped "
                "(batch, lookback, features), "
                f"got {tuple(x.shape)}."
            )
        # Unpack the exact structural dimensions of the batch
        batch_size, lookback, input_size = x.shape
        
        # Verify that the sequence length matches what the model was constructed to receive
        if lookback != self.input_sequence_length:
            raise ValueError(
                "TimeRAN received lookback length "
                f"{lookback}, but the model was configured for "
                f"{self.input_sequence_length}."
            )
        # Verify that the number of time-series features matches the configuration
        if input_size != self.input_size:
            raise ValueError(
                "TimeRAN received "
                f"{input_size} features, but the model was "
                f"configured for {self.input_size}."
            )

        # Pivot data: from (batch, time, features) to MOMENT's format: (batch, features/channels, time)
        # .contiguous() forces memory optimization after changing array layouts
        moment_input = x.transpose(1, 2).contiguous()
        
        # Create an all-ones binary mask indicating that none of the time steps are padded/missing
        # Ensures mask matches the data type and hardware device (CPU/GPU) of the input tensor
        input_mask = torch.ones(
            batch_size,
            lookback,
            dtype=moment_input.dtype,
            device=moment_input.device,
        )
        
        # Execute forward pass through the MOMENT underlying neural network architecture
        model_output = self.moment(
            x_enc=moment_input,
            input_mask=input_mask,
        )
        # Extract the specialized forecast tensor object from the multi-attribute model output
        forecast = model_output.forecast

        # Confirm that the output of MOMENT contains exactly 3 dimensional axes
        if forecast.ndim != 3:
            raise ValueError(
                "MOMENT returned an unexpected forecast shape: "
                f"{tuple(forecast.shape)}."
            )

        # Pivot data back: from MOMENT's (batch, features, forecast_horizon) 
        # to the pipeline's expected (batch, prediction_horizon, features)
        forecast = forecast.transpose(1, 2).contiguous()
        
        # Define expected shape configuration tuple for post-execution verification
        expected_shape = (
            batch_size,
            self.prediction_horizon,
            self.input_size,
        )
        # Guard check ensuring the resulting shape perfectly conforms to expected pipeline specifications
        if tuple(forecast.shape) != expected_shape:
            raise ValueError(
                "TimeRAN forecast shape does not match the "
                f"expected shape. Expected {expected_shape}, "
                f"got {tuple(forecast.shape)}."
            )
        # Return the cleanly reformatted predictions array back to the wrapper pipeline execution engine
        return forecast
