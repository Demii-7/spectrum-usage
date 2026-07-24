"""
Shared forecasting policies for training, validation, and evaluation.

This module contains the single forecasting implementation used throughout the
integrated spectrum-prediction pipeline. It supports both vector-based and
map-based models as long as the model follows the common time-first batch
interface.

Two forecasting strategies are provided:

Teacher-forced rollout:
    Used with one-step models when ground-truth future targets are available.
    The model predicts one timestep at a time, while each input window is
    updated using the corresponding true target timestep.

Autoregressive rollout:
    Used with one-step models when targets are unavailable. The model predicts
    one timestep at a time, and each prediction is appended to the window as
    input for the next step.

Direct multi-step forecast:
    Used when the model's configured prediction horizon equals the requested
    rollout horizon. The model performs one forward pass and returns the full
    sequence directly.

Expected model interfaces:

    CSV:
        input:  (B, T, F)
        output: (B, prediction_horizon, F)

    Map:
        input:  (B, T, F, H, W)
        output: (B, prediction_horizon, F, H, W)

The public forecast function validates that the model uses one of the supported
policies:

- prediction_horizon == 1; or
- prediction_horizon == rollout_horizon.

This module is the authoritative forecasting implementation. Training,
validation, and evaluation should call forecast rather than reimplementing
rollout logic.
"""


from __future__ import annotations

import torch
import torch.nn as nn

def teacher_forced_rollout( model: nn.Module, x: torch.Tensor, y: torch.Tensor, ) -> torch.Tensor:
    """Generate the full forecast using ground truth to update the window."""

    predictions = []
    window = x

    for step in range(y.shape[1]):
        # Model predicts one future timestep.
        output = model(window)

        if output.shape[1] != 1:
            raise ValueError(
                "Error! Teacher-forced rollout requires prediction_horizon=1"
            )

        # Remove the one-step time dimension.
        next_prediction = output[:, 0]

        predictions.append(next_prediction)

        # Teacher forcing:
        # append the real target for this step, not the prediction.
        true_next_timestep = y[:, step].unsqueeze(1)

        window = torch.cat(
            [window[:, 1:], true_next_timestep],
            dim=1,
        )
    return torch.stack(predictions, dim=1)

    
def autoregressive_rollout( model: nn.Module, x: torch.Tensor, rollout_horizon: int, ) -> torch.Tensor:
    """Repeatedly use a one-step model to predict multiple future steps."""

    predictions = []

    # Begin with the ground-truth lookback window.
    window = x

    for _ in range(rollout_horizon):
        
        # Preducit next timestep
        # Expected output:
        # CSV: (batch, 1, frequency)
        # Map:     (batch, 1, frequency, height, width)
        
        output = model(window)

        if output.shape[1] != 1:
            raise ValueError(
                "Error! Autoregressive rollout requires the model's "
                "prediction_horizon to be 1"
            )

        # Remove the one-step time dimension.
        next_prediction = output[:, 0]

        predictions.append(next_prediction)

        # Add the time dimension back before appending to the window.
        next_timestep = next_prediction.unsqueeze(1)

        # Remove the oldest input and append the prediction.
        window = torch.cat(
            [window[:, 1:], next_timestep],
            dim=1,
        )
        
    # Restore the complete forecast time dimension.
    return torch.stack(predictions, dim=1)


def forecast(
    model: nn.Module,
    x: torch.Tensor,
    prediction_horizon: int,
    rollout_horizon: int,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply the exact shared forecasting policy.

    One-step model:
        teacher-forced rollout when targets are provided;
        autoregressive rollout otherwise.

    Multi-step model:
        one direct forward call.
    """

    if prediction_horizon == 1:
        if targets is not None:
            pred = teacher_forced_rollout(
                model,
                x,
                targets,
            )
            return pred

        pred = autoregressive_rollout(
            model,
            x,
            rollout_horizon,
        )
        return pred

    if prediction_horizon == rollout_horizon:
        output = model(x)

        if output.shape[1] != rollout_horizon:
            raise RuntimeError(
                "Direct model output has the wrong horizon. "
                f"Expected {rollout_horizon}, "
                f"got {output.shape[1]}."
            )

        return output

    raise ValueError(
        "prediction_horizon must be either 1 "
        "or rollout_horizon."
    )