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

import numpy as np
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

    
class RolloutRemask:
    """Opt-in: before a predicted timestep is fed back as input, replace the columns OUTSIDE the target region by FRESH noise.

    Used by the ``region_only_remask`` spectral-ablation condition (a model trained with the loss on the region's bins only never learns what to output for the other
    bins, so feeding its own output there back in corrupts the window). The noise follows the calibration of the input mask by construction: its per-column mean and
    standard deviation are those of the (masked, i.e. noise-floor) out-of-region columns of the lookback window itself, pooled over the batch and the window; it is drawn
    from a generator that is re-seeded at the start of every rollout, so a given batch always gets the same noise. The returned forecast keeps the model's own output.
    """

    def __init__(self, columns: torch.Tensor, seed: int = 0) -> None:
        self.columns = columns.long()
        self.seed = int(seed)

    def generator(self, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        return generator

    def apply(self, next_prediction: torch.Tensor, window: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if next_prediction.ndim != 2:
            raise ValueError("RolloutRemask supports (batch, frequency) one-step outputs only")
        columns = self.columns.to(next_prediction.device)
        reference = window.index_select(2, columns)
        mean = reference.mean(dim=(0, 1))
        std = reference.std(dim=(0, 1)).clamp_min(1e-6)
        noise = torch.randn((next_prediction.shape[0], len(columns)), generator=generator, device=next_prediction.device, dtype=next_prediction.dtype)
        fed_back = next_prediction.clone()
        fed_back[:, columns] = noise * std + mean
        return fed_back


def make_rollout_remask(config: dict, frequencies, device=None):
    """The ``RolloutRemask`` of a config with ``data.mask.rollout_remask: true`` (the columns outside ``data.mask.frequency_ranges``), else None (default behaviour)."""
    mask = (config.get("data") or {}).get("mask") or {}
    if not mask.get("rollout_remask"):
        return None
    if frequencies is None:
        raise ValueError("data.mask.rollout_remask needs the data frequencies")
    from training.common.data import frequency_range_mask

    keep = frequency_range_mask(np.asarray(frequencies), mask["frequency_ranges"])
    return RolloutRemask(torch.as_tensor(np.flatnonzero(~keep), dtype=torch.long), seed=int(mask.get("seed", 0)) + 1)


def autoregressive_rollout( model: nn.Module, x: torch.Tensor, rollout_horizon: int, remask: "RolloutRemask | None" = None, ) -> torch.Tensor:
    """Repeatedly use a one-step model to predict multiple future steps (``remask``: opt-in fresh noise in the fed-back out-of-region columns)."""

    predictions = []
    generator = remask.generator(x.device) if remask is not None else None

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
        fed_back = next_prediction if remask is None else remask.apply(next_prediction, window, generator)
        next_timestep = fed_back.unsqueeze(1)

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
    remask: "RolloutRemask | None" = None,
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
            remask=remask,
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