"""Framework-neutral training progress events."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


TrainingCallback = Callable[[dict[str, Any]], None]


def emit_training_event(
    callback: TrainingCallback | None,
    *,
    model_name: str,
    stage: str,
    epoch: int,
    epochs: int,
    metrics: Mapping[str, Any],
    selection_metric: str,
    selection_mode: str,
    duration: float,
    prunable: bool,
    chunk_id: str | None = None,
    is_best: bool | None = None,
    stage_epoch: int | None = None,
    stage_epochs: int | None = None,
    selection_eligible: bool = True,
    best_epoch: int | None = None,
    best_value: float | None = None,
) -> None:
    """Send one plain mapping to a callback, allowing callback errors to propagate."""
    if callback is None:
        return

    event: dict[str, Any] = {
        "model_name": model_name,
        "stage": stage,
        "epoch": int(epoch),
        "epochs": int(epochs),
        "metrics": dict(metrics),
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "selection_value": metrics.get(selection_metric),
        "is_best": is_best,
        "duration": float(duration),
        "prunable": bool(prunable),
        "selection_eligible": bool(selection_eligible),
    }
    if chunk_id is not None:
        event["chunk_id"] = chunk_id
    if stage_epoch is not None:
        event["stage_epoch"] = int(stage_epoch)
    if stage_epochs is not None:
        event["stage_epochs"] = int(stage_epochs)
    if best_epoch is not None:
        event["best_epoch"] = int(best_epoch)
    if best_value is not None:
        event["best_value"] = float(best_value)
    callback(event)
