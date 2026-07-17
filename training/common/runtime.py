"""
Runtime environment and execution metadata utilities.

This module centralizes small runtime decisions that are shared across training,
evaluation, export, and result-generation scripts. It provides a consistent
interface for selecting the PyTorch execution device and producing standardized
UTC timestamps for logs, checkpoints, and result metadata.

Primary responsibilities include:

- reading runtime and device preferences from the shared configuration;
- selecting CPU, CUDA, or another supported PyTorch device;
- validating requested accelerator availability;
- applying a deterministic fallback when the requested device is unavailable,
  when such fallback behavior is permitted;
- returning device objects suitable for model and tensor placement; and
- generating timezone-aware UTC timestamps in a stable serialized format.

Keeping these operations in one module prevents training and evaluation scripts
from implementing different device-selection or timestamp-formatting policies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import torch

def device_for(config: dict[str, Any]) -> torch.device:
    """Looks up the device parameter in config file to use gpu compute if available"""
    requested = str(config["training"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def epoch_log_row(
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    epoch_start_time: str,
    epoch_end_time: str,
    epoch_duration_sec: float,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch_start_time": epoch_start_time,
        "epoch_end_time": epoch_end_time,
        "epoch_duration_sec": epoch_duration_sec,
    }
    if learning_rate is not None:
        row["learning_rate"] = learning_rate
    return row
