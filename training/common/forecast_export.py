"""
Forecast serialization utilities for integrated model evaluation.

This module writes model predictions, aligned targets, target-row indices, and
evaluation metadata to portable forecast artifact files. It provides one shared
export path for both vector-based spectrum forecasts and spatial spectrum-map
forecasts.

Forecast arrays are organized by requested prediction horizon. Each exported
artifact preserves enough information to reproduce downstream comparisons,
plotting, and analysis without rerunning model inference.

Primary responsibilities include:

- validating that predictions, targets, and target-row arrays are available for
  the same set of horizons;
- checking that prediction and target shapes agree for each horizon;
- preserving the established export layouts for CSV and map models;
- storing horizon-specific predictions and targets in a structured archive;
- storing split-local target-row indices used to align forecasts with source
  timesteps;
- converting metadata values into serializable representations;
- preserving model, split, chunk, frequency, normalization, checkpoint, and
  forecasting-policy information;
- constructing stable output filenames; and
- creating destination directories when necessary.

Expected forecast layouts:

    Vector forecasts:
        predictions: (N, F)
        targets:     (N, F)

    Map forecasts:
        predictions: (N, F, H, W)
        targets:     (N, F, H, W)

This module does not calculate predictions or metrics. It only validates and
serializes completed forecast results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def export_map_forecasts(
    out_dir: Path,
    chunk_id: str,
    model_name: str,
    predictions_by_horizon: dict[int, np.ndarray],
    targets_by_horizon: dict[int, np.ndarray],
    target_rows_by_horizon: dict[int, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    forecast_dir = out_dir / "forecasts"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    pred_payload: dict[str, np.ndarray] = {}
    target_payload: dict[str, np.ndarray] = {}
    horizon_meta: list[dict[str, Any]] = []
    for horizon in sorted(predictions_by_horizon):
        pred_key = f"t_plus_{horizon}"
        row_key = f"rows_t_plus_{horizon}"
        pred_payload[pred_key] = predictions_by_horizon[horizon].astype(np.float32)
        pred_payload[row_key] = target_rows_by_horizon[horizon].astype(np.int32)
        target_payload[pred_key] = targets_by_horizon[horizon].astype(np.float32)
        target_payload[row_key] = target_rows_by_horizon[horizon].astype(np.int32)
        rows = target_rows_by_horizon[horizon]
        horizon_meta.append(
            {
                "horizon": int(horizon),
                "n_targets": int(len(rows)),
                "target_row_start_zero_based": int(rows[0]) if len(rows) else None,
                "target_row_end_zero_based": int(rows[-1]) if len(rows) else None,
                "prediction_shape": list(predictions_by_horizon[horizon].shape),
            }
        )

    stem = f"{chunk_id}_{model_name}"
    np.savez(forecast_dir / f"{stem}_predictions.npz", **pred_payload)
    np.savez(forecast_dir / f"{stem}_targets.npz", **target_payload)

    metadata_payload = metadata_json_ready(metadata)
    metadata_payload["horizons"] = horizon_meta
    with (forecast_dir / f"{stem}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)


def metadata_json_ready(
    payload: dict[str, Any],
) -> dict[str, Any]:
    ready: dict[str, Any] = {}

    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            ready[key] = value.tolist()
        elif isinstance(value, np.generic):
            ready[key] = value.item()
        elif isinstance(value, Path):
            ready[key] = str(value)
        elif isinstance(value, dict):
            ready[key] = metadata_json_ready(value)
        elif isinstance(value, (list, tuple)):
            ready[key] = [
                item.tolist()
                if isinstance(item, np.ndarray)
                else item.item()
                if isinstance(item, np.generic)
                else str(item)
                if isinstance(item, Path)
                else item
                for item in value
            ]
        else:
            ready[key] = value

    return ready