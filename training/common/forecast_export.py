from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from training.common.interpolated_map import metadata_json_ready


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
