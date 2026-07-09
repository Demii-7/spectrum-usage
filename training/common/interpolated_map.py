from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from training.common.config import resolve_path


def load_interpolated_map_npz(
    path: str | Path,
    map_key: str = "map_db",
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = resolve_path(path)
    loaded = np.load(resolved, allow_pickle=True)
    data = loaded[map_key].astype(np.float32).transpose(0, 3, 1, 2)
    metadata: dict[str, Any] = {"path": str(resolved), "map_key": map_key}
    for key in ("timestamps", "freqs_mhz", "site_names", "metadata"):
        if key not in loaded:
            continue
        value = loaded[key]
        if key == "metadata":
            metadata[key] = _decode_metadata_value(value)
        else:
            metadata[key] = value
    return data, metadata


def normalize_map_by_frequency(
    train_map: np.ndarray,
    test_map: np.ndarray | None = None,
    enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any] | None]:
    if not enabled:
        return train_map.astype(np.float32), None if test_map is None else test_map.astype(np.float32), None

    mean = np.mean(train_map, axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.std(train_map, axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, 1e-8, std)
    train_norm = ((train_map - mean) / std).astype(np.float32)
    test_norm = None if test_map is None else ((test_map - mean) / std).astype(np.float32)
    stats = {"method": "zscore_per_frequency", "mean": mean, "std": std}
    return train_norm, test_norm, stats


def denormalize_map(data: np.ndarray, stats: dict[str, Any] | None) -> np.ndarray:
    if not stats:
        return data.astype(np.float32)
    if stats.get("method") != "zscore_per_frequency":
        raise ValueError(f"Unsupported map normalization method: {stats.get('method')}")
    return (data * stats["std"] + stats["mean"]).astype(np.float32)


def prediction_start_row(config: dict[str, Any], total_rows: int) -> int:
    row = config.get("evaluation", {}).get("prediction_start_row")
    if row is None:
        return 0
    row = int(row)
    if row <= 0:
        raise ValueError("evaluation.prediction_start_row must be a positive 1-based row number")
    index = row - 1
    if index >= total_rows:
        raise ValueError(
            f"evaluation.prediction_start_row={row} is beyond the available {total_rows} rows"
        )
    return index


def metadata_json_ready(payload: dict[str, Any]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            ready[key] = value.tolist()
        elif isinstance(value, np.generic):
            ready[key] = value.item()
        elif isinstance(value, Path):
            ready[key] = str(value)
        else:
            ready[key] = value
    return ready


def _decode_metadata_value(value: np.ndarray) -> Any:
    if value.shape == ():
        item = value.item()
    else:
        item = value.tolist()
    if isinstance(item, str):
        try:
            return json.loads(item)
        except json.JSONDecodeError:
            return item
    return item
