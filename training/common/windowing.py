from __future__ import annotations

import numpy as np


def target_rows_for(
    split_length: int,
    history_offset: int,
    horizon: int,
    lookback: int,
    min_history: int = 4320,
) -> np.ndarray:
    start = max(history_offset, horizon + lookback - 1, min_history)
    end = history_offset + split_length
    if start >= end:
        raise ValueError(
            f"Split length {split_length} is too short for horizon {horizon}, "
            f"lookback {lookback}, and min_history {min_history}."
        )
    return np.arange(start, end)


def lagged_matrix(series: np.ndarray, target_rows: np.ndarray, horizon: int, lookback: int) -> np.ndarray:
    starts = target_rows - horizon - lookback + 1
    return np.stack([series[start : start + lookback] for start in starts], axis=0).astype(np.float32)


def aligned_history_matrix(x: np.ndarray, target_rows: np.ndarray, horizon: int, lookback: int) -> np.ndarray:
    starts = target_rows - horizon - lookback + 1
    return np.stack([x[start : start + lookback] for start in starts], axis=0).astype(np.float32)


def selected_horizon_index(horizon: int) -> int:
    return int(horizon) - 1
