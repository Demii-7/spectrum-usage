from __future__ import annotations

from typing import Any

import numpy as np


def denormalize(x: np.ndarray, normalization: dict[str, Any] | None) -> np.ndarray:
    if normalization is None:
        return x.astype(np.float32, copy=False)
    return (x * np.float32(normalization["std_dbm"]) + np.float32(normalization["mean_dbm"])).astype(np.float32)


def metric_values_dbm(
    pred: np.ndarray,
    target_raw_dbm: np.ndarray,
    normalization: dict[str, Any] | None = None,
) -> tuple[float, float]:
    pred_dbm = denormalize(pred, normalization)
    err = pred_dbm - target_raw_dbm.astype(np.float32, copy=False)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def absolute_and_squared_errors_dbm(
    pred: np.ndarray,
    target_raw_dbm: np.ndarray,
    normalization: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_dbm = denormalize(pred, normalization)
    err = pred_dbm - target_raw_dbm.astype(np.float32, copy=False)
    return pred_dbm, np.abs(err), err**2
