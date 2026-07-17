"""
Metric and denormalization utilities for spectrum forecasts.

This module converts model outputs into physical dBm values when normalization
is enabled and calculates the elementwise errors used by the integrated result
aggregation pipeline.

Primary responsibilities include:

- reading per-frequency normalization metadata;
- denormalizing model predictions using the training-derived mean and standard
  deviation;
- preserving predictions unchanged when normalization is disabled;
- validating prediction, target, and normalization compatibility;
- calculating elementwise absolute error in dB;
- calculating elementwise squared error in dB squared;
- returning arrays in layouts expected by the result-aggregation functions; and
- avoiding model-specific assumptions beyond frequency-axis broadcasting.

For vector forecasts, frequency is expected on the final axis:

    (N, F)

For map forecasts, callers convert model layout into frequency-last layout
before invoking normalization and error calculation:

    model layout:         (N, F, H, W)
    normalization layout: (N, H, W, F)

Spatial reduction, frequency aggregation, and frequency-band aggregation are
performed outside this module.
"""

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
