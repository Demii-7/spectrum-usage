"""Shared color-limit helpers for dB spectrograms."""

from __future__ import annotations

import numpy as np


MIN_DB_SPAN = 5.0


def ensure_minimum_db_span(vmin: float, vmax: float, min_span_db: float = MIN_DB_SPAN) -> tuple[float, float]:
    """Expand finite color limits symmetrically to at least ``min_span_db``."""
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("spectrogram color limits must be finite")
    if vmax < vmin:
        raise ValueError("spectrogram maximum must not be below its minimum")
    if vmax - vmin >= min_span_db:
        return vmin, vmax
    center = (vmin + vmax) / 2.0
    half_span = min_span_db / 2.0
    return center - half_span, center + half_span
