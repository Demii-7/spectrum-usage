"""Color-limit helpers for dB spectrograms."""

from __future__ import annotations

import numpy as np


def ensure_minimum_db_span(vmin: float, vmax: float, min_span_db: float = 5.0) -> tuple[float, float]:
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("spectrogram color limits must be finite")
    if vmax - vmin >= min_span_db:
        return vmin, vmax
    center = (vmin + vmax) / 2.0
    return center - min_span_db / 2.0, center + min_span_db / 2.0
