"""Frozen-input frequency ablation helpers (evaluation only; nothing in this module trains or fits anything).

For one region (a set of bins) the inputs of every OTHER bin are replaced by the same bins shifted in time, the region's own bins are left real:

    shift96    24 hours at 15 minutes (96 steps): keeps each other bin's level, statistics and daily phase, loses its co-movement with the region
    shift210   52.5 hours (210 steps): also breaks the daily phase

The shift wraps around the series (``np.roll``), as ``training.ray.neighbor_ablation`` does for the neighbouring receivers; windows whose history starts
before row ``shift`` therefore contain wrapped rows (the first ``shift`` origins at most), which the analysis can exclude.
"""

from __future__ import annotations

import numpy as np

SHIFTS = {"real": 0, "shift96": 96, "shift210": 210}


def band_frequencies(band: str, n_bins: int = 200) -> np.ndarray:
    """Bin centre frequencies in MHz of a band like ``600_800`` (1 MHz bins: 600.5 ... 799.5)."""
    start = float(band.split("_")[0])
    return start + 0.5 + np.arange(n_bins, dtype=np.float64)


def region_mask(frequencies: np.ndarray, start_mhz: float, end_mhz: float) -> np.ndarray:
    """Bins whose centre lies inside the inclusive region [start_mhz, end_mhz] (the plan's region tables)."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    mask = (frequencies >= float(start_mhz)) & (frequencies <= float(end_mhz))
    if not mask.any():
        raise ValueError("region contains no bins")
    return mask


def shift_other_bins(data: np.ndarray, in_region: np.ndarray, shift: int) -> np.ndarray:
    """A copy of ``data`` (time x bins, optionally with trailing axes) in which every column outside ``in_region`` is rolled by ``shift`` rows.

    ``shift == 0`` returns an unchanged copy; the columns inside the region are never touched.
    """
    data = np.asarray(data)
    in_region = np.asarray(in_region, dtype=bool)
    if in_region.shape != (data.shape[1],):
        raise ValueError(f"mask {in_region.shape} does not match {data.shape[1]} columns")
    out = np.array(data, copy=True)
    if shift:
        out[:, ~in_region] = np.roll(data[:, ~in_region], int(shift), axis=0)
    return out


def shift_other_bins_multi_receiver(data: np.ndarray, in_region: np.ndarray, shift: int) -> np.ndarray:
    """The same for a receiver-major column layout (receivers x bins concatenated: TSS-LCD's 4 x 200 = 800 columns): the same bins at every receiver."""
    data = np.asarray(data)
    n_bins = len(in_region)
    if data.shape[1] % n_bins:
        raise ValueError("column count is not a multiple of the bin count")
    full = np.tile(np.asarray(in_region, dtype=bool), data.shape[1] // n_bins)
    return shift_other_bins(data, full, shift)


def region_errors(prediction: np.ndarray, target: np.ndarray, in_region: np.ndarray) -> np.ndarray:
    """Per-origin mean absolute error (dB) over the region's bins; prediction and target are origins x bins."""
    return np.abs(np.asarray(prediction, dtype=np.float64)[:, in_region] - np.asarray(target, dtype=np.float64)[:, in_region]).mean(axis=1)
