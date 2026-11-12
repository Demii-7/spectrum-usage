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


# --------------------------------------------------------------------------------------------- periodicity-crossed disturbances
# (evaluation/results/bin_periodicity_classes.csv; brief_periodicity_frequency_ablation.md). Existing functions above are unchanged.
CLASS_SETS = ("all", "P_only", "N_only", "noise_only")
DISTURBANCES = ("shift96", "shift192", "shift144", "shift210", "level")
SHIFT_STEPS = {"shift96": 96, "shift192": 192, "shift144": 144, "shift210": 210}      # 24 h, 48 h, 36 h (12 h out of phase), 52.5 h
CLASS_NAMES = ("P", "N", "middle", "noise")


def disturbed_mask(in_region: np.ndarray, classes: np.ndarray, class_set: str) -> np.ndarray:
    """Out-of-region bins that are disturbed for ``class_set``: all (every out-of-region bin), P_only, N_only or noise_only (class of the bin
    in the periodicity table; the target region's own bins are never disturbed; 'middle' bins are disturbed only in ``all``)."""
    out = ~np.asarray(in_region, dtype=bool)
    classes = np.asarray(classes)
    if class_set == "all":
        return out
    if class_set not in ("P_only", "N_only", "noise_only"):
        raise ValueError(f"unknown class set {class_set}")
    return out & (classes == class_set.split("_")[0])


def level_windows(windows: np.ndarray, disturbed: np.ndarray) -> np.ndarray:
    """Windows (n, lookback, bins[, ...]) with every disturbed bin replaced by its mean over the input window (no dynamics, no phase)."""
    windows = np.asarray(windows)
    out = np.array(windows, copy=True)
    if disturbed.any():
        out[:, :, disturbed] = windows[:, :, disturbed].mean(axis=1, keepdims=True)
    return out


def disturb_series(data: np.ndarray, disturbed: np.ndarray, kind: str) -> np.ndarray:
    """Series (time x bins) with the disturbed columns rolled in time (shift kinds, wrap-around as in ``shift_other_bins``); ``level`` is applied per window by
    ``level_windows`` and returns the series unchanged here."""
    if kind == "level":
        return np.array(data, copy=True)
    return shift_other_bins(data, ~np.asarray(disturbed, dtype=bool), SHIFT_STEPS[kind])


def class_bin_sums(error: np.ndarray, in_region: np.ndarray, classes: np.ndarray) -> dict[str, np.ndarray]:
    """Per-origin SUM of the absolute errors over the region's bins of each class ('P', 'N', 'middle'); ``error`` is origins x bins (abs errors)."""
    in_region = np.asarray(in_region, dtype=bool)
    classes = np.asarray(classes)
    return {c: error[:, in_region & (classes == c)].sum(axis=1) for c in ("P", "N", "middle")}
