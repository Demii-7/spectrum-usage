#!/usr/bin/env python3
"""Forecastability atlas: how much do spectrum measurements differ across sites and bands, and what does it mean for
forecasting?

One row per (testbed, site, run, band) "cell", plus one row per bin. For each cell, from the raw per-minute power:

* level and noise floor, dynamic range, share of "active" bins (15-minute dynamic range above 3 dB);
* temporal structure: lag-1/15/60/1440-minute autocorrelation, excess kurtosis, daily-profile R^2;
* predictability: LZ78 entropy and Fano predictability of the median-centred series, and of its 1-step changes
  (the same definitions as ``compute_region_entropy.py``);
* headroom at resolutions of 1, 5, 15 and 60 minutes (block means in LINEAR power): the gap between a causal
  predictor (mean of the previous k blocks) and a two-sided oracle (median of k blocks on each side, target excluded),
  in MAE and MSE;
* realized gain of a cheap learned model (per-bin ridge AR on residuals from the window mean, trained on the first
  70% of the run, scored on the last 30%) over the window mean, next to the oracle's gain on the same rows.

Runs with more than 400 bins are subsampled to 200 evenly spaced bins (stated in the output).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESOLUTION_STEPS = (1, 5, 15, 60)           # block sizes in minutes (for 1-minute data)
CAUSAL_SPANS = {1: (2, 30), 5: (2, 6), 15: (2,), 60: (2,)}   # k values per resolution (30-minute spans where possible)
LAGS_MINUTES = (1, 15, 60, 1440)
ACTIVE_DYNAMIC_RANGE_DB = 3.0
MAX_BINS = 400
SUBSAMPLE_BINS = 200
RIDGE_ALPHA = 1.0
RIDGE_LAGS = 12
TRAIN_FRACTION = 0.7


# ------------------------------------------------------------------------------------------------ loading
def read_trace(path: Path, step_minutes: int = 1, time_format: str | None = None) -> pd.DataFrame:
    """CSV (first column = time) -> frame indexed by a regular grid of ``step_minutes`` with NaN for missing steps."""
    frame = pd.read_csv(path)
    stamps = pd.to_datetime(frame.iloc[:, 0], utc=True, format=time_format).dt.floor(f"{step_minutes}min")
    data = frame.iloc[:, 1:].astype(float)
    data.index = stamps
    data = data[~data.index.duplicated(keep="first")].sort_index()
    return data.reindex(pd.date_range(data.index[0], data.index[-1], freq=f"{step_minutes}min"))


def subsample_bins(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if frame.shape[1] <= MAX_BINS:
        return frame, False
    keep = np.unique(np.linspace(0, frame.shape[1] - 1, SUBSAMPLE_BINS).round().astype(int))
    return frame.iloc[:, keep], True


def block_db(frame: pd.DataFrame, factor: int, step_minutes: int = 1, min_valid: float = 0.8) -> pd.DataFrame:
    """Aligned blocks of ``factor`` native steps, averaged in linear power and returned in dB (NaN if <80% valid)."""
    if factor == 1:
        return frame
    power = np.power(10.0, frame / 10.0)
    block = power.index.floor(f"{factor * step_minutes}min")
    grouped = power.groupby(block)
    mean = grouped.mean().where(grouped.count() >= int(np.ceil(min_valid * factor)))
    full = pd.date_range(mean.index.min(), mean.index.max(), freq=f"{factor * step_minutes}min")
    return 10.0 * np.log10(mean.reindex(full))


# ------------------------------------------------------------------------------------ per-bin statistics
def autocorrelation(values: np.ndarray, lag: int) -> np.ndarray:
    """Per-bin Pearson correlation between x[t] and x[t+lag] over steps where both are finite."""
    if lag >= len(values) - 10:
        return np.full(values.shape[1], np.nan)
    a, b = values[:-lag], values[lag:]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(mask, a, np.nan), np.where(mask, b, np.nan)
    with np.errstate(all="ignore"):
        da, db = a - np.nanmean(a, axis=0), b - np.nanmean(b, axis=0)
        return np.nansum(da * db, axis=0) / np.sqrt(np.nansum(da ** 2, axis=0) * np.nansum(db ** 2, axis=0))


def excess_kurtosis(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        centered = values - np.nanmean(values, axis=0)
        return np.nanmean(centered ** 4, axis=0) / np.nanmean(centered ** 2, axis=0) ** 2 - 3.0


def daily_profile_r2(blocks: pd.DataFrame, steps_per_day: int) -> np.ndarray:
    """Cross-validated share of each bin's variance explained by its mean time-of-day profile.

    Each sample is predicted from the same time of day on the OTHER days (leave-one-day-out), so noise scores about 0
    (or slightly negative) instead of the overfit value a profile fitted on few days would give.
    """
    slot = np.asarray((blocks.index.hour * 60 + blocks.index.minute) // (1440 // steps_per_day))
    values = blocks.to_numpy(dtype=float)
    frame = pd.DataFrame(values)
    total = frame.groupby(slot).transform("sum").to_numpy()
    count = frame.notna().groupby(slot).transform("sum").to_numpy()
    with np.errstate(all="ignore"):
        loo = (total - np.nan_to_num(values)) / (count - 1)
        loo = np.where(count > 1, loo, np.nan)
        variance = np.nanvar(values, axis=0)
        return np.where(variance > 0, 1.0 - np.nanmean((values - loo) ** 2, axis=0) / variance, np.nan)


def causal_vs_oracle(values: np.ndarray, k: int, rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-bin MAE/MSE of the causal predictor (mean of previous k) and the two-sided oracle (median of +-k)."""
    values = values.astype(np.float32)
    steps = len(values)
    rows = np.arange(k, steps - k) if rows is None else rows[(rows >= k) & (rows < steps - k)]
    past = np.stack([values[rows - j] for j in range(1, k + 1)])
    future = np.stack([values[rows + j] for j in range(1, k + 1)])
    target = values[rows]
    valid = np.isfinite(past).all(axis=0) & np.isfinite(future).all(axis=0) & np.isfinite(target)
    causal = past.mean(axis=0)
    oracle = np.median(np.concatenate([past, future]), axis=0)
    out = {}
    with np.errstate(all="ignore"):
        for name, estimate in (("causal", causal), ("oracle", oracle)):
            error = np.where(valid, estimate - target, np.nan)
            out[f"{name}_mae"] = np.nanmean(np.abs(error), axis=0)
            out[f"{name}_mse"] = np.nanmean(error ** 2, axis=0)
    return out


def residual_ar_gain(values: np.ndarray, lags: int = RIDGE_LAGS, train_fraction: float = TRAIN_FRACTION,
                     alpha: float = RIDGE_ALPHA) -> dict[str, np.ndarray]:
    """Per-bin ridge AR on residuals from the window mean; MAE of model, window mean and oracle on the test rows."""
    steps, bins = values.shape
    rows = np.arange(lags, steps)
    lag_stack = np.stack([values[rows - j] for j in range(1, lags + 1)], axis=1)       # rows x lags x bins
    mean = lag_stack.mean(axis=1)
    features = lag_stack - mean[:, None, :]
    target = values[rows] - mean
    split = int(train_fraction * steps)
    train, test = rows < split, rows >= split
    model_mae = np.full(bins, np.nan)
    mean_mae = np.full(bins, np.nan)
    for b in range(bins):
        ok = np.isfinite(target[:, b]) & np.isfinite(features[:, :, b]).all(axis=1)
        tr, te = ok & train, ok & test
        if tr.sum() < 3 * lags or te.sum() < 10:
            continue
        x = np.column_stack([features[tr, :, b], np.ones(tr.sum())])
        gram = x.T @ x + alpha * np.diag(np.r_[np.ones(lags), 0.0])
        weights = np.linalg.solve(gram, x.T @ target[tr, b])
        pred = np.column_stack([features[te, :, b], np.ones(te.sum())]) @ weights
        model_mae[b] = np.abs(pred - target[te, b]).mean()
        mean_mae[b] = np.abs(target[te, b]).mean()
    oracle = causal_vs_oracle(values, 2, rows=rows[rows >= split])["oracle_mae"]
    return {"model_mae": model_mae, "window_mean_mae": mean_mae, "oracle_mae_test": oracle}


def entropy_per_bin(values: np.ndarray) -> pd.DataFrame:
    """E_actual / diff-entropy / Fano predictability per bin on the median-centred series (as compute_region_entropy)."""
    from compute_region_entropy import DEFAULT_EDGES, metrics

    edges = np.asarray(DEFAULT_EDGES, dtype=float)
    edges = edges - np.median(edges)
    rows = []
    for b in range(values.shape[1]):
        column = values[:, b]
        column = column - np.nanmedian(column)
        m = metrics(column, edges)
        rows.append({"E_actual": m["E_actual_normalized"], "diff_E_actual": m["diff_E_actual_normalized"],
                     "fano": m["fano_predictability"], "diff_fano": m["diff_fano_predictability"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ cells
def summarize_cell(frame: pd.DataFrame, meta: dict[str, Any], step_minutes: int = 1,
                   resolutions=RESOLUTION_STEPS, spans=CAUSAL_SPANS) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(cell row, per-bin rows) for one trace on a regular grid."""
    frame, subsampled = subsample_bins(frame)
    values = frame.to_numpy(dtype=float)
    freqs = [float(c) for c in frame.columns]
    cell = {**meta, "rows": len(frame), "hours": round(len(frame) * step_minutes / 60, 1), "bins_used": frame.shape[1],
            "bins_subsampled": subsampled, "missing_frac": float(np.isnan(values).mean()),
            "median_db": float(np.nanmedian(values))}
    per_bin: dict[str, np.ndarray] = {}
    per_bin["mean_db"] = np.nanmean(values, axis=0)
    per_bin["std_native"] = np.nanstd(values, axis=0)
    per_bin["p5_db"] = np.nanpercentile(values, 5, axis=0)
    per_bin["kurtosis"] = excess_kurtosis(values)
    for lag_minutes in LAGS_MINUTES:
        lag = lag_minutes // step_minutes
        if lag >= 1:
            per_bin[f"autocorr_{lag_minutes}min"] = autocorrelation(values, lag)
    coarse_factor = max(1, 15 // step_minutes)
    coarse = block_db(frame, coarse_factor, step_minutes)
    per_bin["dyn_range_15min"] = np.nanpercentile(coarse.to_numpy(), 95, axis=0) - np.nanpercentile(coarse.to_numpy(), 5, axis=0)
    if len(coarse) >= 2 * (1440 // 15):
        per_bin["daily_r2"] = daily_profile_r2(coarse, 1440 // 15)
    entropy = entropy_per_bin(values)
    for column in entropy.columns:
        per_bin[column] = entropy[column].to_numpy()
    cell["active_frac"] = float(np.nanmean(per_bin["dyn_range_15min"] > ACTIVE_DYNAMIC_RANGE_DB))
    cell["noise_floor_db"] = float(np.nanmedian(per_bin["p5_db"]))
    for name, array in per_bin.items():
        cell[f"{name}_median"] = float(np.nanmedian(array))
    # headroom and realized gain at each resolution
    for minutes in resolutions:
        factor = max(1, minutes // step_minutes)
        if minutes < step_minutes:
            continue
        blocks = block_db(frame, factor, step_minutes)
        arr = blocks.to_numpy(dtype=float)
        if len(arr) < 60:
            continue
        for k in spans[minutes]:
            res = causal_vs_oracle(arr, k)
            cell[f"r{minutes}_k{k}_causal_mae"] = float(np.nanmean(res["causal_mae"]))
            cell[f"r{minutes}_k{k}_oracle_mae"] = float(np.nanmean(res["oracle_mae"]))
            cell[f"r{minutes}_k{k}_headroom_mae"] = float(1 - np.nanmean(res["oracle_mae"]) / np.nanmean(res["causal_mae"]))
            cell[f"r{minutes}_k{k}_headroom_mse"] = float(1 - np.nanmean(res["oracle_mse"]) / np.nanmean(res["causal_mse"]))
            if minutes == 15 and k == 2:
                per_bin["headroom_mae_15min"] = 1 - res["oracle_mae"] / res["causal_mae"]
        if len(arr) >= 4 * RIDGE_LAGS + 40:
            gain = residual_ar_gain(arr)
            window_mae = float(np.nanmean(gain["window_mean_mae"]))
            cell[f"r{minutes}_window_mean_mae"] = window_mae
            cell[f"r{minutes}_model_mae"] = float(np.nanmean(gain["model_mae"]))
            cell[f"r{minutes}_realized_gain_mae"] = float(1 - np.nanmean(gain["model_mae"]) / window_mae)
            cell[f"r{minutes}_oracle_gain_mae_test"] = float(1 - np.nanmean(gain["oracle_mae_test"]) / window_mae)
            if minutes == 15:
                per_bin["realized_gain_15min"] = 1 - gain["model_mae"] / gain["window_mean_mae"]
    bin_rows = []
    for index, freq in enumerate(freqs):
        bin_rows.append({**meta, "frequency_mhz": freq, **{name: float(array[index]) for name, array in per_bin.items()}})
    return cell, bin_rows


def run_cell(task: dict[str, Any]) -> dict[str, Any]:
    """Load one CSV and summarize it (used locally and as a Ray task)."""
    meta = {k: task[k] for k in ("testbed", "site", "run", "band")}
    try:
        frame = read_trace(Path(task["path"]), task.get("step_minutes", 1), task.get("time_format"))
        cell, bins = summarize_cell(frame, meta, task.get("step_minutes", 1),
                                    task.get("resolutions", RESOLUTION_STEPS), task.get("spans", CAUSAL_SPANS))
        return {"ok": True, "cell": cell, "bins": bins}
    except Exception as error:  # keep going; the failure is recorded
        import traceback

        return {"ok": False, "meta": meta, "error": f"{type(error).__name__}: {error}"[:300],
                "trace": traceback.format_exc()[-600:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--testbed", default="x")
    parser.add_argument("--site", default="x")
    parser.add_argument("--run", default="x")
    parser.add_argument("--band", default="x")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_cell({"path": str(args.csv), "testbed": args.testbed, "site": args.site, "run": args.run,
                       "band": args.band})
    if not result["ok"]:
        raise SystemExit(result["error"])
    pd.DataFrame([result["cell"]]).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
