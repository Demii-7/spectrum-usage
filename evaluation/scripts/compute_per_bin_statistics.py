#!/usr/bin/env python3
"""Compute descriptive per-frequency-bin statistics for spectrum power CSVs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CSV_NAME = "power_1mhz_avg_per_minute.csv"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "statistical_analysis"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def discover_csvs(input_root: Path, datasets: set[str], nodes: set[str], run_ids: set[str], bands: set[str]) -> list[Path]:
    paths = sorted(input_root.glob(f"**/{CSV_NAME}"))
    selected = []
    for path in paths:
        meta = path_metadata(input_root, path)
        if meta is None:
            continue
        if datasets and meta["dataset"] not in datasets:
            continue
        if nodes and meta["node"] not in nodes:
            continue
        if run_ids and meta["run_id"] not in run_ids:
            continue
        if bands and meta["band"] not in bands:
            continue
        selected.append(path)
    return selected


def path_metadata(input_root: Path, path: Path) -> dict[str, str] | None:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5:
        return None
    return {
        "dataset": parts[0],
        "node": parts[1],
        "run_id": parts[2],
        "band": parts[3],
        "relative_path": str(rel),
    }


def frequency_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col == "timestamp_utc":
            continue
        try:
            float(col)
        except ValueError:
            continue
        cols.append(col)
    return cols


def read_trace(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    if "timestamp_utc" not in df.columns:
        raise ValueError("missing timestamp_utc column")
    freqs = frequency_columns(df)
    if not freqs:
        raise ValueError("no numeric frequency columns")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce").dt.floor("min")
    df = df.dropna(subset=["timestamp_utc"])
    if df.empty:
        raise ValueError("no valid timestamps")

    # Multiple samples can floor to the same minute. Average them to one row per minute.
    df = df.groupby("timestamp_utc", sort=True)[freqs].mean()
    for col in freqs:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="min", tz="UTC")
    return df.reindex(full_index), freqs


def gap_lengths(mask: np.ndarray) -> list[int]:
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return (ends - starts).astype(int).tolist()


def leading_missing_count(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask[0]:
        return 0
    return int(np.argmax(~mask)) if np.any(~mask) else int(mask.size)


def autocorr(values: np.ndarray, lag: int) -> float:
    if values.size <= lag:
        return math.nan
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def daily_profile_range(values: np.ndarray, steps_per_day: int) -> float:
    if steps_per_day <= 0:
        return math.nan
    full_days = values.size // steps_per_day
    if full_days < 1:
        return math.nan
    daily = values[: full_days * steps_per_day].reshape(full_days, steps_per_day)
    profile = daily.mean(axis=0)
    return float(np.quantile(profile, 0.95) - np.quantile(profile, 0.05))


def binary_entropy(states: np.ndarray) -> float:
    if states.size == 0:
        return math.nan
    p = float(np.mean(states))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def conditional_entropy_lag1(states: np.ndarray) -> float:
    if states.size < 2:
        return math.nan
    prev = states[:-1].astype(np.int8)
    cur = states[1:].astype(np.int8)
    total = prev.size
    entropy = 0.0
    for state in (0, 1):
        mask = prev == state
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        entropy += (count / total) * binary_entropy(cur[mask])
    return float(entropy)


def transition_stats(states: np.ndarray, step_minutes: float) -> dict[str, float]:
    if states.size < 2:
        return {
            "transition_rate_per_hour": math.nan,
            "burst_count_per_hour": math.nan,
            "mean_occupied_duration_min": math.nan,
            "mean_idle_duration_min": math.nan,
        }
    transitions = np.flatnonzero(states[1:] != states[:-1]) + 1
    hours = states.size * step_minutes / 60.0
    burst_starts = int(np.count_nonzero((states[1:] == 1) & (states[:-1] == 0)))
    if states[0] == 1:
        burst_starts += 1

    lengths = gap_lengths(states.astype(bool))
    idle_lengths = gap_lengths(~states.astype(bool))
    return {
        "transition_rate_per_hour": float(transitions.size / hours) if hours else math.nan,
        "burst_count_per_hour": float(burst_starts / hours) if hours else math.nan,
        "mean_occupied_duration_min": float(np.mean(lengths) * step_minutes) if lengths else 0.0,
        "mean_idle_duration_min": float(np.mean(idle_lengths) * step_minutes) if idle_lengths else 0.0,
    }


def weekday_weekend_stats(index: pd.DatetimeIndex, values: np.ndarray, states: np.ndarray) -> dict[str, float]:
    weekdays = index.weekday < 5
    has_weekend = bool(np.any(~weekdays))
    weekday_mean = float(np.mean(values[weekdays])) if np.any(weekdays) else math.nan
    weekend_mean = float(np.mean(values[~weekdays])) if has_weekend else math.nan
    weekday_activity = float(np.mean(states[weekdays])) if np.any(weekdays) else math.nan
    weekend_activity = float(np.mean(states[~weekdays])) if has_weekend else math.nan
    return {
        "weekday_mean_power_dbm": weekday_mean,
        "weekend_mean_power_dbm": weekend_mean,
        "weekday_weekend_mean_diff_db": weekday_mean - weekend_mean if has_weekend else math.nan,
        "weekday_activity_rate": weekday_activity,
        "weekend_activity_rate": weekend_activity,
        "weekday_weekend_activity_diff": weekday_activity - weekend_activity if has_weekend else math.nan,
    }


def infer_step_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return math.nan
    diffs = np.diff(index.view("int64")) / 1e9 / 60.0
    return float(np.median(diffs))


def compute_file(path: Path, input_root: Path, max_fill_gap: int) -> tuple[pd.DataFrame, dict[str, object]]:
    meta = path_metadata(input_root, path)
    if meta is None:
        raise ValueError("path does not match dataset/node/run/band layout")
    frame, freq_cols = read_trace(path)
    missing = frame[freq_cols].isna()
    filled = frame[freq_cols].ffill().bfill()
    step_minutes = infer_step_minutes(filled.index)
    steps_per_day = int(round(1440 / step_minutes)) if step_minutes and math.isfinite(step_minutes) else 0

    rows = []
    total_missing = int(missing.to_numpy().sum())
    max_gap_overall = 0
    gap_count_overall = 0
    leading_fill_overall = 0

    for col in freq_cols:
        miss = missing[col].to_numpy(dtype=bool)
        gaps = gap_lengths(miss)
        longest_gap = max(gaps, default=0)
        gap_count = len(gaps)
        leading_fill = leading_missing_count(miss)
        max_gap_overall = max(max_gap_overall, longest_gap)
        gap_count_overall += gap_count
        leading_fill_overall += leading_fill

        values = filled[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            # A fully missing bin cannot be characterized after ffill/bfill.
            continue
        p05, p10, p50, p90, p95 = np.quantile(values, [0.05, 0.10, 0.50, 0.90, 0.95])
        threshold = p05 + 3.0
        states = values > threshold
        trans = transition_stats(states, step_minutes)
        ww = weekday_weekend_stats(filled.index, values, states)
        row = {
            **{key: meta[key] for key in ("dataset", "node", "run_id", "band", "relative_path")},
            "frequency_mhz": float(col),
            "timestamp_count": int(values.size),
            "step_minutes": step_minutes,
            "missing_count": int(np.count_nonzero(miss)),
            "missing_rate": float(np.mean(miss)),
            "gap_count": gap_count,
            "longest_gap_steps": longest_gap,
            "forward_filled_count": int(np.count_nonzero(miss) - leading_fill),
            "backfilled_leading_count": leading_fill,
            "long_gap_flag": bool(longest_gap > max_fill_gap),
            "mean_power_dbm": float(np.mean(values)),
            "p05_power_dbm": float(p05),
            "p10_power_dbm": float(p10),
            "p50_power_dbm": float(p50),
            "p90_power_dbm": float(p90),
            "p95_power_dbm": float(p95),
            "p95_p05_range_db": float(p95 - p05),
            "std_power_db": float(np.std(values)),
            "variance_power_db2": float(np.var(values)),
            "short_timescale_std_db": float(np.std(np.diff(values)) / math.sqrt(2.0)) if values.size > 1 else math.nan,
            "noise_floor_estimate_dbm": float(p05),
            "occupancy_threshold_dbm": float(threshold),
            "occupancy_rate": float(np.mean(states)),
            "autocorr_lag_1": autocorr(values, 1),
            "autocorr_lag_5": autocorr(values, 5),
            "autocorr_lag_15": autocorr(values, 15),
            "autocorr_lag_60": autocorr(values, 60),
            "autocorr_lag_1440": autocorr(values, steps_per_day) if steps_per_day else math.nan,
            "daily_profile_range_db": daily_profile_range(values, steps_per_day),
            "binary_entropy_bits": binary_entropy(states),
            "conditional_entropy_lag1_bits": conditional_entropy_lag1(states),
        }
        row.update(trans)
        row.update(ww)
        rows.append(row)

    trace_summary = {
        **{key: meta[key] for key in ("dataset", "node", "run_id", "band", "relative_path")},
        "start_time_utc": filled.index[0].isoformat(),
        "end_time_utc": filled.index[-1].isoformat(),
        "duration_minutes": int(len(filled.index) - 1),
        "time_resolution_minutes": step_minutes,
        "timestamp_count": int(len(filled.index)),
        "frequency_bin_count": int(len(freq_cols)),
        "start_mhz": float(freq_cols[0]),
        "end_mhz": float(freq_cols[-1]),
        "frequency_resolution_mhz": float(np.median(np.diff([float(c) for c in freq_cols]))) if len(freq_cols) > 1 else math.nan,
        "missing_count": total_missing,
        "missing_rate": float(total_missing / (len(filled.index) * len(freq_cols))),
        "gap_count": gap_count_overall,
        "longest_gap_steps": max_gap_overall,
        "forward_filled_count": int(total_missing - leading_fill_overall),
        "backfilled_leading_count": int(leading_fill_overall),
        "long_gap_flag": bool(max_gap_overall > max_fill_gap),
        "mean_power_dbm": float(filled.to_numpy(dtype=np.float64).mean()),
        "p05_power_dbm": float(np.quantile(filled.to_numpy(dtype=np.float64), 0.05)),
        "p50_power_dbm": float(np.quantile(filled.to_numpy(dtype=np.float64), 0.50)),
        "p95_power_dbm": float(np.quantile(filled.to_numpy(dtype=np.float64), 0.95)),
    }
    return pd.DataFrame(rows), trace_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT, help="Root containing dataset/node/run/band CSVs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output CSVs.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset prefix to include. Repeatable.")
    parser.add_argument("--node", action="append", default=[], help="Node/site name to include. Repeatable.")
    parser.add_argument("--run-id", action="append", default=[], help="Run ID to include. Repeatable.")
    parser.add_argument("--band", action="append", default=[], help="Band label to include. Repeatable.")
    parser.add_argument("--max-fill-gap", type=int, default=10, help="Gap length in samples above which long_gap_flag is set.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N matching CSVs, for testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    output_dir = resolve_path(args.output_dir)
    paths = discover_csvs(
        input_root,
        set(args.dataset),
        set(args.node),
        set(args.run_id),
        set(args.band),
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("ERROR: no matching power CSVs found", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    bin_frames = []
    trace_rows = []
    failed = 0
    for path in paths:
        rel = path.relative_to(input_root)
        try:
            bin_frame, trace_summary = compute_file(path, input_root, args.max_fill_gap)
        except Exception as exc:
            failed += 1
            print(f"ERROR: failed {rel}: {exc}", file=sys.stderr)
            continue
        bin_frames.append(bin_frame)
        trace_rows.append(trace_summary)
        print(f"processed {rel}: bins={len(bin_frame)}")

    if not bin_frames:
        print("ERROR: no files processed successfully", file=sys.stderr)
        return 1

    pd.concat(bin_frames, ignore_index=True).to_csv(output_dir / "frequency_bin_features.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(output_dir / "trace_summary.csv", index=False)
    print(f"wrote {output_dir / 'frequency_bin_features.csv'}")
    print(f"wrote {output_dir / 'trace_summary.csv'}")
    print(f"complete: processed={len(trace_rows)}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
