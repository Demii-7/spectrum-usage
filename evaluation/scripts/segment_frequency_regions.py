#!/usr/bin/env python3
"""Segment frequency bins into homogeneous descriptive regions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "statistical_analysis" / "frequency_bin_features.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "statistical_analysis"
GROUP_COLUMNS = ["dataset", "node", "run_id", "band"]

DEFAULT_SEGMENT_FEATURES = [
    "p50_power_dbm",
    "p05_power_dbm",
    "p95_power_dbm",
    "std_power_db",
    "short_timescale_std_db",
    "occupancy_rate",
    "autocorr_lag_1440",
    "daily_profile_range_db",
    "binary_entropy_bits",
]


@dataclass(frozen=True)
class Segment:
    start: int
    end: int


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def robust_standardize(values: np.ndarray) -> np.ndarray:
    center = np.nanmedian(values, axis=0)
    scale = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    fallback = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    standardized = (values - center) / scale
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)


def segment_sse(prefix: np.ndarray, prefix_sq: np.ndarray, start: int, end: int) -> float:
    count = end - start
    if count <= 0:
        return 0.0
    total = prefix[end] - prefix[start]
    total_sq = prefix_sq[end] - prefix_sq[start]
    return float(np.sum(total_sq - (total * total / count)))


def best_split(prefix: np.ndarray, prefix_sq: np.ndarray, segment: Segment, min_bins: int) -> tuple[int | None, float]:
    if segment.end - segment.start < 2 * min_bins:
        return None, 0.0
    base = segment_sse(prefix, prefix_sq, segment.start, segment.end)
    best_idx = None
    best_gain = 0.0
    for split in range(segment.start + min_bins, segment.end - min_bins + 1):
        left = segment_sse(prefix, prefix_sq, segment.start, split)
        right = segment_sse(prefix, prefix_sq, split, segment.end)
        gain = base - left - right
        if gain > best_gain:
            best_idx = split
            best_gain = gain
    return best_idx, best_gain


def binary_segments(features: np.ndarray, min_bins: int, penalty: float, max_segments: int) -> list[Segment]:
    if features.shape[0] == 0:
        return []
    prefix = np.vstack([np.zeros(features.shape[1]), np.cumsum(features, axis=0)])
    prefix_sq = np.vstack([np.zeros(features.shape[1]), np.cumsum(features * features, axis=0)])
    segments = [Segment(0, features.shape[0])]

    while len(segments) < max_segments:
        candidates = []
        for idx, segment in enumerate(segments):
            split, gain = best_split(prefix, prefix_sq, segment, min_bins)
            if split is not None:
                candidates.append((gain, idx, split))
        if not candidates:
            break
        gain, idx, split = max(candidates, key=lambda item: item[0])
        if gain <= penalty:
            break
        old = segments.pop(idx)
        segments.insert(idx, Segment(split, old.end))
        segments.insert(idx, Segment(old.start, split))
    return segments


def contiguous_chunks(group: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = group.sort_values("frequency_mhz").reset_index(drop=True)
    freqs = ordered["frequency_mhz"].to_numpy(dtype=np.float64)
    if len(freqs) <= 1:
        return [ordered]
    diffs = np.diff(freqs)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return [ordered]
    expected = float(np.median(positive_diffs))
    breaks = np.flatnonzero(diffs > expected * 1.5) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(ordered)]
    return [ordered.iloc[start:end].reset_index(drop=True) for start, end in zip(starts, ends)]


def mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else math.nan


def max_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.max()) if values.notna().any() else math.nan


def classify_segment(row: dict[str, object], thresholds: dict[str, float]) -> str:
    std = float(row.get("std_power_db", math.nan))
    short_std = float(row.get("short_timescale_std_db", math.nan))
    occupancy = float(row.get("occupancy_rate", math.nan))
    autocorr = float(row.get("autocorr_lag_1440", math.nan))
    daily_range = float(row.get("daily_profile_range_db", math.nan))
    mean_power = float(row.get("mean_power_dbm", math.nan))

    if autocorr >= thresholds["diurnal_autocorr"] and daily_range >= thresholds["diurnal_daily_range_db"]:
        return "diurnal_pattern"
    if mean_power <= thresholds["mean_power_median_dbm"] and std <= thresholds["low_std_db"] and occupancy <= 0.05:
        return "noise_floor"
    if occupancy >= 0.95 and std <= thresholds["low_std_db"]:
        return "constant_occupancy"
    if short_std >= thresholds["high_short_std_db"]:
        return "bursty_short_timescale"
    if std >= thresholds["high_std_db"] or 0.05 < occupancy < 0.95:
        return "intermittent_occupancy"
    return "mixed_activity"


def group_thresholds(group: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_power_median_dbm": float(group["mean_power_dbm"].median()),
        "low_std_db": 0.5,
        "high_std_db": float(group["std_power_db"].quantile(2 / 3)),
        "high_short_std_db": float(group["short_timescale_std_db"].quantile(2 / 3)),
        "diurnal_autocorr": 0.5,
        "diurnal_daily_range_db": 3.0,
    }


def summarize_segment(
    group_keys: dict[str, object],
    chunk_id: int,
    segment_id: int,
    rows: pd.DataFrame,
    thresholds: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, object]:
    freqs = rows["frequency_mhz"].to_numpy(dtype=np.float64)
    out: dict[str, object] = {
        **group_keys,
        "chunk_id": chunk_id,
        "segment_id": segment_id,
        "start_mhz": float(freqs[0]),
        "end_mhz": float(freqs[-1]),
        "center_mhz": float((freqs[0] + freqs[-1]) / 2.0),
        "bandwidth_mhz": float((freqs[-1] - freqs[0]) + rows["frequency_mhz"].diff().median()) if len(freqs) > 1 else math.nan,
        "bin_count": int(len(rows)),
        "min_bins": args.min_bins,
        "penalty": args.penalty,
        "max_segments": args.max_segments,
    }
    summary_columns = [
        "missing_rate",
        "longest_gap_steps",
        "mean_power_dbm",
        "p05_power_dbm",
        "p50_power_dbm",
        "p95_power_dbm",
        "p95_p05_range_db",
        "std_power_db",
        "variance_power_db2",
        "short_timescale_std_db",
        "occupancy_rate",
        "transition_rate_per_hour",
        "burst_count_per_hour",
        "mean_occupied_duration_min",
        "mean_idle_duration_min",
        "autocorr_lag_1",
        "autocorr_lag_5",
        "autocorr_lag_15",
        "autocorr_lag_60",
        "autocorr_lag_1440",
        "daily_profile_range_db",
        "binary_entropy_bits",
        "conditional_entropy_lag1_bits",
        "weekday_weekend_mean_diff_db",
        "weekday_weekend_activity_diff",
    ]
    for col in summary_columns:
        if col in rows.columns:
            out[col] = mean_or_nan(rows[col])
    if "long_gap_flag" in rows.columns:
        out["long_gap_flag"] = bool(rows["long_gap_flag"].astype(bool).any())
    out["task_regime"] = classify_segment(out, thresholds)
    return out


def segment_group(group: pd.DataFrame, features: list[str], args: argparse.Namespace) -> list[dict[str, object]]:
    group = group.sort_values("frequency_mhz").reset_index(drop=True)
    group_keys = {col: group[col].iloc[0] for col in GROUP_COLUMNS}
    thresholds = group_thresholds(group)
    rows = []
    segment_id = 1
    for chunk_id, chunk in enumerate(contiguous_chunks(group), start=1):
        available_features = [col for col in features if col in chunk.columns and chunk[col].notna().any()]
        if not available_features:
            raise ValueError("no requested segmentation features exist in input table")
        matrix = robust_standardize(chunk[available_features].to_numpy(dtype=np.float64))
        segments = binary_segments(matrix, args.min_bins, args.penalty, args.max_segments)
        for segment in segments:
            segment_rows = chunk.iloc[segment.start : segment.end]
            rows.append(summarize_segment(group_keys, chunk_id, segment_id, segment_rows, thresholds, args))
            segment_id += 1
    return rows


def write_summary(segments: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Frequency Segmentation Summary", ""]
    lines.append(f"Segments: `{len(segments)}`")
    lines.append("")
    lines.append("## Task Regime Counts")
    lines.append("| Task regime | Segments |")
    lines.append("|---|---:|")
    for regime, count in segments["task_regime"].value_counts().sort_index().items():
        lines.append(f"| {regime} | {count} |")
    lines.append("")
    lines.append("## Segment Counts by Dataset")
    lines.append("| Dataset | Segments |")
    lines.append("|---|---:|")
    for dataset, count in segments["dataset"].value_counts().sort_index().items():
        lines.append(f"| {dataset} | {count} |")
    (output_dir / "frequency_segmentation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input frequency_bin_features.csv path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for frequency_segments.csv.")
    parser.add_argument("--feature", action="append", default=[], help="Segmentation feature column. Repeatable.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset prefix to include. Repeatable.")
    parser.add_argument("--node", action="append", default=[], help="Node/site name to include. Repeatable.")
    parser.add_argument("--run-id", action="append", default=[], help="Run ID to include. Repeatable.")
    parser.add_argument("--band", action="append", default=[], help="Band label to include. Repeatable.")
    parser.add_argument("--min-bins", type=int, default=5, help="Minimum bins per segment.")
    parser.add_argument("--penalty", type=float, default=8.0, help="Minimum SSE reduction required to split a segment.")
    parser.add_argument("--max-segments", type=int, default=80, help="Maximum segments per contiguous frequency chunk.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    features = args.feature or DEFAULT_SEGMENT_FEATURES
    data = pd.read_csv(input_path)
    for col, values in (("dataset", args.dataset), ("node", args.node), ("run_id", args.run_id), ("band", args.band)):
        if values:
            data = data[data[col].isin(set(values))]
    if data.empty:
        raise SystemExit("ERROR: no matching rows in input feature table")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_segments = []
    for keys, group in data.groupby(GROUP_COLUMNS, sort=True):
        group_segments = segment_group(group, features, args)
        all_segments.extend(group_segments)
        label = "/".join(str(value) for value in keys)
        print(f"segmented {label}: segments={len(group_segments)}")

    segments = pd.DataFrame(all_segments)
    output_path = output_dir / "frequency_segments.csv"
    segments.to_csv(output_path, index=False)
    write_summary(segments, output_dir)
    print(f"wrote {output_path}")
    print(f"wrote {output_dir / 'frequency_segmentation_summary.md'}")
    print(f"complete: groups={data.groupby(GROUP_COLUMNS).ngroups}, segments={len(segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
