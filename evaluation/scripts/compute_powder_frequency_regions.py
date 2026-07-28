#!/usr/bin/env python3
"""Compute frequency region definitions for POWDER data via statistical segmentation."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "evaluation" / "results" / "statistical_analysis"

SEGMENT_FEATURES = [
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


# ---------------------------------------------------------------------------
# Per-bin statistics
# ---------------------------------------------------------------------------


def autocorr(values: np.ndarray, lag: int) -> float:
    if values.size <= lag:
        return math.nan
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def binary_entropy(states: np.ndarray) -> float:
    if states.size == 0:
        return math.nan
    p = float(np.mean(states))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def gap_lengths(mask: np.ndarray) -> list[int]:
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return (ends - starts).astype(int).tolist()


def daily_profile_range(values: np.ndarray, steps_per_day: int) -> float:
    if steps_per_day <= 0:
        return math.nan
    full_days = values.size // steps_per_day
    if full_days < 1:
        return math.nan
    daily = values[: full_days * steps_per_day].reshape(full_days, steps_per_day)
    profile = daily.mean(axis=0)
    return float(np.quantile(profile, 0.95) - np.quantile(profile, 0.05))


def per_bin_statistics(data: np.ndarray, freqs: np.ndarray, occupancy_threshold: float) -> pd.DataFrame:
    n_freq = data.shape[1]
    rows = []
    for idx in range(n_freq):
        values = data[:, idx]
        missing = np.isnan(values)
        if np.all(missing):
            rows.append({"frequency_mhz": float(freqs[idx])})
            continue
        filled = np.nan_to_num(values, nan=np.nanmedian(values[~missing]))
        states = (filled >= occupancy_threshold).astype(float)

        rows.append(
            {
                "frequency_mhz": float(freqs[idx]),
                "missing_rate": float(np.mean(missing)),
                "longest_gap_steps": max(gap_lengths(missing), default=0),
                "mean_power_dbm": float(np.mean(filled)),
                "std_power_db": float(np.std(filled)),
                "variance_power_db2": float(np.var(filled)),
                "p05_power_dbm": float(np.percentile(filled, 5)),
                "p50_power_dbm": float(np.percentile(filled, 50)),
                "p95_power_dbm": float(np.percentile(filled, 95)),
                "occupancy_rate": float(np.mean(states)),
                "binary_entropy_bits": binary_entropy(states),
                "short_timescale_std_db": float(np.std(np.diff(filled))),
                "autocorr_lag_1": autocorr(filled, 1),
                "autocorr_lag_5": autocorr(filled, 5),
                "autocorr_lag_15": autocorr(filled, 15),
                "autocorr_lag_1440": autocorr(filled, 1440),
                "daily_profile_range_db": daily_profile_range(filled, 1440),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Segmentation (binary top-down SSE-based)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    start: int
    end: int


def robust_standardize(values: np.ndarray) -> np.ndarray:
    center = np.nanmedian(values, axis=0)
    scale = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    fallback = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    return np.nan_to_num((values - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def segment_sse(prefix: np.ndarray, prefix_sq: np.ndarray, start: int, end: int) -> float:
    count = end - start
    if count <= 0:
        return 0.0
    total = prefix[end] - prefix[start]
    total_sq = prefix_sq[end] - prefix_sq[start]
    return float(np.sum(total_sq - (total * total / count)))


def best_split(prefix: np.ndarray, prefix_sq: np.ndarray, segment: Segment, min_bins: int):
    if segment.end - segment.start < 2 * min_bins:
        return None, 0.0
    base = segment_sse(prefix, prefix_sq, segment.start, segment.end)
    best_idx = None
    best_gain = 0.0
    for split in range(segment.start + min_bins, segment.end - min_bins + 1):
        gain = base - segment_sse(prefix, prefix_sq, segment.start, split) - segment_sse(prefix, prefix_sq, split, segment.end)
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


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_segment(agg: dict[str, float], thresholds: dict[str, float]) -> str:
    std = agg.get("std_power_db", math.nan)
    short_std = agg.get("short_timescale_std_db", math.nan)
    occupancy = agg.get("occupancy_rate", math.nan)
    autocorr = agg.get("autocorr_lag_1440", math.nan)
    daily_range = agg.get("daily_profile_range_db", math.nan)
    mean_power = agg.get("mean_power_dbm", math.nan)

    if autocorr >= thresholds["diurnal_autocorr"] and daily_range >= thresholds["diurnal_daily_range_db"]:
        return "diurnal_pattern"
    if mean_power <= thresholds["mean_power_median_dbm"] and std <= thresholds["low_std_db"] and (math.isnan(occupancy) or occupancy <= 0.05):
        return "noise_floor"
    if not math.isnan(occupancy) and occupancy >= 0.95 and std <= thresholds["low_std_db"]:
        return "constant_occupancy"
    if not math.isnan(short_std) and short_std >= thresholds["high_short_std_db"]:
        return "bursty_short_timescale"
    if std >= thresholds["high_std_db"] or (not math.isnan(occupancy) and 0.05 < occupancy < 0.95):
        return "intermittent_occupancy"
    return "mixed_activity"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_thresholds(stats_df: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_power_median_dbm": float(stats_df["mean_power_dbm"].median()),
        "low_std_db": 0.5,
        "high_std_db": float(stats_df["std_power_db"].quantile(2 / 3)),
        "high_short_std_db": float(stats_df["short_timescale_std_db"].quantile(2 / 3)),
        "diurnal_autocorr": 0.5,
        "diurnal_daily_range_db": 3.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute frequency region definitions for POWDER data.")
    parser.add_argument("--npz", type=Path, default=ROOT / "data" / "powder_20260618T0036Z_humanities_guesthouse_600_800.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--site-index", type=int, default=-1, help="Site index (-1 = average all sites)")
    parser.add_argument("--occupancy-threshold", type=float, default=-95.0, help="dBm threshold for occupancy")
    parser.add_argument("--min-bins", type=int, default=5)
    parser.add_argument("--penalty", type=float, default=8.0)
    parser.add_argument("--max-segments", type=int, default=80)
    parser.add_argument("--chunk-id", default="powder_600_800")
    parser.add_argument("--start-mhz", type=float, default=600.0)
    parser.add_argument("--end-mhz", type=float, default=800.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    npz = np.load(args.npz)
    raw_site = npz["raw_site_data_db"]
    freqs = npz["freqs_mhz"]
    site_names = npz["site_names"]

    print(f"Loaded {args.npz.name}: raw_site_data_db shape {raw_site.shape}, {len(freqs)} freq bins")
    print(f"Sites: {list(site_names)}")

    # Average across sites or select one
    if args.site_index < 0:
        data = raw_site.mean(axis=0)
        print(f"Using average of all {raw_site.shape[0]} sites")
    else:
        data = raw_site[args.site_index]
        print(f"Using site {args.site_index} ({site_names[args.site_index]})")
    print(f"Data shape: {data.shape}")

    # Ensure (T, F)
    if data.ndim == 2 and data.shape[0] < data.shape[1] and data.shape[0] == len(freqs):
        data = data.T

    # Compute per-bin statistics
    stats_df = per_bin_statistics(data, freqs, args.occupancy_threshold)
    stats_path = output_dir / "frequency_bin_features_600_800.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Wrote per-bin statistics: {stats_path}")

    # Segmentation
    available = [c for c in SEGMENT_FEATURES if c in stats_df.columns]
    print(f"Segmentation features ({len(available)}): {available}")

    matrix = robust_standardize(stats_df[available].to_numpy(dtype=np.float64))
    segments = binary_segments(matrix, args.min_bins, args.penalty, args.max_segments)
    print(f"Found {len(segments)} segments")

    # Classify each segment
    thresholds = build_thresholds(stats_df)

    AGG_COLS = [
        "std_power_db",
        "short_timescale_std_db",
        "occupancy_rate",
        "autocorr_lag_1440",
        "daily_profile_range_db",
        "mean_power_dbm",
    ]

    region_rows = []
    for seg_idx, seg in enumerate(segments, start=1):
        seg_df = stats_df.iloc[seg.start : seg.end]
        agg = {col: float(seg_df[col].mean()) for col in AGG_COLS}
        freqs_list = seg_df["frequency_mhz"].tolist()

        region_rows.append(
            {
                "chunk_id": args.chunk_id,
                "band_id": f"region_{seg_idx:02d}",
                "start_mhz": float(seg_df["frequency_mhz"].iloc[0]),
                "end_mhz": float(seg_df["frequency_mhz"].iloc[-1]),
                "behavior_category": classify_segment(agg, thresholds),
                "bin_count": len(seg_df),
                "included_frequency_mhz": " ".join(str(round(float(f), 1)) for f in freqs_list),
            }
        )

    regions_df = pd.DataFrame(region_rows)
    output_path = output_dir / "frequency_segments_600_800.csv"
    regions_df.to_csv(output_path, index=False)
    print(f"Wrote region definitions: {output_path}")
    print("\nRegion counts by behavior:")
    for cat, count in regions_df["behavior_category"].value_counts().items():
        print(f"  {cat}: {count}")
    print(f"\nTotal: {len(regions_df)} regions covering {stats_df['frequency_mhz'].iloc[0]:.1f}–{stats_df['frequency_mhz'].iloc[-1]:.1f} MHz")


if __name__ == "__main__":
    main()
