#!/usr/bin/env python3
"""Plot full spectral context beside a region-only ablation input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.common.data import apply_2d_spectral_mask

from spectrogram_colormap import ensure_minimum_db_span


NOISE_RANGES = [[608.5, 621.5], [676.5, 690.5], [777.5, 788.5], [795.5, 799.5]]
DEFAULT_REGIONS = ROOT / "evaluation" / "analysis" / "plan_regions_600_800.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV containing the plotted trace")
    parser.add_argument(
        "--calibration-file",
        type=Path,
        action="append",
        required=True,
        help="Training CSV used to calibrate the low-tail replacement; repeatable",
    )
    parser.add_argument("--region", default="R13", help="Retained region ID (default: R13)")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--start", required=True, help="UTC timestamp for the first plotted minute")
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--split-seed-offset", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "results" / "figures" / "spectral_context_ablation_R13.png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def frequency_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if column == "timestamp_utc":
            continue
        try:
            float(column)
        except (TypeError, ValueError):
            continue
        columns.append(column)
    return sorted(columns, key=float)


def read_trace(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    columns = frequency_columns(frame)
    if not columns or "timestamp_utc" not in frame.columns:
        raise ValueError(f"{path} must contain timestamp_utc and numeric frequency columns")
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    valid = timestamps.notna()
    return (
        pd.DatetimeIndex(timestamps[valid]),
        np.asarray([float(column) for column in columns]),
        frame.loc[valid, columns].to_numpy(dtype=np.float32),
    )


def region_bounds(path: Path, region_id: str) -> tuple[float, float]:
    regions = pd.read_csv(path)
    selected = regions[regions["band_id"].astype(str) == region_id]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one region named {region_id!r} in {path}")
    row = selected.iloc[0]
    if str(row["is_noise_floor"]).lower() in {"true", "1"}:
        raise ValueError(f"{region_id} is marked as a noise-floor region")
    return float(row["start_mhz"]), float(row["end_mhz"])


def select_window(
    timestamps: pd.DatetimeIndex,
    values: np.ndarray,
    start: str,
    minutes: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if minutes < 1:
        raise ValueError("--minutes must be positive")
    requested = pd.Timestamp(start)
    if requested.tzinfo is None:
        requested = requested.tz_localize("UTC")
    else:
        requested = requested.tz_convert("UTC")
    index = int(np.argmin(np.abs((timestamps - requested).asi8)))
    stop = index + minutes
    if stop > len(timestamps):
        raise ValueError(f"requested window extends past the end of {len(timestamps)} available rows")
    return timestamps[index:stop], values[index:stop]


def extent_for(times: pd.DatetimeIndex, frequencies: np.ndarray) -> list[float]:
    frequency_step = float(np.median(np.diff(frequencies))) if len(frequencies) > 1 else 1.0
    return [
        mdates.date2num(times[0].to_pydatetime()),
        mdates.date2num(times[-1].to_pydatetime()),
        float(frequencies[0] - frequency_step / 2),
        float(frequencies[-1] + frequency_step / 2),
    ]


def main() -> int:
    args = parse_args()
    times, frequencies, values = read_trace(args.input)
    calibration_frames = [read_trace(path)[2] for path in args.calibration_file]
    if any(frame.shape[1] != values.shape[1] for frame in calibration_frames):
        raise ValueError("input and calibration files must have the same frequency bins")
    calibration = np.concatenate(calibration_frames, axis=0)
    start_mhz, end_mhz = region_bounds(args.regions, args.region)
    window_times, full_context = select_window(times, values, args.start, args.minutes)
    masked_context = apply_2d_spectral_mask(
        full_context,
        frequencies,
        {
            "frequency_ranges": [[start_mhz, end_mhz]],
            "replacement": "low_tail_gaussian",
            "calibration_frequency_ranges": NOISE_RANGES,
            "low_tail_quantile": 0.05,
            "minimum_noise_std_db": 0.05,
            "seed": args.seed,
        },
        training_data=calibration,
        split_seed_offset=args.split_seed_offset,
    )

    finite = np.concatenate([full_context[np.isfinite(full_context)], masked_context[np.isfinite(masked_context)]])
    vmin, vmax = ensure_minimum_db_span(*np.percentile(finite, [1, 99]))
    extent = extent_for(window_times, frequencies)
    cmap = plt.get_cmap("viridis").copy()
    figure, axes = plt.subplots(1, 2, figsize=(4.0, 1.0), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for axis, data, title in zip(
        axes,
        (full_context, masked_context),
        ("Full context", "Region only (other regions masked)"),
    ):
        image = axis.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title, fontsize=5.5, fontweight="bold", pad=2)
        axis.set_xlabel("UTC time", fontsize=5)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=window_times.tz))
        axis.tick_params(top=True, right=True, length=2, labelsize=4)
    axes[0].set_ylabel("Frequency (MHz)", fontsize=5)
    axes[1].text(
        0.02,
        0.97,
        f"retained: {start_mhz:g}–{end_mhz:g} MHz",
        transform=axes[1].transAxes,
        va="top",
        color="white",
        fontsize=4,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
    )
    colorbar = figure.colorbar(image, ax=axes, pad=0.02, shrink=0.92)
    colorbar.set_label("Power (dB)", fontsize=5)
    colorbar.ax.tick_params(labelsize=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
