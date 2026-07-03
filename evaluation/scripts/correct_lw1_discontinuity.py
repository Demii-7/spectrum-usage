#!/usr/bin/env python3
"""Correct the AERPAW LW1 power discontinuity in the canonical CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "aerpaw" / "LW1" / "20220208T1757Z" / "87_6020" / "power_1mhz_avg_per_minute.csv"
DEFAULT_REPORT = ROOT / "results" / "statistical_analysis" / "lw1_discontinuity_correction.csv"
DEFAULT_OFFSETS = ROOT / "results" / "statistical_analysis" / "lw1_discontinuity_offsets.csv"
DEFAULT_DISCONTINUITY_ROW = 6840
DEFAULT_WINDOW_ROWS = 60


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def uncorrected_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_uncorrected{path.suffix}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def correction_offsets(values: pd.DataFrame, discontinuity_row: int, window_rows: int) -> pd.Series:
    if discontinuity_row < window_rows:
        raise ValueError("discontinuity row must be at least window_rows")
    if discontinuity_row + window_rows > len(values):
        raise ValueError("not enough rows after discontinuity for correction window")
    before = values.iloc[discontinuity_row - window_rows : discontinuity_row].mean(axis=0)
    after = values.iloc[discontinuity_row : discontinuity_row + window_rows].mean(axis=0)
    return after - before


def apply_correction(df: pd.DataFrame, freq_cols: list[str], discontinuity_row: int, offsets: pd.Series) -> pd.DataFrame:
    corrected = df.copy()
    corrected.loc[: discontinuity_row - 1, freq_cols] = corrected.loc[: discontinuity_row - 1, freq_cols].add(offsets, axis=1)
    return corrected


def mean_delta(values: pd.DataFrame, discontinuity_row: int, window_rows: int) -> float:
    before = values.iloc[discontinuity_row - window_rows : discontinuity_row].to_numpy(dtype=np.float64).mean()
    after = values.iloc[discontinuity_row : discontinuity_row + window_rows].to_numpy(dtype=np.float64).mean()
    return float(after - before)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Canonical LW1 power CSV to correct.")
    parser.add_argument("--discontinuity-row", type=int, default=DEFAULT_DISCONTINUITY_ROW)
    parser.add_argument("--window-rows", type=int, default=DEFAULT_WINDOW_ROWS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--offsets", type=Path, default=DEFAULT_OFFSETS)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing uncorrected backup.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    backup_path = uncorrected_path(input_path)
    report_path = resolve_path(args.report)
    offsets_path = resolve_path(args.offsets)

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if backup_path.exists():
        if args.force:
            backup_path.unlink()
            input_path.rename(backup_path)
        else:
            print(f"Using existing uncorrected backup: {backup_path}")
    else:
        input_path.rename(backup_path)

    original = pd.read_csv(backup_path)
    freq_cols = frequency_columns(original)
    if not freq_cols:
        raise ValueError("no numeric frequency columns found")
    for col in freq_cols:
        original[col] = pd.to_numeric(original[col], errors="coerce")

    values = original[freq_cols]
    offsets = correction_offsets(values, args.discontinuity_row, args.window_rows)
    corrected = apply_correction(original, freq_cols, args.discontinuity_row, offsets)
    corrected.to_csv(input_path, index=False)

    offsets_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frequency_mhz": [float(col) for col in freq_cols],
            "offset_db_added_to_rows_before_discontinuity": offsets.to_numpy(dtype=np.float64),
        }
    ).to_csv(offsets_path, index=False)

    corrected_values = corrected[freq_cols]
    original_delta = mean_delta(values, args.discontinuity_row, args.window_rows)
    corrected_delta = mean_delta(corrected_values, args.discontinuity_row, args.window_rows)
    per_bin_original_delta = values.iloc[args.discontinuity_row : args.discontinuity_row + args.window_rows].mean(axis=0) - values.iloc[
        args.discontinuity_row - args.window_rows : args.discontinuity_row
    ].mean(axis=0)
    per_bin_corrected_delta = corrected_values.iloc[args.discontinuity_row : args.discontinuity_row + args.window_rows].mean(axis=0) - corrected_values.iloc[
        args.discontinuity_row - args.window_rows : args.discontinuity_row
    ].mean(axis=0)
    report = {
        "input_csv": display_path(input_path),
        "uncorrected_csv": display_path(backup_path),
        "discontinuity_row": args.discontinuity_row,
        "window_rows": args.window_rows,
        "rows_corrected": args.discontinuity_row,
        "frequency_bins": len(freq_cols),
        "original_60row_mean_delta_db": original_delta,
        "corrected_60row_mean_delta_db": corrected_delta,
        "original_per_bin_delta_min_db": float(per_bin_original_delta.min()),
        "original_per_bin_delta_median_db": float(per_bin_original_delta.median()),
        "original_per_bin_delta_max_db": float(per_bin_original_delta.max()),
        "corrected_per_bin_delta_min_db": float(per_bin_corrected_delta.min()),
        "corrected_per_bin_delta_median_db": float(per_bin_corrected_delta.median()),
        "corrected_per_bin_delta_max_db": float(per_bin_corrected_delta.max()),
        "offsets_csv": display_path(offsets_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(report_path, index=False)

    print(f"wrote corrected canonical CSV: {input_path}")
    print(f"preserved uncorrected CSV: {backup_path}")
    print(f"wrote offsets: {offsets_path}")
    print(f"wrote report: {report_path}")
    print(f"original_delta_db={original_delta:.3f}")
    print(f"corrected_delta_db={corrected_delta:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
