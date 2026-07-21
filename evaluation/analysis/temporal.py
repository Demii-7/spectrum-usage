#!/usr/bin/env python3
"""Compute temporal statistics for annotated spectrum regions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POWER_ROOT = REPOSITORY_ROOT / "evaluation"
DEFAULT_ANNOTATION_ROOT = REPOSITORY_ROOT / "data" / "annotations"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "evaluation" / "results" / "tables" / "temporal.csv"
POWER_FILE = "power_1mhz_avg_per_minute.csv"
POWER_DIFF_EDGES_DB = np.asarray((-20, -10, -5, -2, 0, 2, 5, 10, 20), dtype=float)
AUTOCORRELATION_LAG = 1440


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def quantized_power_diff_entropy(power: np.ndarray) -> float:
    valid_pairs = np.isfinite(power[:-1]) & np.isfinite(power[1:])
    differences = power[1:][valid_pairs] - power[:-1][valid_pairs]
    if differences.size == 0:
        return math.nan

    state_count = POWER_DIFF_EDGES_DB.size - 1
    states = np.clip(
        np.searchsorted(POWER_DIFF_EDGES_DB, differences, side="right") - 1,
        0,
        state_count - 1,
    )
    counts = np.bincount(states, minlength=state_count)
    probabilities = counts[counts > 0] / states.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def lag_1440_autocorrelation(power: np.ndarray) -> float:
    if power.size <= AUTOCORRELATION_LAG:
        return math.nan
    left = power[:-AUTOCORRELATION_LAG]
    right = power[AUTOCORRELATION_LAG:]
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def fano_predictability_bound(entropy_bits: float) -> float:
    if not math.isfinite(entropy_bits):
        return math.nan

    state_count = POWER_DIFF_EDGES_DB.size - 1
    entropy = float(np.clip(entropy_bits, 0.0, math.log2(state_count)))
    low, high = 0.0, (state_count - 1.0) / state_count
    for _ in range(80):
        error = (low + high) / 2.0
        binary_entropy = 0.0 if error == 0.0 else (
            -error * math.log2(error) - (1.0 - error) * math.log2(1.0 - error)
        )
        if binary_entropy + error * math.log2(state_count - 1) < entropy:
            low = error
        else:
            high = error
    return 1.0 - (low + high) / 2.0


def numeric_frequency_columns(frame: pd.DataFrame) -> dict[str, float]:
    columns = {}
    for column in frame.columns:
        try:
            columns[column] = float(column)
        except (TypeError, ValueError):
            continue
    return columns


def compute_region_rows(
    dataset: str,
    node: str,
    annotations: pd.DataFrame,
    power_root: Path,
) -> list[dict[str, object]]:
    rows = []
    for (run_id, band), regions in annotations.groupby(["run_id", "band"], sort=True):
        source = power_root / dataset / node / str(run_id) / str(band) / POWER_FILE
        if not source.is_file():
            raise FileNotFoundError(f"missing power trace for annotations: {source}")

        frame = pd.read_csv(source)
        frequencies = numeric_frequency_columns(frame)
        if not frequencies:
            raise ValueError(f"no numeric frequency columns in {source}")

        for region in regions.itertuples(index=False):
            selected = [
                column
                for column, frequency in frequencies.items()
                if region.region_start_mhz <= frequency <= region.region_end_mhz
            ]
            if not selected:
                raise ValueError(
                    f"no bins in annotated region {dataset}/{node}/{run_id}/{band}/"
                    f"{region.region_id}: {region.region_start_mhz}-{region.region_end_mhz} MHz"
                )

            values = frame[selected].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            region_power = np.nanmedian(values, axis=1)
            entropy = quantized_power_diff_entropy(region_power)
            rows.append(
                {
                    "dataset": dataset,
                    "node": node,
                    "run_id": run_id,
                    "band": band,
                    "region_id": region.region_id,
                    "region_start_mhz": region.region_start_mhz,
                    "region_end_mhz": region.region_end_mhz,
                    "bin_count": len(selected),
                    "sample_count": int(np.count_nonzero(np.isfinite(region_power))),
                    "power_diff_entropy_bits": entropy,
                    "autocorrelation_lag_1440": lag_1440_autocorrelation(region_power),
                    "fano_predictability_bound": fano_predictability_bound(entropy),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    annotation_paths = sorted(args.annotation_root.glob("*/*.csv"))
    if not annotation_paths:
        raise FileNotFoundError(f"no annotation CSVs under {args.annotation_root}")

    required = {"run_id", "band", "region_id", "region_start_mhz", "region_end_mhz"}
    for path in annotation_paths:
        annotations = pd.read_csv(path, dtype={"run_id": str, "band": str})
        missing = required.difference(annotations.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows.extend(compute_region_rows(path.parent.name, path.stem, annotations, args.power_root))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} region rows to {args.output}")


if __name__ == "__main__":
    main()
