#!/usr/bin/env python3
"""Consolidate retained Ray per-frequency metrics for replicated 2D runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


MODELS = (
    "autoformer_csa",
    "linearar2d",
    "residuallinearar2d",
    "lstmattn",
    "residualvanillalstm",
    "temporalconvnet",
    "vanillalstm",
)
SEEDS = (40, 41, 42, 43, 44)
SEED_PATTERN = re.compile(r"_seed=(\d+)_")


def collect(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()

    for path in sorted(input_root.glob("**/evaluation/per_frequency_metrics.csv")):
        model = next((name for name in MODELS if f"/{name}-replicate/" in str(path)), None)
        match = SEED_PATTERN.search(str(path))
        if model is None or match is None:
            continue
        seed = int(match.group(1))
        if seed not in SEEDS:
            continue
        key = (model, seed)
        if key in seen:
            continue
        seen.add(key)
        frame = pd.read_csv(path)
        required = {"frequency_mhz", "horizon", "mae_db", "rmse_db"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["seed"] = seed
        frame["model"] = model
        frame = frame.rename(columns={"frequency_mhz": "bin_mhz"})
        leading = ["model", "seed"]
        frame = frame[leading + [column for column in frame.columns if column not in leading]]
        rows.append(frame)

    for model in MODELS:
        for seed in SEEDS:
            matches = [frame for frame in rows if frame.iloc[0]["model"] == model and int(frame.iloc[0]["seed"]) == seed]
            coverage.append({
                "model": model,
                "seed": seed,
                "status": "available" if matches else "missing",
                "row_count": int(len(matches[0])) if matches else 0,
            })

    if not rows:
        raise RuntimeError(f"No retained Ray metrics found under {input_root}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coverage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    args = parser.parse_args()

    metrics, coverage = collect(args.input_root)
    metrics = metrics.sort_values(["model", "seed", "horizon", "bin_mhz"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    coverage.to_csv(args.coverage_output, index=False)
    print(f"Wrote {len(metrics)} metric rows from {coverage['status'].eq('available').sum()} model-seed runs")
    print(f"Wrote coverage to {args.coverage_output}")


if __name__ == "__main__":
    main()
