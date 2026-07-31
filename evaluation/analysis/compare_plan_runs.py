#!/usr/bin/env python3
"""Combine completed PLAN run metrics into a compact comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/plan_basic_summary"))
    args = parser.parse_args()
    frames = []
    for run in args.run:
        for path in sorted(run.glob("*/aggregate_metrics.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame["run"] = run.name
            frames.append(frame)
    if not frames:
        raise RuntimeError("no aggregate metrics found")
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics[["run", "model", "horizon", "mae_db", "rmse_db", "n_targets"]]
    metrics = metrics.sort_values(["horizon", "mae_db", "model"])
    winners = metrics.loc[metrics.groupby("horizon")["mae_db"].idxmin()].sort_values("horizon")
    pivot = metrics.pivot_table(index="model", columns="horizon", values="mae_db", aggfunc="mean").reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "basic_metrics_long.csv", index=False)
    winners.to_csv(args.output_dir / "basic_winners.csv", index=False)
    pivot.to_csv(args.output_dir / "basic_mae_by_model_horizon.csv", index=False)
    print(f"wrote {len(metrics)} metric rows to {args.output_dir}")


if __name__ == "__main__":
    main()
