#!/usr/bin/env python3
"""Summarize retained-bin spectral ablations against an unmasked control."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONDITIONS = {"r1": (600.5, 607.5), "r5": (647.5, 655.5), "r13": (756.5, 768.5)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--masked", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/spectral_ablation_summary.csv"))
    args = parser.parse_args()
    full = pd.read_csv(args.full / "per_frequency_metrics.csv")
    rows = []
    for name, directory in args.masked:
        masked = pd.read_csv(Path(directory) / "per_frequency_metrics.csv")
        start, end = CONDITIONS[name]
        full_region = full[full.frequency_mhz.between(start, end)]
        masked_region = masked[masked.frequency_mhz.between(start, end)]
        merged = full_region.merge(masked_region, on=["frequency_mhz", "horizon"], suffixes=("_full", "_masked"))
        for horizon, group in merged.groupby("horizon"):
            rows.append({"condition": name, "horizon": horizon, "bins": len(group), "full_mae_db": group.mae_db_full.mean(), "masked_mae_db": group.mae_db_masked.mean(), "masked_minus_full_db": (group.mae_db_masked - group.mae_db_full).mean(), "full_rmse_db": group.rmse_db_full.mean(), "masked_rmse_db": group.rmse_db_masked.mean()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
