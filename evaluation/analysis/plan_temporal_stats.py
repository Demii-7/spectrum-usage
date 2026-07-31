#!/usr/bin/env python3
"""Compute PLAN temporal statistics for the six-site POWDER T4 training split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SITES = ("cpg", "ebc", "humanities", "madsen", "moran", "sagepoint")
REGIONS = {
    1: (600.5, 607.5), 3: (622.5, 641.5), 4: (642.5, 646.5),
    5: (647.5, 655.5), 6: (656.5, 675.5), 8: (691.5, 728.5),
    9: (729.5, 734.5), 10: (735.5, 740.5), 11: (741.5, 745.5),
    12: (746.5, 755.5), 13: (756.5, 768.5), 14: (769.5, 776.5),
    16: (789.5, 794.5),
}
LAGS = (1, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440)


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag >= len(values):
        return np.nan
    left, right = values[:-lag], values[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/powder"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/plan_temporal_stats"))
    args = parser.parse_args()
    rows = []
    for site in SITES:
        path = args.data_root / f"{site}-nuc1" / {
            "cpg": "20260628T0437Z", "ebc": "20260628T0436Z",
            "humanities": "20260628T0436Z", "madsen": "20260628T0437Z",
            "moran": "20260628T0437Z", "sagepoint": "20260628T0437Z",
        }[site] / "600_800" / "power_1mhz_avg_per_minute.csv"
        frame = pd.read_csv(path)
        stamps = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.floor("min")
        keep = (stamps >= pd.Timestamp("2026-06-28T04:37:00Z")) & (stamps <= pd.Timestamp("2026-07-02T02:23:00Z"))
        frame = frame.loc[keep].apply(pd.to_numeric, errors="coerce").interpolate(limit=9).ffill(limit=9).bfill(limit=9)
        for column in frame.columns:
            frequency = float(column)
            values = frame[column].to_numpy(dtype=float)
            row = {"site": site, "frequency_mhz": frequency, "variance_db2": float(np.var(values))}
            correlations = {f"acf_{lag}": autocorrelation(values, lag) for lag in LAGS}
            row.update(correlations)
            row["first_lag_acf_below_0_5"] = next((lag for lag in LAGS if correlations[f"acf_{lag}"] < 0.5), np.nan)
            rows.append(row)
    per_bin = pd.DataFrame(rows)
    summary_rows = []
    for region_id, (start, end) in REGIONS.items():
        selected = per_bin[per_bin.frequency_mhz.between(start, end)]
        row = {"region_id": region_id, "start_mhz": start, "end_mhz": end, "bins": len(selected)}
        for column in ["variance_db2", *[f"acf_{lag}" for lag in LAGS], "first_lag_acf_below_0_5"]:
            row[f"median_{column}"] = selected[column].median()
            row[f"q1_{column}"] = selected[column].quantile(0.25)
            row[f"q3_{column}"] = selected[column].quantile(0.75)
        summary_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_bin.to_csv(args.output_dir / "temporal_per_bin.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "temporal_region_summary.csv", index=False)
    print(f"wrote {len(per_bin)} bin rows and {len(summary_rows)} region rows to {args.output_dir}")


if __name__ == "__main__":
    main()
