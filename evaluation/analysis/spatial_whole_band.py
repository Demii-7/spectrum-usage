"""Whole-band physical-receiver errors from the per-region files of the spatial evaluations.

``region_metrics.csv`` holds MAE and RMSE per frequency region, including the noise-floor regions, with the number of finite
bin-window pairs in each. The evaluator's own ``mean_horizon_mae_db`` averages the non-noise regions only; every other table in the
benchmark averages all bins. This recomputes the all-bin numbers exactly (count-weighted MAE; RMSE from count-weighted squares), per
cell and per (cell, horizon, receiver), so the 1-minute and 15-minute spatial results use the same definition.

    python evaluation/analysis/spatial_whole_band.py --prefix spatial-oct2026/600_800 --out evaluation/results/spatial_oct2026_whole_band.csv
"""
from __future__ import annotations

import argparse
import io
import os

import boto3
import numpy as np
import pandas as pd

KEYS = ["model", "seed", "condition", "permutation_seed", "horizon", "receiver"]


def whole_band(regions: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, seed, condition, permutation, horizon, receiver) with all-bin MAE and RMSE."""
    frame = regions.copy()
    frame["permutation_seed"] = frame["permutation_seed"].fillna(-1)
    frame["abs_sum"] = frame["mae_db"] * frame["count"]
    frame["sq_sum"] = frame["rmse_db"] ** 2 * frame["count"]
    grouped = frame.groupby(KEYS, as_index=False).agg(count=("count", "sum"), abs_sum=("abs_sum", "sum"), sq_sum=("sq_sum", "sum"),
                                                      n_origins=("n_origins", "first"))
    grouped["mae_db_all_bins"] = grouped["abs_sum"] / grouped["count"]
    grouped["rmse_db_all_bins"] = np.sqrt(grouped["sq_sum"] / grouped["count"])
    return grouped.drop(columns=["abs_sum", "sq_sum"])


def cell_summary(per_receiver: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: all-bin MAE and RMSE averaged over horizons and receivers (mean of the per-receiver values)."""
    keys = ["model", "seed", "condition", "permutation_seed"]
    return per_receiver.groupby(keys, as_index=False).agg(mae_db_all_bins=("mae_db_all_bins", "mean"),
                                                           rmse_db_all_bins=("rmse_db_all_bins", "mean"), n_origins=("n_origins", "first"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--bucket", default="spectrum")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    client = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    frames = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=args.prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("region_metrics.csv"):
                frames.append(pd.read_csv(io.BytesIO(client.get_object(Bucket=args.bucket, Key=obj["Key"])["Body"].read())))
    per_receiver = whole_band(pd.concat(frames, ignore_index=True))
    per_receiver.to_csv(args.out.replace(".csv", "_per_receiver.csv"), index=False)
    cell_summary(per_receiver).to_csv(args.out, index=False)
    print(f"{len(frames)} cells -> {args.out}")


if __name__ == "__main__":
    main()
