"""Per-horizon baseline table from the median-study sweep records: every family, every horizon, all five metrics.

For each (resolution, band, family) the parameter (window K for the mean and median, gain for the EWMA; the train mean, persistence and
per-bin Kalman filter have none) is chosen on the VALIDATION range, either separately for each horizon (``per_horizon``) or once for all
horizons by the mean validation MAE (``one_parameter``, the convention of a one-step model that is rolled out), and the TEST metrics at
that parameter are reported. Persistence is the mean over a window of 1. The windows are those of the model tables (the repository's
own pipeline), so the rows sit next to the learned models'.

    python evaluation/analysis/baseline_table.py --prefix baselines-all-horizons --out evaluation/results/baselines_all_horizons.csv
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

METRICS = {"mae_db": "test_mae_db_t{h}", "mse_db2": "test_mse_db2_t{h}", "rmse_db": "test_rmse_db_t{h}",
           "mape_linear_pct": "test_mape_linear_pct_t{h}", "mape_dbm_pct": "test_mape_dbm_pct_t{h}"}
SELECT_HORIZONS = {"1min": (1, 15, 60), "15min": (4, 12, 24, 48)}
FAMILIES = {"train_mean": ("train_mean", None), "persistence": ("mean", 1), "mean": ("mean", None), "median": ("median", None),
            "ewma": ("ewma", None), "kalman": ("kalman", None)}


def rows_for(record_rows: list[dict], rng: str, k: int | None = None) -> list[dict]:
    return [r for r in record_rows if r["range"] == rng and (k is None or r["k"] == k)]


def choose(rows: list[dict], horizons, mode: str, horizon: int):
    """The parameter key k with the lowest validation MAE (one horizon, or the mean over ``horizons``)."""
    val = rows_for(rows, "validation")
    if not val:
        raise ValueError("no validation rows")
    if mode == "per_horizon":
        return min(val, key=lambda r: (r[f"mae_t{horizon}"], r["k"]))["k"]
    return min(val, key=lambda r: (sum(r[f"mae_t{h}"] for h in horizons) / len(horizons), r["k"]))["k"]


def table(sweeps: dict[tuple, dict]) -> pd.DataFrame:
    out = []
    for (resolution, band, statistic), record in sorted(sweeps.items()):
        for family, (stat, fixed_k) in FAMILIES.items():
            if stat != statistic:
                continue
            rows = record["rows"]
            horizons = sorted(int(key.split("_t")[-1]) for key in rows[0]["metrics"] if key.startswith("test_mae_db_t"))
            candidates = rows if fixed_k is None else [r for r in rows if r["k"] == fixed_k]
            if not candidates:
                continue
            for mode in ("per_horizon", "one_parameter"):
                for h in horizons:
                    k = choose(candidates, SELECT_HORIZONS[resolution], mode, h)
                    test = next(r for r in rows_for(candidates, "test", k))
                    row = {"resolution": resolution, "band": band, "family": family, "selection": mode, "horizon": h, "parameter_key": k}
                    row.update({name: test["metrics"][pattern.format(h=h)] for name, pattern in METRICS.items()})
                    out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    import boto3

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--bucket", default="spectrum")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    client = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    sweeps = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=args.prefix.rstrip("/") + "/sweep/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if obj["Key"].endswith(".json") and len(parts) == 5:
                record = json.loads(client.get_object(Bucket=args.bucket, Key=obj["Key"])["Body"].read())
                if record.get("ok"):
                    sweeps[(parts[2], parts[3], parts[4].removesuffix(".json"))] = record
    frame = table(sweeps)
    frame.to_csv(args.out, index=False)
    pivot = frame[frame.selection == "per_horizon"].pivot_table(index=["resolution", "band", "family"], columns="horizon", values="mae_db")
    print(pivot.round(3).to_string())
    print(f"{len(frame)} rows -> {args.out}")


if __name__ == "__main__":
    main()
