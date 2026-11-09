"""The t+15-minute (one-step, 15-minute resolution) cells: direct finals trained for horizon 1 and rolled-out models rescored at horizon 1.

Sources (all on the common test range 2026-10-27 20:00 to 11-03 20:00 UTC):
  * direct models: ``direct-tuning/15min/<band>/<model>/finals/h1/seed-N.json`` (trained for one step with the existing winners)
  * rolled-out models: ``rescored-h1/p15tuned/<band>/<rep>/<model>/seed-N.json`` (existing checkpoints, horizon 1 added to the windowing)
Mean and seed sd over the available seeds, all five metrics. The baselines at t+15 are in ``baselines_all_horizons.csv`` (horizon 1, 15min).

    python evaluation/analysis/t15_table.py --out evaluation/results/t15_cells.csv
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st

import boto3
import pandas as pd

METRICS = {"mae_db": "test_mae_db_t1", "mse_db2": "test_mse_db2_t1", "rmse_db": "test_rmse_db_t1",
           "mape_linear_pct": "test_mape_linear_pct_t1", "mape_dbm_pct": "test_mape_dbm_pct_t1"}


def summarize(records: list[dict], keys: dict[str, str]) -> dict:
    """Mean over the records for each metric, the seed sd of the MAE and the number of seeds."""
    out = {name: sum(r[key] for r in records) / len(records) for name, key in keys.items()}
    out["mae_sd"] = st.pstdev([r[keys["mae_db"]] for r in records]) if len(records) > 1 else 0.0
    out["n_seeds"] = len(records)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default="spectrum")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    client = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])

    def listing(prefix):
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def load(key):
        return json.loads(client.get_object(Bucket=args.bucket, Key=key)["Body"].read())

    rows = []
    groups: dict[tuple, list[dict]] = {}
    for key in listing("direct-tuning/15min/"):
        parts = key.split("/")
        if key.endswith(".json") and len(parts) == 7 and parts[4] == "finals" and parts[5] == "h1":
            record = load(key)
            if record.get("ok") and METRICS["mae_db"] in record:
                groups.setdefault(("direct", parts[2], "2d direct", parts[3]), []).append(record)
    for key in listing("rescored-h1/"):
        if key.endswith(".json") and "/per-bin/" not in key:
            record = load(key)
            if record.get("ok") and METRICS["mae_db"] in record:
                groups.setdefault(("rolled-out", record["band"], record["rep"], record["model"]), []).append(record)
    for (kind, band, rep, model), records in sorted(groups.items()):
        rows.append({"kind": kind, "band": band, "rep": rep, "model": model, **summarize(records, METRICS)})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    print(frame.pivot_table(index=["kind", "rep", "model"], columns="band", values="mae_db").round(3).to_string())
    print(f"{len(frame)} rows -> {args.out}")


if __name__ == "__main__":
    main()
