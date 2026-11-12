"""Table of the target-trained RevIN-MLP at COSMOS / AERPAW / ARA (records of ``training.ray.transfer_external_target_trained``).

Long format like the transfer tables: platform, site, run, band, model, horizon, metric (MAE / MSE / RMSE, dB), mean, sd over seeds, n_seeds,
n_test_origins, first_test_target_time. Rows for the models ``revinmlp_target_trained``, ``persistence`` and ``lookbackmean`` (the last two on exactly the
same origins). Also pooled rows (platform = ALL per band). Reads the bucket; run on the cluster (see run_external_target_trained_table.sh).

    python evaluation/analysis/transfer_external_target_trained.py --out evaluation/results/transfer_external_target_trained.csv
"""
import argparse
import sys
sys.path.insert(0, ".")
import json
import os

import numpy as np
import pandas as pd

PREFIX = "transfer-external-target-trained"


def load(client, bucket, prefix):
    rows = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for o in page.get("Contents", []):
            if o["Key"].endswith(".json") and not o["Key"].endswith("-error.json"):
                rows.append(json.loads(client.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()))
    return [r for r in rows if r.get("ok")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="evaluation/results/transfer_external_target_trained.csv")
    p.add_argument("--bucket", default=os.environ.get("SPECTRUM_S3_BUCKET", "spectrum"))
    p.add_argument("--prefix", default=PREFIX)
    a = p.parse_args()
    from training.ray.storage import spectrum_s3_client

    records = load(spectrum_s3_client(), a.bucket, a.prefix)
    out = []
    for r in records:
        for model, key in (("revinmlp_target_trained", "model"), ("persistence", "persistence"), ("lookbackmean", "lookbackmean")):
            mae, mse = r[f"{key}_mae_db"], r[f"{key}_mse_db"]
            for metric, value in (("MAE", mae), ("MSE", mse), ("RMSE", float(np.sqrt(mse)))):
                out.append({"platform": r["platform"], "site": r["site"], "run": r["run"], "band": r["band"], "model": model,
                            "horizon": r["horizon"], "metric": metric, "seed": r["seed"], "value": value,
                            "n_test_origins": r["n_test_origins"], "first_test_target_time": r["first_test_target_time"]})
    frame = pd.DataFrame(out)
    if frame.empty:
        raise SystemExit("no ok records")
    keys = ["platform", "site", "run", "band", "model", "horizon", "metric"]
    per = frame.groupby(keys).agg(mean=("value", "mean"), sd=("value", lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")),
                                  n_seeds=("seed", "nunique"), n_test_origins=("n_test_origins", "first"),
                                  first_test_target_time=("first_test_target_time", "first")).reset_index()
    # pooled over targets (each target's seed-mean, equal weight per target) per platform and over all platforms
    pooled = []
    for platform_label, sub in [(pl, per[per.platform == pl]) for pl in sorted(per.platform.unique())] + [("ALL", per)]:
        g = sub.groupby(["band", "model", "horizon", "metric"])["mean"].agg(["mean", "count"]).reset_index()
        for row in g.to_dict("records"):
            pooled.append({"platform": platform_label, "site": "ALL", "run": "ALL", "band": row["band"], "model": row["model"],
                           "horizon": row["horizon"], "metric": row["metric"], "mean": row["mean"], "sd": float("nan"),
                           "n_seeds": int(row["count"]), "n_test_origins": np.nan, "first_test_target_time": ""})
    table = pd.concat([per, pd.DataFrame(pooled)], ignore_index=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    table.to_csv(a.out, index=False)
    print(f"{len(records)} records -> {a.out}")
    print(table[(table.site == "ALL") & (table.metric == "MAE")].to_string())


if __name__ == "__main__":
    main()
