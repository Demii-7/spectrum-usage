"""Aggregate the frozen-transfer results on COSMOS, AERPAW and ARA (``training.ray.transfer_external``) from the durable bucket.

Reads ``transfer15-external-oct2026/{A,B,baseline}/**/summary.json`` (stage A frozen, stage B per-window centering, persistence and
96-step lookback mean), writes one long CSV (run x horizon) and prints, per platform and band at one horizon, the mean over targets
(each target counted once, seeds averaged first) of MAE for every model in stages A and B, with persistence and the lookback mean.

    python evaluation/analysis/transfer_external_table.py [--horizon 12] [--out evaluation/results/transfer_external_stage_ab.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

PREFIX = "transfer15-external-oct2026/"


def load(bucket: str = "spectrum") -> pd.DataFrame:
    client = spectrum_s3_client()
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=PREFIX)
            for o in page.get("Contents", []) if o["Key"].endswith("/summary.json")]
    with ThreadPoolExecutor(16) as pool:
        summaries = list(pool.map(lambda k: json.loads(client.get_object(Bucket=bucket, Key=k)["Body"].read()), keys))
    rows = []
    for s in summaries:
        if not s.get("metrics"):
            continue
        for horizon, m in s["metrics"].items():
            rows.append({"stage": s.get("stage", "baseline"), "platform": s["platform"], "site": s["site"], "run": s["run"],
                         "target_band": s["target_band"], "rep": s.get("rep", "baseline"), "model": s.get("model", s.get("baseline")),
                         "seed": s.get("seed"), "horizon": int(horizon), "n_origins": s["n_origins"],
                         "missing_share": s.get("missing_share_before_interpolation"), "mae_db": m["mae_db"], "bias_db": m["bias_db"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--out", default="evaluation/results/transfer_external_stage_ab.csv")
    args = parser.parse_args()
    frame = load()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    d = frame[frame.horizon == args.horizon]
    # one value per target and model (seeds averaged), then the mean over targets
    per_target = d.groupby(["platform", "target_band", "site", "run", "stage", "rep", "model"]).mae_db.mean().reset_index()
    pd.set_option("display.width", 220)
    for (platform, band), g in per_target.groupby(["platform", "target_band"]):
        n = g[["site", "run"]].drop_duplicates().shape[0]
        base = g[g.stage == "baseline"].groupby("model").mae_db.mean().round(3).to_dict()
        t = g[g.stage != "baseline"].pivot_table(index=["rep", "model"], columns="stage", values="mae_db", aggfunc="mean")
        t["B-A"] = t["B"] - t["A"]
        t["A/lbm"] = t["A"] / base["lookbackmean"]
        print(f"\n{platform} {band} ({n} targets): MAE (dB) at t+{args.horizon}; baselines {base}")
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()
