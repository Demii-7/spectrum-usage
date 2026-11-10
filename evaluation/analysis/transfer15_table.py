"""Aggregate the 15-minute frozen transfer results (stage A frozen zero-shot, stage B per-window centering) from the durable bucket.

Reads ``<prefix>/**/summary.json`` for stage A (``transfer15-oct2026``, including the two baselines) and stage B
(``transfer15-oct2026-B``), writes one long CSV (one row per run x horizon) and prints the MAE table at one horizon, averaged over the
held-out receivers and the five seeds. Needs the usual SPECTRUM_S3_* environment (the training storage client).

    python evaluation/analysis/transfer15_table.py [--horizon 12] [--out evaluation/results/transfer15_stage_ab.csv]
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

STAGES = (("A", "transfer15-oct2026/"), ("B", "transfer15-oct2026-B/"))


def load(bucket: str = "spectrum") -> pd.DataFrame:
    client = spectrum_s3_client()
    rows = []
    for stage, prefix in STAGES:
        keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
                for o in page.get("Contents", []) if o["Key"].endswith("/summary.json")]
        with ThreadPoolExecutor(16) as pool:
            summaries = list(pool.map(lambda k: json.loads(client.get_object(Bucket=bucket, Key=k)["Body"].read()), keys))
        for s in summaries:
            for horizon, m in s["metrics"].items():
                rows.append({"stage": stage, "source_band": s.get("source_band"), "target_band": s["target_band"],
                             "receiver": s["target_receiver"], "rep": s.get("rep", "baseline"), "model": s.get("model", s.get("baseline")),
                             "seed": s.get("seed"), "horizon": int(horizon), "n_origins": s["n_origins"],
                             "intrinsic_window_normalization": s.get("intrinsic_window_normalization"),
                             "centering_applied": s.get("centering_applied"), **{k: m[k] for k in (
                                 "mae_db", "mse_db2", "rmse_db", "bias_db", "mape_linear_pct", "mape_dbm_pct")}})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--out", default="evaluation/results/transfer15_stage_ab.csv")
    args = parser.parse_args()
    frame = load()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    for band in sorted(frame.target_band.unique()):
        d = frame[(frame.target_band == band) & (frame.horizon == args.horizon)]
        print(f"\n{band}: MAE (dB) at t+{args.horizon}; baselines", d[d.rep == "baseline"].groupby("model").mae_db.mean().round(3).to_dict())
        t = d[d.rep != "baseline"].pivot_table(index=["rep", "model"], columns="stage", values="mae_db", aggfunc="mean").round(3)
        t["B-A"] = (t["B"] - t["A"]).round(3)
        print(t.to_string())


if __name__ == "__main__":
    main()
