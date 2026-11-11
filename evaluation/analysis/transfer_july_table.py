"""Aggregate the frozen-transfer results on the POWDER June-July 2026 collection (``training.ray.transfer_july``) from the durable bucket.

Reads ``transfer15-july-2026/{A,B,baseline}/**/summary.json`` (stage A frozen, stage B per-window centering, persistence and the 96-step
lookback mean). Prints, per band and receiver group (core four = humanities, madsen, moran, sagepoint; other = guesthouse, cpg, ebc), the MAE at
one horizon, averaged over targets (each receiver-run counted once, seeds averaged first), for every model in stages A and B, with the
baselines and the ratio to the lookback mean. Also splits by run (the 06-28 run and the 07-03 run).

    python evaluation/analysis/transfer_july_table.py [--horizon 12] [--out evaluation/results/transfer_july_stage_ab.csv]
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

PREFIX = "transfer15-july-2026/"
CORE = {"humanities", "madsen", "moran", "sagepoint"}


def load(bucket: str = "spectrum") -> pd.DataFrame:
    client = spectrum_s3_client()
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=PREFIX)
            for o in page.get("Contents", []) if o["Key"].endswith("/summary.json")]
    with ThreadPoolExecutor(16) as pool:
        records = list(pool.map(lambda k: json.loads(client.get_object(Bucket=bucket, Key=k)["Body"].read()), keys))
    rows = []
    for s in records:
        if not s.get("metrics"):
            continue
        receiver = s["site"].removesuffix("-nuc1")
        for horizon, m in s["metrics"].items():
            rows.append({"stage": s.get("stage", "baseline"), "receiver": receiver, "group": "core four" if receiver in CORE else "guesthouse/cpg/ebc",
                         "run": s["run"], "run_day": s["run"][:8], "target_band": s["target_band"], "rep": s.get("rep", "baseline"),
                         "model": s.get("model", s.get("baseline")), "seed": s.get("seed"), "horizon": int(horizon), "n_origins": s["n_origins"],
                         "mae_db": m["mae_db"], "bias_db": m["bias_db"]})
    return pd.DataFrame(rows)


def show(d: pd.DataFrame, horizon: int, title: str) -> None:
    per_target = d.groupby(["receiver", "run", "stage", "rep", "model"]).mae_db.mean().reset_index()
    n = per_target[["receiver", "run"]].drop_duplicates().shape[0]
    base = per_target[per_target.stage == "baseline"].groupby("model").mae_db.mean().round(3).to_dict()
    t = per_target[per_target.stage != "baseline"].pivot_table(index=["rep", "model"], columns="stage", values="mae_db", aggfunc="mean")
    t["B-A"] = t["B"] - t["A"]
    t["A/lbm"] = t["A"] / base["lookbackmean"]
    print(f"\n{title} ({n} targets): MAE (dB) at t+{horizon}; baselines {base}")
    print(t.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--out", default="evaluation/results/transfer_july_stage_ab.csv")
    args = parser.parse_args()
    frame = load()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    d = frame[frame.horizon == args.horizon]
    pd.set_option("display.width", 220)
    for band in sorted(d.target_band.unique()):
        for group in ("core four", "guesthouse/cpg/ebc"):
            show(d[(d.target_band == band) & (d.group == group)], args.horizon, f"{band}, {group}")
    print("\n--- by run (all receivers) ---")
    for band in sorted(d.target_band.unique()):
        for day in sorted(d.run_day.unique()):
            show(d[(d.target_band == band) & (d.run_day == day)], args.horizon, f"{band}, run {day}")


if __name__ == "__main__":
    main()
