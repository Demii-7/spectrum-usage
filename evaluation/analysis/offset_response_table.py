"""Summarize the constant-offset response test (``training.ray.offset_response``) from the durable bucket.

Reads ``offset-response-oct2026/<band>/<receiver>/<rep>/<model>/seed-N/summary.json`` and prints, per band and model, the gain
(forecast shift / input offset) at several offsets and horizons, averaged over seeds and receivers. Gain 1 = the level is carried through
exactly, 0 = ignored. Also writes the long CSV (one row per run x offset x horizon).

    python evaluation/analysis/offset_response_table.py [--horizon 12] [--out evaluation/results/offset_response.csv]
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

PREFIX = "offset-response-oct2026/"


def load(bucket: str = "spectrum", prefix: str = PREFIX) -> pd.DataFrame:
    client = spectrum_s3_client()
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for o in page.get("Contents", []) if o["Key"].endswith("/summary.json")]
    with ThreadPoolExecutor(16) as pool:
        records = list(pool.map(lambda k: json.loads(client.get_object(Bucket=bucket, Key=k)["Body"].read()), keys))
    rows = []
    for r in records:
        if not r.get("ok"):
            continue
        for row in r["rows"]:
            rows.append({"band": r["band"], "receiver": r["receiver"], "rep": r["rep"], "model": r["model"], "seed": r["seed"],
                         "intrinsic": r.get("intrinsic_window_normalization"), **row})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--out", default="evaluation/results/offset_response.csv")
    parser.add_argument("--prefix", default=PREFIX, help="offset-response-direct-oct2026/ for the direct models (one checkpoint per trained horizon)")
    args = parser.parse_args()
    frame = load(prefix=args.prefix)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(f"{frame.groupby(['band', 'rep', 'model', 'seed', 'receiver']).ngroups} runs")
    for band in sorted(frame.band.unique()):
        d = frame[(frame.band == band) & (frame.horizon == args.horizon)]
        t = d.pivot_table(index=["rep", "model"], columns="offset_db", values="gain", aggfunc="mean").round(3)
        t.columns = [f"{c:+g}" for c in t.columns]
        t["n_runs"] = d.groupby(["rep", "model"]).apply(lambda g: g.drop_duplicates(["receiver", "seed"]).shape[0], include_groups=False)
        print(f"\n{band}: mean gain of the forecast shift to an input offset of c dB at t+{args.horizon} (columns: c); 1 = level carried exactly")
        print(t.to_string())


if __name__ == "__main__":
    main()
