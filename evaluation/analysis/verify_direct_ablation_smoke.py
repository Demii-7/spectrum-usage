"""Smoke check of the direct-model spectral ablation: does ``full_context`` (region loss) reproduce the tuned final's own per-bin error over the region?

Compares, per (model, band, region, horizon), the smoke record's test MAE over the region's bins with the MAE over the same bins from the
tuned final's per-bin table (``direct-tuning/15min/<band>/<model>/per-bin/h<h>/seed-<seed>.csv``, full-band loss) for the same seed, and prints
the spread of the finals over seeds 40-44 as the seed-noise yardstick. Needs the SPECTRUM_S3_* environment.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="oct2026-spectral-context-15min-strong-smoke")
args = parser.parse_args()
client = spectrum_s3_client()
rows = []
for page in client.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix=args.prefix + "/"):
    for o in page.get("Contents", []):
        if o["Key"].endswith(".json") and not o["Key"].endswith(".error.json"):
            r = json.loads(client.get_object(Bucket="spectrum", Key=o["Key"])["Body"].read())
            if r.get("ok") and r["condition"] == "full_context":
                rows.append(r)
cache = {}


def per_bin(model, band, h, seed):
    key = (model, band, h, seed)
    if key not in cache:
        body = client.get_object(Bucket="spectrum", Key=f"direct-tuning/15min/{band}/{model}/per-bin/h{h}/seed-{seed}.csv")["Body"].read()
        cache[key] = pd.read_csv(io.BytesIO(body))
    return cache[key]


out = []
for r in rows:
    h, seed = int(r["horizon"]), int(r["seed"])
    table = per_bin(r["model"], r["band"], h, seed)
    col = next(c for c in table.columns if c in ("frequency_mhz", "frequency", "freq_mhz"))
    mae = next(c for c in table.columns if "mae" in c and "mape" not in c)
    sel = table[(table[col] >= r["start_mhz"] - 1e-6) & (table[col] <= r["end_mhz"] + 1e-6)]
    final = float(sel[mae].mean())
    seeds = [float(per_bin(r["model"], r["band"], h, s).pipe(lambda t: t[(t[col] >= r["start_mhz"] - 1e-6) & (t[col] <= r["end_mhz"] + 1e-6)][mae].mean()))
             for s in range(40, 45)]
    value = float(r[f"test_mae_db_t{h}"])
    out.append({"model": r["model"], "band": r["band"], "region": r["region"], "h": h, "seed": seed, "full_context": value, "final_same_seed": final,
                "diff": value - final, "final_seed_sd": float(np.std(seeds, ddof=1))})
frame = pd.DataFrame(out).sort_values(["model", "band", "region", "h"])
pd.set_option("display.width", 200)
print(frame.round(4).to_string(index=False))
print("\nmean |diff|:", round(frame["diff"].abs().mean(), 4), " mean diff:", round(frame["diff"].mean(), 4), " mean seed sd of the finals:", round(frame.final_seed_sd.mean(), 4))
print("cells within 2 sd of the finals' seed spread:", int((frame["diff"].abs() <= 2 * frame.final_seed_sd).sum()), "of", len(frame))
