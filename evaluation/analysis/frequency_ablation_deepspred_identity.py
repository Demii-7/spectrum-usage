#!/usr/bin/env python3
"""No-shift identity check for 3D-SwinSTB: recomputed real-input per-bin MAE (probe objects of ``training.ray.frequency_ablation_deepspred --probe``) against the
finals' own per-bin tables ``direct-tuning/15min/<band>/deepspred2d/per-bin/h<h>/seed-N.csv`` (pooled over the four receivers; each trained horizon on the windows valid
for THAT horizon). Probe variants: A_main (origins with lookback inside the test week, valid for t+48), B_all, C_valid_per_horizon, D_end_inclusive.

    python evaluation/analysis/frequency_ablation_deepspred_identity.py --seed 42 [--upload]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
RECEIVERS = ("humanities", "madsen", "moran", "sagepoint")


def get(client, key):
    return client.get_object(Bucket="spectrum", Key=key)["Body"].read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="frequency-ablation-2026-11-classes-deepspred/_probe")
    ap.add_argument("--band", default="600_800")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="evaluation/results/frequency_ablation_deepspred_identity.csv")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    rows = []
    for seed in a.seeds:
        for h in (4, 12, 24, 48):
            try:
                z = np.load(io.BytesIO(get(client, f"{a.prefix}/deepspred2d/{a.band}/seed-{seed}/h{h}/per_origin.npz")), allow_pickle=True)
                stored = pd.read_csv(io.BytesIO(get(client, f"direct-tuning/15min/{a.band}/deepspred2d/per-bin/h{h}/seed-{seed}.csv")))
            except Exception as e:  # noqa: BLE001
                print("missing", seed, h, str(e)[:60])
                continue
            stored = stored[stored["split"].astype(str).str.endswith("test")].sort_values("frequency_mhz")
            ref = stored["mae_db"].to_numpy(float)
            variants = sorted({k.split("|")[1] for k in z.files if k.startswith("probe|") and k.endswith("|sum")})
            for v in variants:
                s = sum(z[f"probe|{v}|{r}|h{h}|sum"] for r in RECEIVERS)
                n = sum(z[f"probe|{v}|{r}|h{h}|n"] for r in RECEIVERS)
                mine = s / n
                d = np.abs(mine - ref)
                rows.append({"band": a.band, "model": "deepspred2d", "seed": seed, "horizon": h, "variant": v, "max_abs_diff_db": float(d.max()),
                             "mean_abs_diff_db": float(d.mean()), "n_bins": len(ref), "origins_per_receiver_mean": float(np.mean(n) / 1) / len(RECEIVERS) if False else float(np.mean(n))})
    table = pd.DataFrame(rows)
    table.to_csv(a.out, index=False)
    print(table.to_string(index=False))
    if a.upload:
        client.put_object(Bucket="spectrum", Key="analysis-outputs/" + Path(a.out).name, Body=Path(a.out).read_bytes())
        print("uploaded", a.out)


if __name__ == "__main__":
    main()
