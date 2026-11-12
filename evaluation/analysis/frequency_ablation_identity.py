#!/usr/bin/env python3
"""No-shift identity check of the frequency ablation against stored results (small S3 files only).

(1) Per origin: the recomputed real-input error over all 200 bins (``allbin|<receiver>`` of ``per_origin.npz`` in the classes prefix) against the stored per-origin
    file ``origin-errors-oct2026/15min/<band>/<receiver>/direct/<model>/h<h>/seed-N/per_origin.csv`` (direct models; matched on origin time).
(2) Per bin: the pooled per-bin error of the recomputed errors against ``rescored/per-bin/...`` (stored per-bin table), for the origin set that the stored table uses:
    direct models score every trained horizon on the windows valid for THAT horizon, rolled-out models on one set for all horizons.

    python evaluation/analysis/frequency_ablation_identity.py --prefix frequency-ablation-2026-11-classes --out evaluation/results/frequency_ablation_identity.csv
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
RECEIVERS = ("humanities", "madsen", "moran", "sagepoint")
DIRECT = ("revinmlp", "revinmlp_mix", "patchtst", "patchtst_joint", "itransformer", "nlinear", "dlinear", "linearar2d", "residuallinearar2d")


def get(client, key):
    return client.get_object(Bucket="spectrum", Key=key)["Body"].read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="frequency-ablation-2026-11-classes")
    ap.add_argument("--models", nargs="+", default=["revinmlp", "dlinear", "nlinear", "patchtst", "patchtst_joint", "revinmlp_mix", "itransformer"])
    ap.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="evaluation/results/frequency_ablation_identity.csv")
    a = ap.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    rows = []
    for model in a.models:
        for band in a.bands:
            for h in (4, 12, 24, 48):
                key = f"{a.prefix}/{model}/{band}/seed-{a.seed}/h{h}/per_origin.npz"
                try:
                    z = np.load(io.BytesIO(get(client, key)), allow_pickle=True)
                except Exception:
                    continue
                diffs, n_match, n_mine, n_stored = [], 0, 0, 0
                for r in RECEIVERS:
                    mine = pd.Series(z[f"allbin|{r}"][:, 0].astype(float), index=pd.to_datetime(z[f"times|{r}"], utc=True))
                    try:
                        st = pd.read_csv(io.BytesIO(get(client, f"origin-errors-oct2026/15min/{band}/{r}/direct/{model}/h{h}/seed-{a.seed}/per_origin.csv")))
                    except Exception:
                        continue
                    st = st[st["horizon"] == h] if "horizon" in st else st
                    tcol = "time" if "time" in st else st.columns[0]
                    st = pd.Series(st["mae_db"].to_numpy(float), index=pd.to_datetime(st[tcol], utc=True))
                    both = mine.index.intersection(st.index)
                    n_mine += len(mine)
                    n_stored += len(st)
                    n_match += len(both)
                    diffs.append((mine.loc[both] - st.loc[both]).abs().to_numpy())
                if diffs:
                    d = np.concatenate(diffs)
                    rows.append({"band": band, "model": model, "seed": a.seed, "horizon": h, "check": "per-origin all-bin MAE vs stored per_origin.csv",
                                 "max_abs_diff_db": float(d.max()), "mean_abs_diff_db": float(d.mean()), "n_matched_origins": n_match,
                                 "n_recomputed_origins": n_mine, "n_stored_origins": n_stored})
    table = pd.DataFrame(rows)
    table.to_csv(a.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
