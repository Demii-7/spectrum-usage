#!/usr/bin/env python3
"""Per-bin test-error tables of the TSS-LCD tuned finals, in the layout of ``rescored/per-bin/...`` (read by ``entropy_advantage_october.py``).

Input: records written by ``tsslcd_authors.py --eval-only --per-bin-output`` under ``--records-prefix`` (``savgol-off/seed-N.json``, field
``per_column_mae``: MAE in dB per horizon and column, raw targets, no evaluation smoothing, every test window). TSS-LCD forecasts the four core
receivers together (columns are receiver-major, 4 x 200 bins): the MAE of a bin is the mean over the four receivers' columns, as in the pooled
per-bin tables of the other models. Output: one table per seed with all horizons, ``rescored/per-bin/tsslcd/<resolution>/<band>/tuned/seed-N.csv``
(columns seed, band, resolution, model, frequency_mhz, site, split, horizon, mae_db).

    python evaluation/analysis/tsslcd_per_bin_tables.py --resolution 15min --band 600_800 --records-prefix tsslcd-authors-perbin-15min-tuned
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def per_bin_table(record: dict, band: str, resolution: str) -> pd.DataFrame:
    """Per-bin table (mean over receivers) of one record with ``per_column_mae``."""
    pc = record["per_column_mae"]
    mae = np.asarray(pc["mae_db"], dtype=float)                     # (horizons, n_sites * n_bins)
    n_sites, n_bins = int(pc["n_sites"]), int(pc["n_bins"])
    per_bin = mae.reshape(len(pc["horizons"]), n_sites, n_bins).mean(axis=1)
    freqs = np.asarray(pc["frequencies_mhz"], dtype=float)
    rows = [{"seed": record["seed"], "band": band, "resolution": resolution, "model": "tsslcd", "frequency_mhz": float(freqs[j]), "site": "unknown",
             "split": "2d_test", "horizon": int(h), "mae_db": float(per_bin[i, j])} for i, h in enumerate(pc["horizons"]) for j in range(n_bins)]
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", required=True, choices=("1min", "15min"))
    p.add_argument("--band", required=True, choices=("600_800", "2400_2600"))
    p.add_argument("--records-prefix", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--out-prefix", default="rescored/per-bin/tsslcd")
    a = p.parse_args()
    from training.ray.storage import spectrum_s3_client
    c = spectrum_s3_client()
    for seed in a.seeds:
        key = f"{a.records_prefix.strip('/')}/savgol-off/seed-{seed}.json"
        try:
            record = json.loads(c.get_object(Bucket="spectrum", Key=key)["Body"].read())
        except Exception as error:
            print(f"[tsslcd-per-bin] missing {key}: {error}", flush=True)
            continue
        if not record.get("ok") or "per_column_mae" not in record:
            print(f"[tsslcd-per-bin] {key}: not ok or no per_column_mae", flush=True)
            continue
        table = per_bin_table(record, a.band, a.resolution)
        out = f"{a.out_prefix.strip('/')}/{a.resolution}/{a.band}/tuned/seed-{seed}.csv"
        buffer = io.StringIO()
        table.to_csv(buffer, index=False)
        c.put_object(Bucket="spectrum", Key=out, Body=buffer.getvalue().encode())
        print(f"[tsslcd-per-bin] wrote {out}: {len(table)} rows, horizons {sorted(table.horizon.unique())}, mean MAE {table.mae_db.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
