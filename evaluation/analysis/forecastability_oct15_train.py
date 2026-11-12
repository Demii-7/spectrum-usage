#!/usr/bin/env python3
"""Per-bin lag-96 autocorrelation (24 h at 15 minutes) and daily-profile R^2 on the October 15-minute TRAINING run
(data/powder15/<node>/20261015T1434Z/<band>/power_15min.csv) of the four core receivers, both bands.

Uses ``autocorrelation`` and ``daily_profile_r2`` of ``forecastability_atlas.py`` unchanged (daily_profile_r2 is the cross-validated,
leave-one-day-out share of variance explained by the time-of-day profile; 96 steps per day). Output: one row per band, receiver, bin
(columns ``acf_lag96``, ``daily_profile_r2``, ``n_valid``) and a summary per band and receiver.

    python evaluation/analysis/forecastability_oct15_train.py --out evaluation/results/forecastability_oct15_train.csv [--upload]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecastability_atlas import autocorrelation, daily_profile_r2, read_trace  # noqa: E402

NODES = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1")
RUN = "20261015T1434Z"
BANDS = ("600_800", "2400_2600")
STEPS_PER_DAY = 96


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="evaluation/results/forecastability_oct15_train.csv")
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    rows = []
    for band in BANDS:
        for node in NODES:
            path = Path(a.root) / "data/powder15" / node / RUN / band / "power_15min.csv"
            frame = read_trace(path, step_minutes=15)
            values = frame.to_numpy(dtype=float)
            acf = autocorrelation(values, STEPS_PER_DAY)
            r2 = daily_profile_r2(frame, STEPS_PER_DAY)
            valid = np.isfinite(values).sum(axis=0)
            for i, column in enumerate(frame.columns):
                rows.append({"band": band, "receiver": node.split("-")[0], "bin": i, "bin_label": column, "acf_lag96": float(acf[i]),
                             "daily_profile_r2": float(r2[i]), "n_valid": int(valid[i]), "n_steps": len(frame),
                             "first_time": str(frame.index[0]), "last_time": str(frame.index[-1])})
    table = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    summary = table.groupby(["band", "receiver"]).agg(acf_mean=("acf_lag96", "mean"), acf_median=("acf_lag96", "median"),
                                                      r2_mean=("daily_profile_r2", "mean"), r2_median=("daily_profile_r2", "median"),
                                                      n_bins=("bin", "count"), n_steps=("n_steps", "first"),
                                                      first_time=("first_time", "first"), last_time=("last_time", "first")).reset_index()
    pooled = table.groupby("band").agg(acf_mean=("acf_lag96", "mean"), acf_median=("acf_lag96", "median"),
                                       r2_mean=("daily_profile_r2", "mean"), r2_median=("daily_profile_r2", "median")).reset_index()
    summary_path = str(Path(a.out).with_name(Path(a.out).stem + "_summary.csv"))
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(pooled.to_string(index=False))
    if a.upload:
        sys.path.insert(0, ".")
        from training.ray.storage import spectrum_s3_client
        c = spectrum_s3_client()
        for f in (a.out, summary_path):
            c.put_object(Bucket="spectrum", Key="analysis-outputs/" + Path(f).name, Body=open(f, "rb").read())
            print("uploaded", f)


if __name__ == "__main__":
    main()
