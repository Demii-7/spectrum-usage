#!/usr/bin/env python3
"""Bin classes by daily periodicity (fixed before any ablation result is looked at).

Source: evaluation/results/forecastability_oct15_train.csv (October 15-minute TRAINING week; one row per band, receiver, bin).
Class per (band, receiver, bin):
  noise   : the bin lies in a plan region with is_noise_floor = true (plan_regions_<band>.csv; region membership by bin centre frequency)
  periodic (P) : not noise and 24-hour autocorrelation acf_lag96 >= 0.6
  nonperiodic (N): not noise and acf_lag96 <= 0.4
  middle  : everything else (kept in the file, excluded from P/N contrasts)
daily_profile_r2 is carried along for a sensitivity check; the thresholds are NOT to be moved after results are seen.

    python evaluation/analysis/bin_periodicity_classes.py [--out evaluation/results/bin_periodicity_classes.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P_MIN, N_MAX = 0.6, 0.4


def classes(source: Path) -> pd.DataFrame:
    table = pd.read_csv(source)
    out = []
    for band, part in table.groupby("band"):
        plan = pd.read_csv(HERE / f"plan_regions_{band}.csv")
        freq = part["bin_label"].to_numpy(dtype=float)
        region = np.full(len(part), "", dtype=object)
        noise = np.zeros(len(part), dtype=bool)
        for row in plan.itertuples():
            inside = (freq >= row.start_mhz - 1e-9) & (freq <= row.end_mhz + 1e-9)
            region[inside] = row.band_id
            noise[inside] = str(row.is_noise_floor).lower() == "true"
        acf = part["acf_lag96"].to_numpy()
        cls = np.where(noise, "noise", np.where(acf >= P_MIN, "P", np.where(acf <= N_MAX, "N", "middle")))
        out.append(part.assign(region=region, is_noise_floor=noise, **{"class": cls})
                   [["band", "receiver", "bin", "bin_label", "region", "is_noise_floor", "acf_lag96", "daily_profile_r2", "class"]])
    return pd.concat(out, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(HERE.parent / "results" / "forecastability_oct15_train.csv"))
    p.add_argument("--out", default=str(HERE.parent / "results" / "bin_periodicity_classes.csv"))
    a = p.parse_args()
    table = classes(Path(a.source))
    table.to_csv(a.out, index=False)
    counts = table.pivot_table(index=["band", "receiver"], columns="class", values="bin", aggfunc="count", fill_value=0)
    print(counts.to_string())
    print(table.pivot_table(index="band", columns="class", values="bin", aggfunc="count", fill_value=0).to_string())
    odd = table[(table["class"].isin(["P", "N"])) & (table["acf_lag96"] >= P_MIN) & (table["daily_profile_r2"] <= 0.1)]
    print("P bins with daily_profile_r2 <= 0.1 (slow drift, no daily cycle):", len(odd), "(brief: 61)")


if __name__ == "__main__":
    main()
