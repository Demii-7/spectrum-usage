"""TSS-LCD (the authors' released model; MAE and MSE) on the October test week: every result set in one table, with the heuristics for reference.

Result sets (raw-target MAE, no eval smoothing, mean over seeds 40-44 with seed sd; savgol = the authors' pre-filter on the series):
  1 minute  600-800   untuned ``tsslcd-authors-full`` (off, centered, causal), tuned winner ``tsslcd-authors-tuned`` (off)
  1 minute  2400-2600 ``tsslcd-authors-full-2400`` (off, causal)
  15 minute 600-800   authors' defaults ``tsslcd-authors-15min-default`` (off, causal), tuned ``tsslcd-authors-15min-tuned`` (off)
  15 minute 2400-2600 ``tsslcd-authors-15min-default-2400`` (off, causal)
Heuristics from ``evaluation/results/baselines_all_horizons.csv``: lookback mean and median, tuned (window chosen on validation, per horizon)
and untuned (full lookback window).  Writes ``evaluation/results/tsslcd_all.csv``.

    python evaluation/analysis/tsslcd_table.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

SETS = [("1min", "600_800", "authors defaults", "tsslcd-authors-full", ("off", "centered", "causal")),
        ("1min", "600_800", "tuned winner", "tsslcd-authors-tuned", ("off",)),
        ("1min", "2400_2600", "authors defaults", "tsslcd-authors-full-2400", ("off", "causal")),
        ("15min", "600_800", "authors defaults", "tsslcd-authors-15min-default", ("off", "causal")),
        ("15min", "600_800", "tuned winner", "tsslcd-authors-15min-tuned", ("off",)),
        ("15min", "2400_2600", "authors defaults", "tsslcd-authors-15min-default-2400", ("off", "causal")),
        ("15min", "2400_2600", "tuned winner", "tsslcd-authors-15min-tuned-2400", ("off",))]
HORIZONS = {"1min": (1, 15, 60), "15min": (1, 4, 12, 24, 48)}


def main() -> None:
    client = spectrum_s3_client()
    jobs = [(res, band, setting, prefix, sg, seed) for res, band, setting, prefix, sgs in SETS for sg in sgs for seed in range(40, 45)]

    def read(j):
        res, band, setting, prefix, sg, seed = j
        try:
            d = json.loads(client.get_object(Bucket="spectrum", Key=f"{prefix}/savgol-{sg}/seed-{seed}.json")["Body"].read())
        except Exception:
            return []
        if not d.get("ok"):
            return []
        m = d["metrics"]
        rows = []
        for metric, key in (("mae", "mae_db"), ("mse", "mse_db2")):
            for h in HORIZONS[res]:
                rows.append({"resolution": res, "band": band, "setting": setting, "savgol": sg, "seed": seed, "horizon": h, "metric": metric,
                             "value": m.get(f"raw_target__off__{key}_t{h}")})
            rows.append({"resolution": res, "band": band, "setting": setting, "savgol": sg, "seed": seed, "horizon": "mean", "metric": metric,
                         "value": m.get(f"raw_target__off__mean_horizon_{key}")})
        return rows

    with ThreadPoolExecutor(16) as pool:
        frame = pd.DataFrame([r for rows in pool.map(read, jobs) for r in rows])
    frame["horizon"] = frame["horizon"].astype(str)
    summary = frame.groupby(["resolution", "band", "setting", "savgol", "horizon", "metric"]).value.agg(mean="mean", sd="std", n="count").reset_index()
    summary.to_csv("evaluation/results/tsslcd_all.csv", index=False)
    base = pd.read_csv("evaluation/results/baselines_all_horizons.csv")
    pd.set_option("display.width", 230)
    for (res, band, metric), g in summary.groupby(["resolution", "band", "metric"]):
        hs = [str(h) for h in HORIZONS[res]] + ["mean"]
        wide = g.pivot_table(index=["setting", "savgol"], columns="horizon", values="mean")[hs].round(3)
        wide["sd(mean)"] = g[g.horizon == "mean"].set_index(["setting", "savgol"])["sd"].round(3)
        wide["seeds"] = g[g.horizon == "mean"].set_index(["setting", "savgol"])["n"]
        print(f"\n{res} {band}: TSS-LCD raw-target {metric.upper()} per horizon, mean over seeds")
        print(wide.to_string())
        if metric == "mae":
            b = base[(base.resolution == res) & (base.band == band) & base.family.isin(["mean", "median"]) & base.selection.isin(["per_horizon", "full_lookback"])]
            t = b.pivot_table(index=["family", "selection"], columns="horizon", values="mae_db").round(3)
            t["mean"] = t.mean(axis=1).round(3)
            print("heuristics (MAE):"); print(t.to_string())

if __name__ == "__main__":
    main()
