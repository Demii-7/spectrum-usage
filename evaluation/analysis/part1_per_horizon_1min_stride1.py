"""Part 1, 1 minute, every test window (stride 1): all models rescored on the common range, with the heuristics.

Sources (all ``rescore.py`` records on 2026-10-27 20:00 to 11-03 20:00 UTC, metrics from the repository's evaluator):
  * ``rescored/checkpoints-oct2026*`` (600-800 only): the October 1-minute 1D, 2D and 4D checkpoints, 5 seeds; the 1D LSTMs are thinned (one window in ten,
    ``--thin-1d-lstm``, disclosed in each record) because a 1D model scores one bin at a time;
  * ``rescored/direct/1min`` (both bands): the direct finals (OFA, Time-LLM, PatchTST, DLinear, RevIN-MLP, NLinear, residual heads) at stride 1.
4D rows are map-level errors over the interpolated grid and are NOT comparable with the 1D/2D/direct rows (listed separately).
Heuristics (tuned per horizon on validation; untuned 60-minute window) and TSS-LCD come from ``baselines_all_horizons.csv`` and ``tsslcd_all.csv``.
Writes ``evaluation/results/part1_per_horizon_1min_stride1.csv``.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

H = (1, 15, 60)


def listing(client, prefix):
    return [o["Key"] for pg in client.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix=prefix)
            for o in pg.get("Contents", []) if o["Key"].endswith(".json")]


def main() -> None:
    client = spectrum_s3_client()
    keys = listing(client, "rescored/checkpoints-oct2026") + listing(client, "rescored/direct/1min")
    with ThreadPoolExecutor(16) as pool:
        recs = list(pool.map(lambda k: (k, json.loads(client.get_object(Bucket="spectrum", Key=k)["Body"].read())), keys))
    rows = []
    for k, d in recs:
        if not d.get("ok"):
            continue
        if "/direct/1min/" in k:
            h = int(d["horizon"])
            if f"test_mae_db_t{h}" in d:
                rows.append({"band": d["band"], "family": "direct", "model": d["model"], "horizon": h, "seed": int(d["seed"]), "mae": float(d[f"test_mae_db_t{h}"]),
                             "stride": d.get("test_stride")})
        else:
            fam = "4D (map-level, not comparable)" if d["rep"] == "4d" else "oct checkpoint"
            name = f"{d['model']} ({d['rep']})"
            for h in H:
                if f"test_mae_db_t{h}" in d:
                    rows.append({"band": d["band"], "family": fam, "model": name, "horizon": h, "seed": int(d["seed"]), "mae": float(d[f"test_mae_db_t{h}"]),
                                 "stride": d.get("test_stride")})
    base = pd.read_csv("evaluation/results/baselines_all_horizons.csv")
    base = base[(base.resolution == "1min") & base.selection.isin(["per_horizon", "full_lookback"]) & base.family.isin(["mean", "median", "persistence", "kalman"])]
    for r in base.itertuples():
        tag = " (full 60 min)" if r.selection == "full_lookback" else (" (tuned)" if r.family in ("mean", "median") else "")
        rows.append({"band": r.band, "family": "heuristic", "model": r.family + tag, "horizon": int(r.horizon), "seed": 0, "mae": float(r.mae_db), "stride": 1})
    ts = pd.read_csv("evaluation/results/tsslcd_all.csv")
    ts = ts[(ts.resolution == "1min") & (ts.savgol == "off") & ts.horizon.isin([str(h) for h in H])]
    for r in ts.itertuples():
        rows.append({"band": r.band, "family": "TSS-LCD", "model": f"tsslcd ({r.setting}, savgol off)", "horizon": int(r.horizon), "seed": 0, "mae": float(r.mean), "stride": 1})
    f = pd.DataFrame(rows)
    s = f.groupby(["band", "family", "model", "horizon"]).agg(mean=("mae", "mean"), sd=("mae", "std"), n=("mae", "count")).reset_index()
    s.to_csv("evaluation/results/part1_per_horizon_1min_stride1.csv", index=False)
    pd.set_option("display.width", 200)
    for band in ("600_800", "2400_2600"):
        w = s[(s.band == band) & (s.family != "4D (map-level, not comparable)")].pivot_table(index=["family", "model"], columns="horizon", values="mean")
        if not set(H) <= set(w.columns):
            continue
        w = w[list(H)].round(3)
        w["mean_3h"] = w.mean(axis=1, skipna=False).round(3)
        w["seeds"] = s[s.band == band].groupby(["family", "model"]).n.max()
        print(f"\n{band}: 1-minute test MAE (dB) at t+1/15/60 minutes, every test window (1D LSTMs: one in ten), mean over seeds")
        print(w.sort_values("mean_3h").to_string())
    w4 = s[(s.band == "600_800") & (s.family.str.startswith("4D"))].pivot_table(index="model", columns="horizon", values="mean").round(3)
    print("\n4D map-level (600-800), not comparable with the rows above:"); print(w4.to_string())


if __name__ == "__main__":
    main()
