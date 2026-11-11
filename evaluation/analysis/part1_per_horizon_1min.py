"""Part 1, 1 minute: per-horizon test MAE (t+1, 15, 60 minutes) of the direct finals, TSS-LCD and the heuristics on the October test week.

Direct finals: ``direct-tuning/1min/<band>/<model>/finals/h<h>/seed-N.json`` (one model per horizon; **test windows thinned 15x**, as
``tune_direct`` scored them; rescoring at stride 1 is a separate step, ``rescored/direct/1min/``). Heuristics: ``baselines_all_horizons.csv`` (tuned per
horizon on validation, and untuned full 60-minute window). TSS-LCD: ``tsslcd_all.csv`` (authors' defaults and tuned, savgol off).
Writes ``evaluation/results/part1_per_horizon_1min.csv`` and prints the table.
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


def main() -> None:
    client = spectrum_s3_client()
    keys = [o["Key"] for pg in client.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix="direct-tuning/1min/")
            for o in pg.get("Contents", []) if "/finals/" in o["Key"] and o["Key"].endswith(".json")]
    with ThreadPoolExecutor(16) as pool:
        recs = list(pool.map(lambda k: (k, json.loads(client.get_object(Bucket="spectrum", Key=k)["Body"].read())), keys))
    rows = []
    for k, d in recs:
        if not d.get("ok"):
            continue
        p = k.split("/")
        band, model, h = p[2], p[3], int(p[5][1:])
        if f"test_mae_db_t{h}" in d:
            rows.append({"band": band, "family": "direct", "model": model, "horizon": h, "seed": int(d["seed"]), "mae": float(d[f"test_mae_db_t{h}"])})
    base = pd.read_csv("evaluation/results/baselines_all_horizons.csv")
    base = base[(base.resolution == "1min") & base.selection.isin(["per_horizon", "full_lookback"]) & base.family.isin(["mean", "median", "persistence", "kalman", "ewma"])]
    for r in base.itertuples():
        tag = " (full 60 min)" if r.selection == "full_lookback" else (" (tuned)" if r.family in ("mean", "median") else "")
        rows.append({"band": r.band, "family": "heuristic", "model": r.family + tag, "horizon": int(r.horizon), "seed": 0, "mae": float(r.mae_db)})
    ts = pd.read_csv("evaluation/results/tsslcd_all.csv")
    ts = ts[(ts.resolution == "1min") & (ts.savgol == "off") & ts.horizon.isin([str(h) for h in H])]
    for r in ts.itertuples():
        rows.append({"band": r.band, "family": "TSS-LCD", "model": f"tsslcd ({r.setting}, savgol off)", "horizon": int(r.horizon), "seed": 0, "mae": float(r.mean)})
    frame = pd.DataFrame(rows)
    s = frame.groupby(["band", "family", "model", "horizon"]).mae.agg(mean="mean", sd="std", n="count").reset_index()
    s.to_csv("evaluation/results/part1_per_horizon_1min.csv", index=False)
    pd.set_option("display.width", 200)
    for band in ("600_800", "2400_2600"):
        w = s[s.band == band].pivot_table(index=["family", "model"], columns="horizon", values="mean")[list(H)].round(3)
        w["mean_3h"] = w.mean(axis=1, skipna=False).round(3)
        w["seeds"] = s[s.band == band].groupby(["family", "model"]).n.max()
        print(f"\n{band}: 1-minute test MAE (dB) at t+1/15/60 minutes (direct finals: windows thinned 15x), mean over seeds")
        print(w.sort_values("mean_3h").to_string())


if __name__ == "__main__":
    main()
