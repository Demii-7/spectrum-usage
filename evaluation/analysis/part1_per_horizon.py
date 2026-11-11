"""Part 1: per-horizon 15-minute test MAE for every model family on the October test week (2026-10-27 20:00 to 11-03 20:00 UTC), both bands.

Sources (all on the common range, all test windows):
  * direct finals ``direct-tuning/15min/<band>/<model>/finals/h<h>/seed-N.json`` (one model trained per horizon; OFA, Time-LLM, PatchTST and
    variants, DLinear, NLinear, RevIN-MLP, residual heads, TimeRAN/MOMENT linear probe),
  * tuned rolled-out 1D/2D models: ``rescored/p15tuned/<band>/<rep>/<model>/seed-N.json`` (t+4..t+48) and ``rescored-h1/p15tuned/...`` (t+1),
  * heuristics: ``evaluation/results/baselines_all_horizons.csv`` (15 minutes; parameter chosen on validation per horizon),
  * TSS-LCD: ``evaluation/results/tsslcd_new_runs.csv`` (authors' defaults, raw-target MAE; savgol off and causal).
Mean over the available seeds and the seed standard deviation. Writes ``evaluation/results/part1_per_horizon_15min.csv``.

    python evaluation/analysis/part1_per_horizon.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

BUCKET, HORIZONS, BANDS = "spectrum", (1, 4, 12, 24, 48), ("600_800", "2400_2600")


def listing(client, prefix: str) -> list[str]:
    return [o["Key"] for pg in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
            for o in pg.get("Contents", []) if o["Key"].endswith(".json")]


def read_all(client, keys: list[str]) -> list[tuple[str, dict]]:
    def one(k):
        try:
            return k, json.loads(client.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        except Exception:
            return k, {}
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, keys))


def collect() -> pd.DataFrame:
    client = spectrum_s3_client()
    rows = []
    # direct finals: direct-tuning/15min/<band>/<model>/finals/h<h>/seed-N.json
    for k, d in read_all(client, [k for k in listing(client, "direct-tuning/15min/") if "/finals/" in k]):
        if not d.get("ok"):
            continue
        parts = k.split("/")
        band, model, h = parts[2], parts[3], int(parts[5][1:])
        if f"test_mae_db_t{h}" in d:
            rows.append({"family": "direct", "band": band, "model": model, "horizon": h, "seed": int(d["seed"]), "mae": float(d[f"test_mae_db_t{h}"])})
    # rolled-out tuned models, rescored on the common range
    for prefix, hs in (("rescored/p15tuned/", (4, 12, 24, 48)), ("rescored-h1/p15tuned/", (1,))):
        for k, d in read_all(client, listing(client, prefix)):
            if not d.get("ok"):
                continue
            for h in hs:
                if f"test_mae_db_t{h}" in d:
                    rows.append({"family": "rolled-out", "band": d["band"], "model": f"{d['model']} ({d['rep']})", "horizon": h,
                                 "seed": int(d["seed"]), "mae": float(d[f"test_mae_db_t{h}"])})
    base = pd.read_csv("evaluation/results/baselines_all_horizons.csv")
    base = base[(base.resolution == "15min") & base.selection.isin(["per_horizon", "full_lookback"])]
    for r in base.itertuples():
        if r.selection == "full_lookback":                       # untuned: the statistic over the whole 96-step lookback window
            rows.append({"family": "heuristic (untuned)", "band": r.band, "model": f"{r.family} (full lookback, 96)", "horizon": int(r.horizon), "seed": 0,
                         "mae": float(r.mae_db)})
        else:
            rows.append({"family": "heuristic", "band": r.band, "model": r.family + (" (tuned)" if r.family in ("mean", "median") else ""),
                         "horizon": int(r.horizon), "seed": 0, "mae": float(r.mae_db)})
    ts = pd.read_csv("evaluation/results/tsslcd_new_runs.csv")
    ts = ts[(ts.res == "15min") & ts.metric.str.startswith("mae_db_t")]
    for r in ts.itertuples():
        rows.append({"family": "TSS-LCD", "band": "600_800", "model": f"tsslcd_authors (savgol {r.savgol})", "horizon": int(r.metric.split("_t")[-1]),
                     "seed": int(r.seed), "mae": float(r.mae)})
    return pd.DataFrame(rows)


def main() -> None:
    frame = collect()
    summary = (frame.groupby(["band", "family", "model", "horizon"]).mae.agg(mean="mean", sd="std", n_seeds="count").reset_index())
    summary["sd"] = summary["sd"].fillna(0.0)
    summary.to_csv("evaluation/results/part1_per_horizon_15min.csv", index=False)
    pd.set_option("display.width", 220)
    for band in BANDS:
        s = summary[(summary.band == band) & summary.horizon.isin(HORIZONS)]
        wide = s.pivot_table(index=["family", "model"], columns="horizon", values="mean").round(3)
        long = [h for h in (4, 12, 24, 48)]                         # the convention of the paper's tables (t+1 is shown separately; TimeRAN/MOMENT have no t+1)
        wide["mean_4h"] = wide[long].mean(axis=1, skipna=False).round(3)       # NaN if a horizon is missing (incomplete rescoring)
        wide["seeds"] = s.groupby(["family", "model"]).n_seeds.max()
        print(f"\n{band}: 15-minute test MAE (dB) per horizon (steps of 15 min), mean over seeds")
        print(wide.sort_values("mean_4h").to_string())


if __name__ == "__main__":
    main()
