"""June-July POWDER ("Separate collection") transfer, all models and regimes: aggregation of the two result sets.

Sources (durable bucket):
  * ``transfer15-july-2026/{A,B,baseline}/powder/<site>/<run>/<band>/...``: the 11 core 1D/2D models, regimes A and B, and the persistence and
    lookback-mean baselines (``training.ray.transfer_july``),
  * ``transfer-all-july-2026/<regime>/powder/<site>/<run>/<band>/<rep>/<model>[/h<h>]/seed-N/``: regimes C and Cm of the 1D/2D models and the direct and
    large models (``training.ray.transfer_all --intervals evaluation/analysis/transfer_july_intervals.csv --no-core-powder``).
Each target counts once per (model, regime) after averaging seeds; a group's value is the mean over its targets. Groups: ``core4`` (humanities,
madsen, moran, sagepoint) and ``other`` (guesthouse, cpg, ebc); ``--by-run`` splits the 06-28 run from the 07-03 run. Rows are printed only where
every target of the group is scored (``complete``); incomplete groups are in the CSV.

    python evaluation/analysis/transfer_july_all_table.py [--by-run] [--cache evaluation/results/transfer_july_all_raw.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BUCKET = "spectrum"
H4 = (4, 12, 24, 48)
CORE4 = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1")
OLD, NEW = "transfer15-july-2026/", "transfer-all-july-2026/"


def site_group(site: str) -> str:
    return "core4" if site in CORE4 else "other"


def parse_key(key: str) -> dict | None:
    """Fields of a summary.json key of either result set (None if it is not a summary of a model or baseline)."""
    parts = key.split("/")
    if len(parts) < 3 or not key.endswith("/summary.json"):
        return None
    root, regime = parts[0] + "/", parts[1]
    if root not in (OLD, NEW) or parts[2] != "powder":
        return None
    if regime == "baseline":                                    # transfer15-july-2026/baseline/powder/<site>/<run>/<band>/<name>/summary.json
        if len(parts) != 8:
            return None
        site, run, band, name = parts[3:7]
        return {"regime": "baseline", "site": site, "run": run, "band": band, "rep": "baseline", "model": name, "horizon_tag": None}
    if len(parts) < 10:                                         # <root>/<regime>/powder/<site>/<run>/<band>/<rep>/<model>[/h<h>]/seed-N/summary.json
        return None
    site, run, band, rep, model = parts[3:8]
    tag = next((int(p[1:]) for p in parts[8:-2] if p.startswith("h") and p[1:].isdigit()), None)
    return {"regime": regime, "site": site, "run": run, "band": band, "rep": rep, "model": model, "horizon_tag": tag}


def load_raw(client) -> pd.DataFrame:
    keys = [o["Key"] for prefix in (OLD, NEW) for pg in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
            for o in pg.get("Contents", []) if o["Key"].endswith("/summary.json")]

    def read(k):
        meta = parse_key(k)
        if meta is None:
            return []
        d = json.loads(client.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        return [{**meta, "target": f"{meta['site']}/{meta['run']}/{meta['band']}", "seed": d.get("seed", 0), "horizon": int(h), "mae": m["mae_db"]}
                for h, m in d["metrics"].items()]

    with ThreadPoolExecutor(48) as pool:
        return pd.DataFrame([row for rows in pool.map(read, keys) for row in rows])


def aggregate(raw: pd.DataFrame, by_run: bool = False) -> pd.DataFrame:
    """Seed-average per target, then mean over the targets of a group; mean_4h only if all four horizons exist for the model."""
    raw = raw.copy()
    raw["group"] = raw.site.map(site_group)
    raw["run_day"] = raw.run.str[:8]
    keys = ["group", "band", "regime", "rep", "model"] + (["run_day"] if by_run else [])
    per_target = raw.groupby(keys + ["target", "horizon"]).mae.mean().reset_index()
    mean = per_target.groupby(keys + ["horizon"]).agg(mae=("mae", "mean"), n_targets=("target", "nunique")).reset_index()
    expected = (per_target[per_target.regime == "baseline"].groupby(["group", "band"] + (["run_day"] if by_run else []))
                .target.nunique().rename("n_expected").reset_index())
    mean = mean.merge(expected, on=["group", "band"] + (["run_day"] if by_run else []), how="left")
    t12 = mean[mean.horizon == 12].rename(columns={"mae": "mae_t12"}).drop(columns="horizon")
    sub = mean[mean.horizon.isin(H4)]
    full = sub.groupby(keys).filter(lambda g: g.horizon.nunique() == 4)
    m4 = full.groupby(keys).mae.mean().rename("mae_mean_4h").reset_index()
    out = t12.merge(m4, on=keys, how="left")
    out["complete"] = out.n_targets == out.n_expected
    return out


def main() -> None:
    from training.ray.storage import spectrum_s3_client

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--by-run", action="store_true")
    parser.add_argument("--out", default="evaluation/results/transfer_july_all_regimes.csv")
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()
    if args.cache and Path(args.cache).exists():
        raw = pd.read_csv(args.cache)
    else:
        raw = load_raw(spectrum_s3_client())
        if args.cache:
            raw.to_csv(args.cache, index=False)
    table = aggregate(raw, args.by_run)
    table.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["group"] + (["run_day"] if args.by_run else [])
    for band in sorted(table.band.unique()):
        t = table[(table.band == band) & (table.regime != "baseline") & table.complete]
        b = table[(table.band == band) & (table.regime == "baseline")]
        print(f"\n{band}: MAE at t+12 (dB), mean over the targets of a group, seeds averaged; baselines:")
        print(b.pivot_table(index="model", columns=cols, values="mae_t12").round(3).to_string())
        print(t.pivot_table(index=["model", "rep", "regime"], columns=cols, values="mae_t12").round(3).to_string())


if __name__ == "__main__":
    main()
