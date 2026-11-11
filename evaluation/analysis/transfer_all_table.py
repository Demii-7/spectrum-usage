"""All final POWDER 15-minute models in transfer regimes A, B and C: aggregation of ``training.ray.transfer_all`` and the earlier stage A/B runs.

Sources (durable bucket): ``transfer-all-oct2026/<regime>/...`` (this runner: direct models A/B/C/Cm, rolled-out models C/Cm),
``transfer15-oct2026`` and ``transfer15-oct2026-B`` (rolled-out 1D/2D, POWDER guesthouse and law73, A and B),
``transfer15-external-oct2026/{A,B,baseline}`` (rolled-out 1D/2D, COSMOS/AERPAW/ARA, A and B, and the baselines on all targets).
Regimes: A frozen; B per-window centering; C causal normalizer refit (mean and std, calibration = 24 h before the first scored origin);
Cm the same with the mean only (definitions: ``training/common/transfer_regimes.py``). Full-intrinsic-normalization models (RevIN-MLP,
PatchTST, OFA, Time-LLM, MOMENT, TimeRAN) are run in A only (B and C are identical by construction; verified on a subset, printed).
Each target counts once per (model, regime) after averaging seeds; a group's value is the mean over its targets.

    python evaluation/analysis/transfer_all_table.py [--cache evaluation/results/transfer_all_raw.csv] [--out evaluation/results/transfer_all_regimes.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

BUCKET = "spectrum"
H4 = (4, 12, 24, 48)


def keys(client, prefix: str) -> list[str]:
    return [o["Key"] for pg in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
            for o in pg.get("Contents", []) if o["Key"].endswith("/summary.json")]


def parse_all(d: dict, regime: str | None = None) -> list[dict]:
    group = "powder" if d["platform"] == "powder" else d["platform"]
    base = {"group": group, "target": d["target"], "band": d["target_band"], "regime": d["regime"], "rep": d["rep"], "model": d["model"],
            "seed": d["seed"], "trained_horizon": d.get("trained_horizon")}
    return [{**base, "horizon": int(h), "mae": m["mae_db"]} for h, m in d["metrics"].items()]


def load_raw(client) -> pd.DataFrame:
    jobs = []        # (kind, key)
    for k in keys(client, "transfer-all-oct2026/"):
        jobs.append(("all", k))
    for k in keys(client, "transfer15-oct2026/"):
        jobs.append(("old_powder_A", k))
    for k in keys(client, "transfer15-oct2026-B/"):
        jobs.append(("old_powder_B", k))
    for k in keys(client, "transfer15-external-oct2026/"):
        jobs.append(("old_ext", k))

    def read(job):
        kind, k = job
        d = json.loads(client.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        parts = k.split("/")
        rows = []
        if kind == "all":
            return parse_all(d)
        if kind in ("old_powder_A", "old_powder_B"):
            band, receiver, rep, model = parts[1], parts[2], parts[3], parts[4]
            if rep == "baseline":
                regime, rep, model = "baseline", "baseline", parts[4]
            else:
                regime = "A" if kind == "old_powder_A" else "B"
                if receiver not in ("guesthouse", "law73"):
                    return []
            base = {"group": "powder", "target": f"powder/{receiver}/20261025T1355Z/{band}", "band": band, "regime": regime, "rep": rep,
                    "model": model, "seed": d.get("seed", 0), "trained_horizon": None}
        else:
            stage = parts[1]
            if stage == "baseline":
                platform, site, run, band, name = parts[2:7]
                base = {"group": platform, "target": f"{platform}/{site}/{run}/{band}", "band": band, "regime": "baseline", "rep": "baseline",
                        "model": name, "seed": 0, "trained_horizon": None}
            else:
                platform, site, run, band, rep, model = parts[2:8]
                base = {"group": platform, "target": f"{platform}/{site}/{run}/{band}", "band": band, "regime": stage, "rep": rep,
                        "model": model, "seed": d.get("seed", 0), "trained_horizon": None}
        return [{**base, "horizon": int(h), "mae": m["mae_db"]} for h, m in d["metrics"].items()]

    with ThreadPoolExecutor(48) as pool:
        out = [row for rows in pool.map(read, jobs) for row in rows]
    return pd.DataFrame(out)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Seed-average per target, then mean over targets of a group; mean_4h only where all four horizons exist."""
    raw = raw.copy()
    raw["model"] = raw.apply(lambda r: f"{r['model']} ({r['rep']})" if r["rep"] in ("1d", "2d") and r["model"] in
                             ("linearar2d", "residuallinearar2d", "residualvanillalstm", "vanillalstm", "linearar1d", "residuallinearar1d")
                             and False else r["model"], axis=1)
    per_target = raw.groupby(["group", "band", "target", "regime", "rep", "model", "horizon"]).mae.mean().reset_index()
    # the October POWDER group is the held-out receivers only (guesthouse, law73): the baselines also exist for the four core receivers, which no model is
    # transferred to, so they must not count towards the expected number of targets
    per_target = per_target[~((per_target.group == "powder") & ~per_target.target.str.contains("guesthouse|law73"))]
    group_mean = per_target.groupby(["group", "band", "regime", "rep", "model", "horizon"]).agg(mae=("mae", "mean"), n_targets=("target", "nunique")).reset_index()
    expected = (per_target[per_target.regime == "baseline"].groupby(["group", "band"]).target.nunique().rename("n_expected").reset_index())
    group_mean = group_mean.merge(expected, on=["group", "band"], how="left")
    t12 = group_mean[group_mean.horizon == 12].rename(columns={"mae": "mae_t12"}).drop(columns="horizon")
    sub = group_mean[group_mean.horizon.isin(H4)]
    full = sub.groupby(["group", "band", "regime", "rep", "model"]).filter(lambda g: g.horizon.nunique() == 4)
    m4 = full.groupby(["group", "band", "regime", "rep", "model"]).mae.mean().rename("mae_mean_4h").reset_index()
    out = t12.merge(m4, on=["group", "band", "regime", "rep", "model"], how="left")
    out["complete"] = out.n_targets == out.n_expected          # every target of the group scored (otherwise the value is not comparable)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="evaluation/results/transfer_all_regimes.csv")
    parser.add_argument("--cache", default=None, help="read/write the long raw table here (re-run the analysis without re-reading the bucket)")
    args = parser.parse_args()
    if args.cache and Path(args.cache).exists():
        raw = pd.read_csv(args.cache)
    else:
        raw = load_raw(spectrum_s3_client())
        if args.cache:
            raw.to_csv(args.cache, index=False)
    table = aggregate(raw)
    table.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    base = table[table.regime == "baseline"]
    for band in sorted(table.band.unique()):
        for metric in ("mae_t12", "mae_mean_4h"):
            t = table[(table.band == band) & (table.regime != "baseline") & table.complete]   # incomplete groups are in the CSV, not printed
            wide = t.pivot_table(index=["model", "rep", "regime"], columns="group", values=metric)
            b = base[base.band == band].pivot_table(index="model", columns="group", values=metric)
            print(f"\n{band}: {metric} (dB), mean over the group's targets, seeds averaged; baselines:")
            print(b.round(3).to_string())
            print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
