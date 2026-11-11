"""Zero-shot MOMENT (masked reconstruction, ``training.ray.moment_zeroshot``): design choice on validation, then the test-week tables.

    python evaluation/analysis/moment_zeroshot_table.py choose     # rank the candidate designs on the VALIDATION range, write the frozen design JSON
    python evaluation/analysis/moment_zeroshot_table.py table      # the October test week: all five metrics next to the baselines and learned models

``choose`` reads ``moment-zeroshot-oct2026/validation/<band>/<receiver>.json`` (one record per band and core receiver with every candidate design scored on the
2-day validation range) and picks the design with the lowest MAE averaged over horizons 1/4/12/24/48, the four core receivers and both bands (equal weight per
cell); it writes ``evaluation/analysis/moment_zeroshot_design.json`` (the chosen design, the rule, and every candidate's validation numbers) and must be committed
before the test week is scored.

``table`` reads the test outputs (``moment-zeroshot-oct2026/A/powder/<receiver>/<run>/<band>/2d/moment_zeroshot_<size>[_ctx<n>]/seed-0/{summary.json,per_origin.csv}``),
pools the four core receivers per band (origin-weighted mean of the per-receiver aggregates; every receiver has the same 625 origins), writes the long CSV
``evaluation/results/moment_zeroshot_15min.csv`` (band, family, model, horizon, metric, mean, sd, n: the format of ``part1_per_horizon_15min.csv``; deterministic, so
sd = 0, n = 1), prints MAE and MSE per horizon next to ``part1_per_horizon_15min.csv`` rows and ``baselines_all_horizons.csv``, a paired block-bootstrap interval against
the tuned median on identical origins (``paired_intervals_part1`` helpers), and the held-out receivers and external sites where they were run.
Run it on the cluster (``training/ray/scripts/run_moment_table.sh``); it fetches its small inputs from ``s3://spectrum/analysis-inputs/`` when the files are missing.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

BUCKET, PREFIX = "spectrum", "moment-zeroshot-oct2026"
DESIGN = "evaluation/analysis/moment_zeroshot_design.json"
HORIZONS = (1, 4, 12, 24, 48)
CORE = ("humanities", "madsen", "moran", "sagepoint")
BANDS = ("600_800", "2400_2600")
METRICS = ("mae_db", "mse_db2", "rmse_db", "mape_linear_pct", "mape_dbm_pct")
SHORT = {"mae_db": "mae", "mse_db2": "mse", "rmse_db": "rmse", "mape_linear_pct": "mape_linear", "mape_dbm_pct": "mape_dbm"}


def client():
    from training.ray.storage import spectrum_s3_client

    return spectrum_s3_client()


def listing(c, prefix: str) -> list[str]:
    return [o["Key"] for pg in c.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix) for o in pg.get("Contents", [])]


def get_json(c, key: str):
    return json.loads(c.get_object(Bucket=BUCKET, Key=key)["Body"].read())


def ensure_input(c, path: str) -> str:
    """The cluster's synced code lacks evaluation/results: fetch a small input from the bucket when it is missing."""
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(c.get_object(Bucket=BUCKET, Key="analysis-inputs/" + p.name)["Body"].read())
    return path


# ------------------------------------------------------------------------------------------------ choose
def choose(args) -> None:
    c = client()
    rows = []
    for key in sorted(k for k in listing(c, f"{PREFIX}/validation/") if k.endswith(".json")):
        rec = get_json(c, key)
        if "designs" not in rec:                      # (the runner writes the record before it sets "ok", so test for the content)
            print("skipping failed validation record", key, rec.get("error"))
            continue
        for name, d in rec["designs"].items():
            maes = [d["metrics"][str(h)]["mae_db"] for h in HORIZONS]
            rows.append({"band": rec["target"]["band"], "receiver": rec["target"]["site"], "design": name, "n_origins": d["n_origins"],
                         "mae_mean": float(np.mean(maes)), **{f"mae_t{h}": m for h, m in zip(HORIZONS, maes)},
                         "last_value_mae_mean": float(np.mean([d["last_value_mae"][str(h)] for h in HORIZONS])), "seconds": d["forecast_seconds"]})
    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"{frame[['band', 'receiver']].drop_duplicates().shape[0]} validation cells")
    pooled = frame.groupby("design").agg(mae_mean=("mae_mean", "mean"), last_value=("last_value_mae_mean", "mean"), seconds=("seconds", "sum"),
                                         **{f"t{h}": (f"mae_t{h}", "mean") for h in HORIZONS}).sort_values("mae_mean")
    print("\nvalidation MAE (dB), mean over the cells, by design (last_value = persistence on the same origins):")
    print(pooled.round(3).to_string())
    print("\nby band:")
    print(frame.groupby(["band", "design"]).mae_mean.mean().unstack().round(3).to_string())
    from training.ray.moment_zeroshot import CANDIDATES

    best = pooled.index[0]
    design = next(d for d in CANDIDATES if d["name"] == best)
    out = {"model": "MOMENT-1 (frozen, pretrained reconstruction head), forecasting as masked reconstruction",
           "rule": "lowest MAE on the validation range 2026-10-25 14:00 to 10-27 13:55 UTC, mean over horizons 1/4/12/24/48, the four core receivers and both bands (equal weight per cell)",
           "design": design, "validation": {"pooled": json.loads(pooled.round(5).to_json(orient="index")), "n_cells": int(frame[['band', 'receiver']].drop_duplicates().shape[0]),
                                           "origins_per_cell": int(frame.n_origins.iloc[0])},
           "frozen_before_test_scoring": True}
    Path(args.design).write_text(json.dumps(out, indent=1))
    print(f"\nchosen design: {design}  -> {args.design}")


# ------------------------------------------------------------------------------------------------ table
def load_test(c, label: str, prefix: str = PREFIX):
    """{(band, receiver): (summary, per_origin DataFrame)} for the core receivers, plus {(group, target): summary} for everything else."""
    core, other = {}, {}
    for key in sorted(k for k in listing(c, f"{prefix}/A/") if k.endswith("/summary.json") and f"/{label}/" in k):
        s = get_json(c, key)
        po = pd.read_csv(io.BytesIO(c.get_object(Bucket=BUCKET, Key=key.replace("summary.json", "per_origin.csv"))["Body"].read()))
        if s["platform"] == "powder" and s["site"] in CORE:
            core[(s["target_band"], s["site"])] = (s, po)
        else:
            other[(s["platform"], s["target"])] = (s, po)
    return core, other


def pooled_metrics(core: dict, band: str) -> dict[int, dict[str, float]]:
    cells = [core[(band, r)][0] for r in CORE if (band, r) in core]
    if len(cells) != len(CORE):
        return {}
    weights = np.array([float(s["n_origins"]) for s in cells])
    return {h: {m: float(np.average([s["metrics"][str(h)][m] for s in cells], weights=weights)) for m in METRICS} for h in HORIZONS}


def table(args) -> None:
    c = client()
    from training.ray.moment_zeroshot import PREFIX as _P  # noqa: F401  (the prefix constant of the runner)

    label = args.label
    core, other = load_test(c, label, args.prefix)
    rows = []
    for band in BANDS:
        pooled = pooled_metrics(core, band)
        for h, d in pooled.items():
            for m, v in d.items():
                rows.append({"band": band, "family": "zero-shot", "model": label.replace("_", " "), "horizon": h, "metric": SHORT[m], "mean": v, "sd": 0.0, "n": 1})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    p1 = pd.read_csv(ensure_input(c, args.part1))
    base = pd.read_csv(ensure_input(c, "evaluation/results/baselines_all_horizons.csv"))
    keep = [("heuristic", "median (tuned)"), ("heuristic", "mean (tuned)"), ("heuristic (untuned)", "median (full lookback, 96)"), ("heuristic (untuned)", "mean (full lookback, 96)"),
            ("heuristic", "persistence"), ("heuristic", "kalman"), ("direct", "revinmlp"), ("direct", "ofa"), ("direct", "patchtst"), ("direct", "timeran_lp"),
            ("direct", "moment_lp"), ("direct", "residuallinearar2d"), ("rolled-out", "residuallinearar2d (2d)")]
    for band in BANDS:
        zs = out[out.band == band]
        if zs.empty:
            print(f"\n{band}: no complete set of the four core receivers yet")
            continue
        for metric in ("mae", "mse", "rmse"):
            tab = {}
            for fam, model in keep:
                d = p1[(p1.band == band) & (p1.family == fam) & (p1.model == model) & (p1.metric == metric)]
                if len(d):
                    tab[f"{model}"] = d.set_index("horizon")["mean"]
            tab[f"ZERO-SHOT {label}"] = zs[zs.metric == metric].set_index("horizon")["mean"]
            t = pd.DataFrame(tab).T
            t.columns = [int(x) for x in t.columns]
            t = t[[h for h in HORIZONS if h in t.columns]]
            t["mean_4h"] = t[[h for h in (4, 12, 24, 48) if h in t.columns]].mean(axis=1)
            print(f"\n{band}: 15-minute test {metric.upper()} per horizon (steps of 15 min), four core receivers pooled")
            print(t.round(3).sort_values("mean_4h").to_string())
    # paired intervals against the tuned median on identical origins
    import paired_intervals_part1 as P

    print("\npaired block bootstrap against the tuned median on identical origins (MAE, dB; delta < 0 means zero-shot MOMENT is better; pooled over the four receivers):")
    horizons = P.HORIZONS["15min"]
    for band in BANDS:
        draws_all, deltas, rows_p = [], [], []
        for r in CORE:
            if (band, r) not in core:
                continue
            series, _ = P.load_receiver(c, "15min", band, r)
            median = series.get("tuned_median")
            po = core[(band, r)][1]
            zs = P.per_horizon_series(po, horizons)
            if median is None or any(h not in zs for h in horizons):
                continue
            frame, _ = P.paired_values(zs, median, horizons)
            delta = (frame.a - frame.b).to_numpy()
            deltas.append(delta.mean())
            draws_all.append(P.block_bootstrap_means(delta, P.BLOCK["15min"]))
            rows_p.append((r, len(delta), frame.a.mean(), frame.b.mean()))
        if draws_all:
            draws = np.mean(np.stack(draws_all), axis=0)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            print(f"  {band}: zero-shot MOMENT {np.mean([x[2] for x in rows_p]):.3f} vs tuned median {np.mean([x[3] for x in rows_p]):.3f}; delta {np.mean(deltas):+.3f} dB [{lo:+.3f}, {hi:+.3f}] "
                  f"({sum(x[1] for x in rows_p)} origins, {len(rows_p)} receivers)")
    # held-out receivers and external sites
    if other:
        print("\nheld-out receivers and external sites (zero-shot MOMENT; MAE at t+12 and mean over t+4/12/24/48, mean over the group's targets):")
        by = {}
        for (platform, target), (s, _) in other.items():
            by.setdefault((platform, s["target_band"]), []).append(s)
        for (platform, band), ss in sorted(by.items()):
            m12 = np.mean([s["metrics"]["12"]["mae_db"] for s in ss])
            m4 = np.mean([np.mean([s["metrics"][str(h)]["mae_db"] for h in (4, 12, 24, 48)]) for s in ss])
            print(f"  {platform:7s} {band}: {len(ss):2d} targets  t+12 {m12:.3f}  mean_4h {m4:.3f}")
    print(f"\nwrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("choose", "table"))
    p.add_argument("--design", default=DESIGN)
    p.add_argument("--label", default="moment_zeroshot_small")
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--part1", default="evaluation/results/part1_per_horizon_15min.csv")
    p.add_argument("--out", default="evaluation/results/moment_zeroshot_15min.csv")
    args = p.parse_args()
    choose(args) if args.command == "choose" else table(args)


if __name__ == "__main__":
    main()
