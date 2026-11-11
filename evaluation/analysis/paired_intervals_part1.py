"""Paired blocked-bootstrap intervals for the Part 1 "ties" claims: learned models against the heuristic baselines, per forecast origin.

Inputs (written by ``training.ray.origin_errors`` unless noted), per core receiver, band and resolution:
  * learned models: ``origin-errors-oct2026/<res>/<band>/<receiver>/direct/<model>/h<h>/seed-N/per_origin.csv`` (direct finals, one checkpoint
    per trained horizon) and, at 1 minute, ``.../<oct|band>/<model>/roll/seed-N/per_origin.csv`` (rolled-out checkpoints); at 15 minutes the
    rolled-out tuned 1D/2D models come from ``transfer15-oct2026/<band>/<receiver>/<rep>/<model>/seed-N/per_origin.csv`` (``transfer15``)
  * baselines: ``origin-errors-oct2026/<res>/<band>/<receiver>/baseline/per_origin.csv`` (persistence, lookback mean, validation-tuned
    median and mean, per-bin Kalman filter).
Every per-origin value is the mean absolute error over the 200 bins (dB) at one forecast origin and horizon. Seeds (40-44) are averaged per
origin; the comparison uses only origins present for every model, seed, horizon and baseline in the comparison, matched on the origin
timestamp. The difference ``model - baseline`` is bootstrapped with a moving block bootstrap over the origin series (block = one day of
origins: 96 at 15 minutes, 1,440 at 1 minute; 2,000 resamples; ``neighbor_value.py`` pattern). A pooled value over the four receivers
averages the per-receiver differences, with the receivers resampled independently. delta < 0: the model is better.
Metrics: the mean over the horizons of the resolution (15 min: t+1h/3h/6h/12h = steps 4/12/24/48; 1 min: 1/15/60 minutes) and the single
horizon t+3h (15 min, step 12) or t+15 min (1 min).

    python evaluation/analysis/paired_intervals_part1.py [--out evaluation/results/paired_intervals_part1.csv]
"""
from __future__ import annotations

import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

BUCKET = "spectrum"
PREFIX = "origin-errors-oct2026"
CORE = ("humanities", "madsen", "moran", "sagepoint")
BANDS = ("600_800", "2400_2600")
SEEDS = (40, 41, 42, 43, 44)
HORIZONS = {"15min": (4, 12, 24, 48), "1min": (1, 15, 60)}
SINGLE = {"15min": 12, "1min": 15}
BLOCK = {"15min": 96, "1min": 1440}
DRAWS = 2000
BASELINES = ("persistence", "lookbackmean", "tuned_median", "tuned_mean", "kalman")
DIRECT = {"15min": ("revinmlp", "patchtst", "dlinear", "nlinear", "residuallinearar2d"),
          "1min": ("patchtst", "dlinear", "residuallinearar2d")}
ROLLED_15 = (("1d", "residuallinearar1d"), ("2d", "residuallinearar2d"), ("1d", "linearar1d"), ("2d", "linearar2d"))
ROLLED_1 = (("oct", "residuallinearar2d"), ("oct", "linearar2d"))      # 'oct' = 600-800, 'band' = 2400-2600
RNG = np.random.default_rng(0)


def read_csv(client, key: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(client.get_object(Bucket=BUCKET, Key=key)["Body"].read()))
    except Exception:
        return None


def per_horizon_series(frame: pd.DataFrame, horizons) -> dict[int, pd.Series]:
    """horizon -> MAE indexed by origin time (UTC)."""
    out = {}
    for h in horizons:
        g = frame[frame.horizon == h]
        if len(g):
            out[h] = pd.Series(g.mae_db.to_numpy(), index=pd.to_datetime(g.time, utc=True)).sort_index()
    return out


def average_seeds(per_seed: list[dict[int, pd.Series]], horizons) -> dict[int, pd.Series] | None:
    """Seed-averaged series per horizon on the origins present in every seed; None if a horizon is missing."""
    if not per_seed:
        return None
    out = {}
    for h in horizons:
        if any(h not in s for s in per_seed):
            return None
        out[h] = pd.concat([s[h] for s in per_seed], axis=1, join="inner").mean(axis=1)
    return out


def load_receiver(client, res: str, band: str, receiver: str) -> tuple[dict[str, dict[int, pd.Series]], dict[str, int]]:
    horizons = HORIZONS[res]
    base = f"{PREFIX}/{res}/{band}/{receiver}"
    series, n_seeds = {}, {}
    baseline = read_csv(client, f"{base}/baseline/per_origin.csv")
    if baseline is not None:
        for stat in BASELINES:
            series[stat] = per_horizon_series(baseline[baseline.stat == stat], horizons)
            n_seeds[stat] = 1
    for model in DIRECT[res]:
        per_seed = []
        for seed in SEEDS:
            parts = {}
            for h in horizons:
                frame = read_csv(client, f"{base}/direct/{model}/h{h}/seed-{seed}/per_origin.csv")
                if frame is not None:
                    parts.update(per_horizon_series(frame, [h]))
            if len(parts) == len(horizons):
                per_seed.append(parts)
        avg = average_seeds(per_seed, horizons)
        if avg is not None:
            series[f"{model} (direct)"], n_seeds[f"{model} (direct)"] = avg, len(per_seed)
    if res == "15min":
        for rep, model in ROLLED_15:
            per_seed = []
            for seed in SEEDS:
                frame = read_csv(client, f"transfer15-oct2026/{band}/{receiver}/{rep}/{model}/seed-{seed}/per_origin.csv")
                if frame is not None:
                    per_seed.append(per_horizon_series(frame, horizons))
            avg = average_seeds(per_seed, horizons)
            if avg is not None:
                series[f"{model} ({rep}, rolled-out)"], n_seeds[f"{model} ({rep}, rolled-out)"] = avg, len(per_seed)
    else:
        source = "oct" if band == "600_800" else "band"
        for _, model in ROLLED_1:
            per_seed = []
            for seed in SEEDS:
                frame = read_csv(client, f"{base}/{source}/{model}/roll/seed-{seed}/per_origin.csv")
                if frame is not None:
                    per_seed.append(per_horizon_series(frame, horizons))
            avg = average_seeds(per_seed, horizons)
            if avg is not None:
                series[f"{model} (2d, rolled-out)"], n_seeds[f"{model} (2d, rolled-out)"] = avg, len(per_seed)
    return series, n_seeds


def block_bootstrap_means(delta: np.ndarray, block: int) -> np.ndarray:
    """Moving-block bootstrap draws of the mean of a per-origin series (prefix-sum form, truncated to len(delta))."""
    n = len(delta)
    if n < 2 * block:
        block = max(1, n // 4)
    n_blocks = int(np.ceil(n / block))
    remainder = n - (n_blocks - 1) * block
    cs = np.concatenate([[0.0], np.cumsum(delta)])
    starts = RNG.integers(0, n - block + 1, size=(DRAWS, n_blocks))
    full = (cs[starts[:, :-1] + block] - cs[starts[:, :-1]]).sum(axis=1) if n_blocks > 1 else np.zeros(DRAWS)
    last = cs[starts[:, -1] + remainder] - cs[starts[:, -1]]
    return (full + last) / n


def paired_values(a: dict[int, pd.Series], b: dict[int, pd.Series], horizons) -> tuple[pd.DataFrame, list[int]]:
    """Per-origin MAE of a and b (mean over ``horizons``) on the origins both have at every horizon."""
    index = None
    for h in horizons:
        index = a[h].index.intersection(b[h].index) if index is None else index.intersection(a[h].index).intersection(b[h].index)
    index = index.sort_values()
    x = np.mean([a[h].loc[index].to_numpy() for h in horizons], axis=0)
    y = np.mean([b[h].loc[index].to_numpy() for h in horizons], axis=0)
    return pd.DataFrame({"a": x, "b": y}, index=index), list(horizons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="evaluation/results/paired_intervals_part1.csv")
    args = parser.parse_args()
    client = spectrum_s3_client()
    jobs = [(res, band, receiver) for res in ("15min", "1min") for band in BANDS for receiver in CORE]
    with ThreadPoolExecutor(8) as pool:
        loaded = dict(zip(jobs, pool.map(lambda j: load_receiver(client, *j), jobs)))
    rows = []
    for res in ("15min", "1min"):
        for band in BANDS:
            models = sorted({m for r in CORE for m in loaded[(res, band, r)][0] if m not in BASELINES})
            for model in models:
                for versus in BASELINES:
                    for metric, horizons in (("mean_over_horizons", HORIZONS[res]), (f"t+{SINGLE[res]}", (SINGLE[res],))):
                        per_receiver, draws = [], []
                        for receiver in CORE:
                            series, _ = loaded[(res, band, receiver)]
                            if model not in series or versus not in series:
                                continue
                            pair, _ = paired_values(series[model], series[versus], horizons)
                            if not len(pair):
                                continue
                            d = pair.a.to_numpy() - pair.b.to_numpy()
                            boot = block_bootstrap_means(d, BLOCK[res])
                            seeds = loaded[(res, band, receiver)][1][model]
                            rows.append({"res": res, "band": band, "model": model, "versus": versus, "metric": metric, "receiver": receiver,
                                         "n_origins": len(pair), "n_seeds": seeds, "mae_model": pair.a.mean(), "mae_baseline": pair.b.mean(),
                                         "delta": d.mean(), "delta_pct": 100 * d.mean() / pair.b.mean(),
                                         "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
                            per_receiver.append(rows[-1])
                            draws.append(boot)
                        if len(per_receiver) == len(CORE):
                            pooled = np.mean(np.stack(draws), axis=0)
                            base = np.mean([r["mae_baseline"] for r in per_receiver])
                            delta = np.mean([r["delta"] for r in per_receiver])
                            rows.append({"res": res, "band": band, "model": model, "versus": versus, "metric": metric, "receiver": "pooled",
                                         "n_origins": int(sum(r["n_origins"] for r in per_receiver)), "n_seeds": per_receiver[0]["n_seeds"],
                                         "mae_model": np.mean([r["mae_model"] for r in per_receiver]), "mae_baseline": base, "delta": delta,
                                         "delta_pct": 100 * delta / base, "ci_low": np.percentile(pooled, 2.5), "ci_high": np.percentile(pooled, 97.5)})
    frame = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    pooled = frame[frame.receiver == "pooled"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["model", "versus", "mae_model", "mae_baseline", "delta", "delta_pct", "ci_low", "ci_high", "n_origins", "n_seeds"]
    for (res, band, metric), g in pooled.groupby(["res", "band", "metric"]):
        print(f"\n{res} {band} {metric}: pooled over four core receivers; delta = model - baseline (dB; < 0 model better)")
        print(g[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
