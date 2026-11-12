#!/usr/bin/env python3
"""Tables for the periodicity-crossed frozen-input frequency ablation (``training/ray/frequency_ablation_classes.py``).

Reads ``<prefix>/<model>/<band>/seed-N[/h<h>]/per_origin.npz`` (default prefix ``frequency-ablation-2026-11-classes``) and writes, with the suffix given:

  frequency_ablation_classes.csv   one row per band, model, seed ('mean' = seed-averaged per-origin differences), receiver ('pooled' = target bins of the four
                                   receivers pooled), target class (P, N, middle, all), disturbed set, disturbance, horizon ('mean' over t+4/12/24/48 or one):
                                   mae_real, mae_disturbed, diff (disturbed - real, dB; > 0 means the disturbed bins were informative), a moving-block bootstrap
                                   95% interval over origins (block 96, 2,000 draws), diff_nowrap (origins without wrapped rows only), n_target_bins, n_origins
  frequency_ablation_classes_checks.csv   controls (must be exactly zero) and the reproduction of the existing ablation per region
  frequency_ablation_classes_report.txt   short text summary by the three explanations (A time-of-day cue, B current activity, C artifact)

Target class = class of the target bin (bin_periodicity_classes.csv); the target bins are the bins of the non-noise plan regions; each region is scored in its own
forward pass in which the other bins of the disturbed set are disturbed and the region's own bins are real.

    python evaluation/analysis/frequency_ablation_classes_table.py [--prefix ...] [--suffix _seed42] [--seeds 42]
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
HERE = Path(__file__).resolve().parent
RECEIVERS = ("humanities", "madsen", "moran", "sagepoint")
CLASS_SETS = ("all", "P_only", "N_only", "noise_only")
DISTURBANCES = ("shift96", "shift192", "shift144", "shift210", "level")
CONTROLS = ("revinmlp", "patchtst", "linearar2d", "residuallinearar2d", "nlinear", "dlinear", "temporalconvnet")
DIRECT = ("revinmlp", "revinmlp_mix", "patchtst", "patchtst_joint", "itransformer", "nlinear", "dlinear")
MAIN_MIN_START, NOWRAP_MIN_START = 96, 210
BLOCK, DRAWS = 96, 2000
RNG = np.random.default_rng(0)
TARGET_CLASSES = (("P", ("P",)), ("N", ("N",)), ("middle", ("M",)), ("all", ("P", "N", "M")))


def get(client, key: str) -> bytes:
    return client.get_object(Bucket="spectrum", Key=key)["Body"].read()


def target_bin_counts(classes: pd.DataFrame, band: str) -> dict[str, dict[str, int]]:
    """receiver -> {'P': n, 'N': n, 'M': n}: target bins = bins of the non-noise plan regions."""
    part = classes[(classes["band"] == band) & (~classes["is_noise_floor"].astype(bool)) & (classes["region"].astype(str) != "nan") & (classes["region"].notna())]
    out = {}
    for receiver, rows in part.groupby("receiver"):
        out[receiver] = {"P": int((rows["class"] == "P").sum()), "N": int((rows["class"] == "N").sum()), "M": int((rows["class"] == "middle").sum())}
    return out


def list_runs(client, prefix: str):
    """(model, band, seed) -> {horizon or None: key}."""
    runs: dict[tuple, dict] = defaultdict(dict)
    for page in client.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            m = re.match(rf"{re.escape(prefix.rstrip('/'))}/([^/]+)/([^/]+)/seed-(\d+)(?:/h(\d+))?/per_origin\.npz$", key)
            if m and m.group(1) != "_probe":
                runs[(m.group(1), m.group(2), int(m.group(3)))][int(m.group(4)) if m.group(4) else None] = key
    return runs


def load_run(client, keys: dict) -> dict:
    """Merge the per-horizon checkpoints of a direct model into one object with arrays (origins, horizons)."""
    parts = []
    for h in sorted(keys, key=lambda x: -1 if x is None else x):
        z = np.load(io.BytesIO(get(client, keys[h])), allow_pickle=True)
        parts.append({k: z[k] for k in z.files})
    out = dict(parts[0])
    if len(parts) > 1:
        out["horizons"] = np.concatenate([p["horizons"] for p in parts])
        for k in parts[0]:
            if k.startswith(("cls|", "rmae|", "mae|")):
                out[k] = np.concatenate([p[k] for p in parts], axis=1)
            elif k.startswith(("bin|", "binnw|", "binmae_main|", "binmae_all|")):
                out[k] = np.concatenate([p[k] for p in parts], axis=0)
    return out


def boot(delta: np.ndarray) -> tuple[float, float]:
    n = len(delta)
    block = BLOCK if n >= 2 * BLOCK else max(1, n // 4)
    nb = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(DRAWS, nb))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(DRAWS, -1)[:, :n]
    draws = delta[index].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="frequency-ablation-2026-11-classes")
    ap.add_argument("--old-prefix", default="frequency-ablation-2026-11")
    ap.add_argument("--out", default="evaluation/results")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    classes = pd.read_csv(io.BytesIO(get(client, "analysis-inputs/bin_periodicity_classes.csv")))
    runs = list_runs(client, args.prefix)
    by_model_band: dict[tuple, dict] = defaultdict(dict)
    for (model, band, seed), keys in runs.items():
        if (args.models and model not in args.models) or (args.seeds and seed not in args.seeds):
            continue
        by_model_band[(model, band)][seed] = keys
    old_runs = list_runs(client, args.old_prefix)
    rows, checks = [], []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for (model, band), by_seed in sorted(by_model_band.items()):
        counts = target_bin_counts(classes, band)
        data = {s: load_run(client, k) for s, k in by_seed.items()}
        seeds = sorted(data)
        first = data[seeds[0]]
        horizons = [int(h) for h in first["horizons"]]
        conds = [str(c) for c in first["conditions"]]
        receivers = [str(r) for r in first["receivers"]]
        # per-seed per-origin sums, aligned by origin time across receivers and seeds
        for cond in conds:
            if cond == "real":
                continue
            cset = next(s for s in CLASS_SETS if cond.startswith(s + "_"))
            dist = cond[len(cset) + 1:]
            for tname, shorts in TARGET_CLASSES:
                # per (seed, receiver): arrays (origins, horizons) of per-origin MEAN over the class's target bins (delta = disturbed - real)
                series: dict[tuple, dict] = {}
                for s in seeds:
                    for r in receivers:
                        n_bins = sum(counts[r][k] for k in shorts)
                        if n_bins == 0:
                            continue
                        real = sum(data[s][f"cls|{r}|real|{k}"] for k in shorts) / n_bins
                        dis = sum(data[s][f"cls|{r}|{cond}|{k}"] for k in shorts) / n_bins
                        series[(s, r)] = {"real": real.astype(np.float64), "dis": dis.astype(np.float64), "starts": data[s][f"starts|{r}"], "n": n_bins}
                if not series:
                    continue

                def emit(seed_label, receiver_label, members):
                    """members: list of (seed, receiver) keys averaged (equal weight); pooled receivers weight by bin count."""
                    ws = np.array([series[m]["n"] if receiver_label == "pooled" else 1.0 for m in members], dtype=float)
                    ws = ws / ws.sum()
                    starts = series[members[0]]["starts"]
                    if any(len(series[m]["starts"]) != len(starts) for m in members):
                        return
                    real = sum(w * series[m]["real"] for w, m in zip(ws, members))
                    dis = sum(w * series[m]["dis"] for w, m in zip(ws, members))
                    main = starts >= MAIN_MIN_START
                    nowrap = starts >= NOWRAP_MIN_START
                    n_bins = int(sum(series[m]["n"] for m in members if m[0] == members[0][0]))
                    for hlabel, js in (("mean", list(range(len(horizons)))), *[(h, [j]) for j, h in enumerate(horizons)]):
                        r_h, d_h = real[:, js].mean(axis=1), dis[:, js].mean(axis=1)
                        diff = d_h[main] - r_h[main]
                        lo, hi = (np.nan, np.nan) if args.no_bootstrap else boot(diff)
                        rows.append({"band": band, "model": model, "seed": seed_label, "receiver": receiver_label, "target_class": tname,
                                     "disturbed_set": cset, "disturbance": dist, "horizon": hlabel, "mae_real": float(r_h[main].mean()),
                                     "mae_disturbed": float(d_h[main].mean()), "diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
                                     "diff_nowrap": float((d_h[nowrap] - r_h[nowrap]).mean()) if nowrap.any() else np.nan,
                                     "n_target_bins": n_bins, "n_origins": int(main.sum()), "n_seeds": len(set(m[0] for m in members))})

                recvs = sorted({r for _, r in series})
                for s in seeds:                                         # per seed: each receiver and pooled
                    for r in recvs:
                        if (s, r) in series:
                            emit(s, r, [(s, r)])
                    emit(s, "pooled", [(s, r) for r in recvs if (s, r) in series])
                if len(seeds) > 1:                                      # seed-averaged
                    for r in recvs:
                        emit("mean", r, [(s, r) for s in seeds if (s, r) in series])
                    emit("mean", "pooled", [(s, r) for s in seeds for r in recvs if (s, r) in series])
        # checks: controls exactly zero; reproduction of the existing per-region ablation
        for s in seeds:
            d = data[s]
            worst = 0.0
            for cond in conds:
                if cond == "real":
                    continue
                for r in receivers:
                    for k in ("P", "N", "M"):
                        worst = max(worst, float(np.abs(d[f"cls|{r}|{cond}|{k}"] - d[f"cls|{r}|real|{k}"]).max()))
            checks.append({"band": band, "model": model, "seed": s, "check": "max |disturbed - real| over all conditions (class sums, dB x bins)",
                           "value": worst, "control": model in CONTROLS})
            old_keys = old_runs.get((model, band, s))
            if old_keys:
                od = load_run(client, old_keys)
                for new_cond, old_mode in (("real", "real"), ("all_shift96", "shift96"), ("all_shift210", "shift210")):
                    worst_region = 0.0
                    for r in receivers:
                        for region in [str(x) for x in d["regions"]]:
                            a, b = d.get(f"rmae|{r}|{region}|{new_cond}"), od.get(f"mae|{r}|{region}|{old_mode}")
                            if a is not None and b is not None and a.shape == b.shape:
                                worst_region = max(worst_region, float(np.abs(a - b).max()))
                    checks.append({"band": band, "model": model, "seed": s, "check": f"max |region MAE per origin, {new_cond} - existing ablation {old_mode}|",
                                   "value": worst_region, "control": model in CONTROLS})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(out / f"frequency_ablation_classes{args.suffix}.csv", index=False)
    pd.DataFrame(checks).to_csv(out / f"frequency_ablation_classes_checks{args.suffix}.csv", index=False)
    print(f"{len(cdf)} rows, {len(checks)} checks")


if __name__ == "__main__":
    main()
