"""Frozen-input frequency ablation: tables from ``training.ray.frequency_ablation`` (S3 prefix ``frequency-ablation-2026-11``).

For every (model, band, non-noise region): the region's mean absolute error (dB, over the region's bins; mean over the four core receivers and over
the available seeds, per forecast origin) when the model sees the real inputs and when the inputs of all OTHER bins are replaced by the same bins shifted
in time by 24 h (shift96) or 52.5 h (shift210), per horizon (t+4/12/24/48 and their mean). delta = shifted - real (> 0: breaking the other bins hurts, i.e.
the model uses them). Nothing is retrained; the 24 h shift is the control for the fact that a frozen model fed inconsistent inputs can do worse for reasons
other than lost information. Summary: per (model, band, mode, horizon) the mean over regions of delta with a moving-block bootstrap interval over
origins (block 96 = one day, 2,000 draws, regions resampled independently and averaged, as ``neighbor_value.py`` does for receivers), the number of regions
with delta > 0 and < 0, and the same delta without the first 210 origins (whose shifted windows contain wrapped rows).

    python evaluation/analysis/frequency_ablation_table.py [--out evaluation/results] [--prefix frequency-ablation-2026-11]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

BANDS = ("600_800", "2400_2600")
MODES = ("shift96", "shift210")
HORIZONS = (4, 12, 24, 48)
BLOCK, DRAWS, WRAP = 96, 2000, 210
LO_MAIN = 96          # origins whose whole lookback lies inside the test week (the origin set of the stored per-bin tables)
RNG = np.random.default_rng(0)
CONTROLS = ("revinmlp", "linearar2d", "residuallinearar2d")
PLANS = {"600_800": "evaluation/analysis/plan_regions_600_800.csv", "2400_2600": "evaluation/analysis/plan_regions_2400_2600.csv"}


def get(client, key):
    return client.get_object(Bucket="spectrum", Key=key)["Body"].read()


def list_runs(client, prefix):
    """{(model, band): {seed: [per_origin keys]}} (rolled models: one key per seed; direct: one per trained horizon)."""
    runs = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix=prefix.rstrip("/") + "/"):
        for o in page.get("Contents", []):
            parts = o["Key"][len(prefix.rstrip("/")) + 1:].split("/")
            if parts[-1] != "per_origin.npz":
                continue
            model, band, seed = parts[0], parts[1], int(parts[2].split("-")[1])
            runs.setdefault((model, band), {}).setdefault(seed, []).append(o["Key"])
    return runs


def load_seed(client, keys):
    """Merge the per-origin arrays of one (model, band, seed): {receiver|region|mode -> (N, H)}, horizons, receivers, regions, binmae {h: (bins,) per receiver}."""
    merged, horizons, binmae = {}, [], {}
    receivers = regions = times = None
    for key in sorted(keys):
        z = np.load(io.BytesIO(get(client, key)), allow_pickle=True)
        hs = [int(h) for h in z["horizons"]]
        receivers, regions = [str(r) for r in z["receivers"]], [str(r) for r in z["regions"]]
        for receiver in receivers:
            t = z[f"times|{receiver}"]
            times = t if times is None else times
            if not np.array_equal(times, t):
                raise ValueError("origins differ between receivers or horizons")
            for index, h in enumerate(hs):
                binmae[(receiver, h)] = z[f"binmae|{receiver}"][index]
        for receiver in receivers:
            for region in regions:
                for mode in ("real", *MODES):
                    arr = z[f"mae|{receiver}|{region}|{mode}"]
                    for index, h in enumerate(hs):
                        merged.setdefault((receiver, region, mode, h), arr[:, index])
        horizons += hs
    return {"mae": merged, "horizons": sorted(set(horizons)), "receivers": receivers, "regions": regions, "binmae": binmae, "times": times}


def block_bootstrap(delta):
    n = len(delta)
    block = BLOCK if n >= 2 * BLOCK else max(1, n // 4)
    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(DRAWS, n_blocks))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(DRAWS, -1)[:, :n]
    return delta[index].mean(axis=1)


def identity_check(client, model, band, seed, data):
    """Largest and mean absolute difference between the recomputed real-mode per-bin errors (mean over receivers) and the stored rescored per-bin file."""
    diffs = []
    direct = model in ("revinmlp", "revinmlp_mix", "patchtst", "patchtst_joint", "itransformer", "nlinear", "dlinear")
    for h in data["horizons"]:
        if direct and h != 48:                     # a direct final is scored on the origins valid for ITS horizon in the stored table; only t+48 shares our origin set
            continue
        keys = ([f"rescored/per-bin/direct/15min/{band}/2d/{model}/h{h}/seed-{seed}.csv"] if model in ("revinmlp", "revinmlp_mix", "patchtst", "patchtst_joint", "itransformer", "nlinear", "dlinear")
                else [f"rescored/per-bin/p15tuned/{band}/2d/{model}/seed-{seed}.csv"])
        try:
            stored = pd.read_csv(io.BytesIO(get(client, keys[0])))
        except Exception:
            return None
        stored = stored[stored.horizon == h].sort_values("frequency_mhz")
        mine = np.mean([data["binmae"][(r, h)] for r in data["receivers"]], axis=0)
        if len(stored) != len(mine):
            return None
        diffs.append(np.abs(stored.mae_db.to_numpy() - mine))
    if not diffs:
        return None
    d = np.concatenate(diffs)
    return float(d.max()), float(d.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="evaluation/results")
    ap.add_argument("--prefix", default="frequency-ablation-2026-11")
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plans = {b: pd.read_csv(p) for b, p in PLANS.items()}
    runs = list_runs(client, args.prefix)
    rows, summary, checks = [], [], []
    for (model, band), by_seed in sorted(runs.items()):
        if args.models and model not in args.models:
            continue
        seeds = sorted(by_seed)
        loaded = {s: load_seed(client, by_seed[s]) for s in seeds}
        first = loaded[seeds[0]]
        regions, receivers = first["regions"], first["receivers"]
        horizons = first["horizons"]
        plan = plans[band].set_index("band_id")
        for s in seeds:
            if model.startswith("tsslcd"):             # no stored per-bin file: the eval-only run records the largest difference to its source run's stored scores
                try:
                    diff = json.loads(get(client, f"{args.prefix.rstrip('/')}/{model}/{band}/seed-{s}/summary.json")).get("zero_shift_max_abs_diff_db")
                except Exception:
                    diff = None
                checks.append({"model": model, "band": band, "seed": s, "identity_max_abs_diff_db": diff, "identity_mean_abs_diff_db": None, "control": False})
                continue
            check = identity_check(client, model, band, s, loaded[s])
            checks.append({"model": model, "band": band, "seed": s, "identity_max_abs_diff_db": None if check is None else check[0],
                           "identity_mean_abs_diff_db": None if check is None else check[1], "control": model in CONTROLS})
        # per-origin series per region, mode, horizon: mean over receivers and seeds
        series = {}
        for region in regions:
            for mode in ("real", *MODES):
                for h in horizons:
                    series[(region, mode, h)] = np.mean([loaded[s]["mae"][(r, region, mode, h)] for s in seeds for r in receivers], axis=0)
        n_all = len(series[(regions[0], "real", horizons[0])])
        lo = 0 if model.startswith("tsslcd") else LO_MAIN          # TSS-LCD windows are scored as in the paper (all windows of its padded test partition)
        series = {k: v[lo:] for k, v in series.items()}
        n = n_all - lo
        for region in regions:
            for mode in MODES:
                for label in (*horizons, "mean"):
                    hs = horizons if label == "mean" else [label]
                    real = np.mean([series[(region, "real", h)] for h in hs], axis=0)
                    shifted = np.mean([series[(region, mode, h)] for h in hs], axis=0)
                    delta = shifted - real
                    rows.append({"model": model, "band": band, "region": region, "start_mhz": plan.loc[region, "start_mhz"], "end_mhz": plan.loc[region, "end_mhz"],
                                 "n_bins": int(plan.loc[region, "bin_count"]), "horizon": label, "mode": mode, "mae_real": float(real.mean()), "mae_shifted": float(shifted.mean()),
                                 "delta": float(delta.mean()), "delta_no_wrap": float(delta[WRAP - lo:].mean()), "n_origins": n, "n_seeds": len(seeds), "seeds": " ".join(map(str, seeds)),
                                 "n_receivers": len(receivers)})
        frame = pd.DataFrame([r for r in rows if r["model"] == model and r["band"] == band])
        for mode in MODES:
            for label in (*horizons, "mean"):
                hs = horizons if label == "mean" else [label]
                draws, deltas, dnw, up_ci, down_ci = [], [], [], 0, 0
                for region in regions:
                    d = np.mean([series[(region, mode, h)] - series[(region, "real", h)] for h in hs], axis=0)
                    b = block_bootstrap(d)
                    draws.append(b)
                    deltas.append(float(d.mean()))
                    dnw.append(float(d[WRAP - lo:].mean()))
                    lo, hi = np.percentile(b, [2.5, 97.5])
                    up_ci += lo > 0
                    down_ci += hi < 0
                pooled = np.mean(np.stack(draws), axis=0)
                real_mean = float(np.mean([series[(r, "real", h)].mean() for r in regions for h in hs]))
                summary.append({"model": model, "band": band, "mode": mode, "horizon": label, "n_regions": len(regions), "mae_real": real_mean,
                                "mae_shifted": real_mean + float(np.mean(deltas)), "delta": float(np.mean(deltas)), "ci_low": float(np.percentile(pooled, 2.5)),
                                "ci_high": float(np.percentile(pooled, 97.5)), "delta_no_wrap": float(np.mean(dnw)),
                                "regions_delta_pos": int(sum(x > 0 for x in deltas)), "regions_delta_neg": int(sum(x < 0 for x in deltas)),
                                "regions_ci_above_zero": int(up_ci), "regions_ci_below_zero": int(down_ci), "n_seeds": len(seeds), "seeds": " ".join(map(str, seeds)),
                                "n_origins": n, "control": model in CONTROLS})
    pd.DataFrame(rows).to_csv(out / "frequency_ablation.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(out / "frequency_ablation_summary.csv", index=False)
    c = pd.DataFrame(checks)
    c.to_csv(out / "frequency_ablation_checks.csv", index=False)
    pd.set_option("display.width", 250)
    lines = ["Frozen-input frequency ablation (15 minutes). Nothing is retrained: the other bins' inputs are replaced by the same bins shifted in time and the region's bins are scored.",
             "delta = MAE(shifted) - MAE(real), dB, on the region's bins, mean over the four core receivers and seeds; delta > 0: breaking the other bins hurts (the model uses them).",
             "Caveat: a frozen model fed inconsistent inputs can do worse for reasons other than lost information; the 24 h shift is the control. Rolled-out models: the shift applies to the initial input window only.", ""]
    mean = s[s.horizon == "mean"]
    table = mean.pivot_table(index=["band", "model"], columns="mode", values=["delta", "ci_low", "ci_high", "regions_delta_pos", "regions_delta_neg", "mae_real"], aggfunc="first")
    lines.append("Mean over horizons (t+4/12/24/48), mean over regions:")
    for (band, model), r in table.iterrows():
        lines.append(f"  {band:9s} {model:20s} real {r[('mae_real', 'shift96')]:.3f}  24h {r[('delta', 'shift96')]:+.3f} [{r[('ci_low', 'shift96')]:+.3f}, {r[('ci_high', 'shift96')]:+.3f}] "
                     f"({int(r[('regions_delta_pos', 'shift96')])} up/{int(r[('regions_delta_neg', 'shift96')])} down)  52.5h {r[('delta', 'shift210')]:+.3f} "
                     f"[{r[('ci_low', 'shift210')]:+.3f}, {r[('ci_high', 'shift210')]:+.3f}] ({int(r[('regions_delta_pos', 'shift210')])} up/{int(r[('regions_delta_neg', 'shift210')])} down)")
    lines += ["", "Identity check (recomputed real-mode per-bin errors vs the stored rescored per-bin files, max abs difference in dB):"]
    for r in c.itertuples():
        lines.append(f"  {r.band:9s} {r.model:20s} seed {r.seed}: " + ("n/a" if pd.isna(r.identity_max_abs_diff_db) else f"max {r.identity_max_abs_diff_db:.2e} mean {r.identity_mean_abs_diff_db:.2e}"))
    text = "\n".join(lines)
    (out / "frequency_ablation_report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
