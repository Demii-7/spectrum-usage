#!/usr/bin/env python3
"""Part 3: does cross-bin context pay off, by target class? (real-input errors of the class-crossed frequency ablation; no new runs)

For target bins of class P (daily-periodic: acf_lag96 >= 0.6) and N (acf_lag96 <= 0.4) of the non-noise plan regions: the difference between a cross-bin model and a per-bin counterpart
in the real-input per-origin error (mean over the class's target bins), PAIRED BY ORIGIN, seeds paired. Negative = the cross-bin model is better.
Pairs: RevIN-MLP + mixing vs RevIN-MLP, PatchTST-joint vs PatchTST, and the other cross-bin models (LSTM 2D, residual LSTM 2D, LSTMAttn, Autoformer-CSA, iTransformer) vs RevIN-MLP.
Not possible here: the LSTM 2D / residual LSTM 2D vs their 1D finals (the 1D finals' per-bin errors are stored pooled over receivers and origins, not per receiver and origin, so they
cannot be split by the receiver-specific classes or paired); TSS-LCD is listed UNPAIRED (class MAE only): its context is 100 steps, its origin alignment with the 96-step models is not verified.
Moving-block bootstrap (block 96, 2,000 draws) over origins.

    python evaluation/analysis/frequency_benefit_by_class.py [--prefix frequency-ablation-2026-11-classes] [--suffix ""]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frequency_ablation_classes_table import MAIN_MIN_START, boot, get, list_runs, load_run  # noqa: E402

PAIRS = [("revinmlp_mix", "revinmlp"), ("patchtst_joint", "patchtst"), ("vanillalstm", "revinmlp"), ("residualvanillalstm", "revinmlp"),
         ("lstmattn", "revinmlp"), ("autoformer_csa", "revinmlp"), ("itransformer", "revinmlp")]
UNPAIRED = ("tsslcd_tuned", "tsslcd_default")
RECEIVERS = ("humanities", "madsen", "moran", "sagepoint")


def class_series(run: dict, receiver: str, shorts: tuple[str, ...]):
    """Per-origin per-horizon mean error over the class's target bins (real input), (origins, horizons), and the number of bins."""
    cnt = run[f"cnt|{receiver}|real"]
    n = int(sum(cnt[{"P": 0, "N": 1, "M": 2}[k]] for k in shorts))
    if n == 0:
        return None, 0
    return sum(run[f"clsr|{receiver}|real|{k}"] for k in shorts) / n, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="frequency-ablation-2026-11-classes")
    ap.add_argument("--out", default="evaluation/results/frequency_benefit_by_class.csv")
    args = ap.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    runs = list_runs(client, args.prefix)
    cache: dict = {}

    def run(model, band, seed):
        key = (model, band, seed)
        if key not in cache:
            cache[key] = load_run(client, runs[key]) if key in runs else None
        return cache[key]

    rows = []
    for band in ("600_800", "2400_2600"):
        for a_model, b_model in PAIRS:
            seeds = sorted(s for (m, b, s) in runs if m == a_model and b == band and (b_model, band, s) in runs)
            if not seeds:
                continue
            for tname, shorts in (("P", ("P",)), ("N", ("N",))):
                series = {}
                for s in seeds:
                    ra, rb = run(a_model, band, s), run(b_model, band, s)
                    for r in RECEIVERS:
                        xa, n = class_series(ra, r, shorts)
                        xb, _ = class_series(rb, r, shorts)
                        if xa is None or xb is None or xa.shape != xb.shape:
                            continue
                        series[(s, r)] = (xa.astype(np.float64), xb.astype(np.float64), n, ra[f"starts|{r}"])
                if not series:
                    continue
                horizons = [int(h) for h in run(a_model, band, seeds[0])["horizons"]]

                def emit(seed_label, receiver_label, members):
                    ws = np.array([series[m][2] if receiver_label == "pooled" else 1.0 for m in members], dtype=float)
                    ws /= ws.sum()
                    starts = series[members[0]][3]
                    a_ = sum(w * series[m][0] for w, m in zip(ws, members))
                    b_ = sum(w * series[m][1] for w, m in zip(ws, members))
                    main = starts >= MAIN_MIN_START
                    for hlabel, js in (("mean", list(range(len(horizons)))), *[(h, [j]) for j, h in enumerate(horizons)]):
                        da, db = a_[:, js].mean(axis=1)[main], b_[:, js].mean(axis=1)[main]
                        lo, hi = boot(da - db)
                        rows.append({"band": band, "cross_bin_model": a_model, "per_bin_model": b_model, "seed": seed_label, "receiver": receiver_label, "target_class": tname,
                                     "horizon": hlabel, "mae_cross_bin": float(da.mean()), "mae_per_bin": float(db.mean()), "diff": float((da - db).mean()),
                                     "ci_low": lo, "ci_high": hi, "n_target_bins": int(sum(series[m][2] for m in members if m[0] == members[0][0])),
                                     "n_origins": int(main.sum()), "n_seeds": len(set(m[0] for m in members)), "paired": True})

                recvs = sorted({r for _, r in series})
                for r in recvs:
                    emit("mean", r, [(s, r) for s in seeds if (s, r) in series])
                emit("mean", "pooled", [(s, r) for s in seeds for r in recvs if (s, r) in series])
        for model in UNPAIRED:
            seeds = sorted(s for (m, b, s) in runs if m == model and b == band)
            for tname, shorts in (("P", ("P",)), ("N", ("N",))):
                vals = []
                for s in seeds:
                    rr = run(model, band, s)
                    for r in RECEIVERS:
                        xa, n = class_series(rr, r, shorts)
                        if xa is not None:
                            vals.append((xa[rr[f"starts|{r}"] >= MAIN_MIN_START].mean(), n))
                if vals:
                    w = np.array([v[1] for v in vals], dtype=float)
                    rows.append({"band": band, "cross_bin_model": model, "per_bin_model": "", "seed": "mean", "receiver": "pooled", "target_class": tname, "horizon": "mean",
                                 "mae_cross_bin": float(np.average([v[0] for v in vals], weights=w)), "mae_per_bin": np.nan, "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                                 "n_target_bins": int(w[:len(RECEIVERS)].sum()), "n_origins": np.nan, "n_seeds": len(seeds), "paired": False})
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(len(table), "rows")
    show = table[(table.receiver == "pooled") & (table.horizon == "mean") & (table.paired)]
    print(show[["band", "cross_bin_model", "per_bin_model", "target_class", "mae_cross_bin", "mae_per_bin", "diff", "ci_low", "ci_high", "n_seeds"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
