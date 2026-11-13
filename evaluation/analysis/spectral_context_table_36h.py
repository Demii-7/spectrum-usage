#!/usr/bin/env python3
"""Spectral-context table of paper section 5.3 at a 36-hour shift (15 minutes; evaluation only; reads the class-crossed frozen-input ablation).

Reads ``<prefix>/<model>/<band>/seed-N[/h<h>]/per_origin.npz`` (default ``frequency-ablation-2026-11-classes``; ``training/ray/frequency_ablation_classes.py``)
and, for Autoformer-CSA, merges the restricted 36-hour run stored under ``--extra-prefix`` (default ``frequency-ablation-2026-11-classes-af36``): conditions
that exist in both objects are taken from the extra object when they are identical origins, others from whichever has them.

Quantities (three-horizon mean over t+4, t+12, t+24 = 1, 3, 6 hours; t+48 excluded; target bins = bins of the non-noise plan regions; receivers pooled by bin count;
seeds averaged per origin; ``seed_sd`` = sd over the per-seed pooled values; interval = moving-block bootstrap over origins (block 96, 2,000 draws) of the
seed-averaged per-origin difference):
  mae_real          MAE with real inputs (condition ``all_shift96``, target class all)
  all_24h           all other bins shifted 24 h        (all x shift96; time of day kept: the control)
  all_36h           all other bins shifted 36 h        (all x shift144; 12 h out of phase)
  N_36h, P_36h      only non-periodic / periodic other bins shifted 36 h (600-800 only)
and for sentences: N_only / P_only x shift96, P_only x shift144 on periodic (P) and non-periodic (N) target bins, all_52h (all x shift210, reproduction check).
Only seeds whose object holds all three horizons are used; the seeds behind every cell are listed (``seeds``, ``n_seeds``).

    python evaluation/analysis/spectral_context_table_36h.py [--out evaluation/results] [--models ...] [--upload]
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frequency_ablation_classes_table as ft  # noqa: E402

HORIZONS = (4, 12, 24)
PAPER_MODELS = ("lstmattn", "tsslcd_tuned", "autoformer_csa", "vanillalstm", "residualvanillalstm")
TEXT_MODELS = ("revinmlp_mix", "patchtst_joint", "tsslcd_default")
CONTROLS = ("revinmlp", "patchtst", "linearar2d")
# (quantity, disturbed set, disturbance, target class)
QUANTITIES = [("mae_real", "all", "shift96", "all"), ("all_24h", "all", "shift96", "all"), ("all_36h", "all", "shift144", "all"),
              ("N_36h", "N_only", "shift144", "all"), ("P_36h", "P_only", "shift144", "all"),
              ("N_24h", "N_only", "shift96", "all"), ("P_24h", "P_only", "shift96", "all"),
              ("P_only_36h_on_P_targets", "P_only", "shift144", "P"), ("P_only_36h_on_N_targets", "P_only", "shift144", "N"),
              ("P_only_24h_on_P_targets", "P_only", "shift96", "P"), ("P_only_24h_on_N_targets", "P_only", "shift96", "N"),
              ("N_only_24h_on_N_targets", "N_only", "shift96", "N"), ("N_only_36h_on_N_targets", "N_only", "shift144", "N"),
              ("all_52h", "all", "shift210", "all")]
# reference values quoted in the brief (three-horizon mean, pooled; 600-800 / 2400-2600): all x shift96 and all x shift210, five seeds
REF_ALL = {"lstmattn": ((0.072, 0.496), (0.077, 0.688)), "tsslcd_tuned": ((0.110, 0.532), (0.041, 0.720)), "vanillalstm": ((0.114, 0.490), (0.108, 0.568)),
           "residualvanillalstm": ((0.069, 0.469), (0.057, 0.722)), "autoformer_csa": ((0.047, 0.060), (0.009, 0.018))}
# seed 42, 600-800, target class all: P_only x shift144 and N_only x shift144
REF_42 = {"lstmattn": (0.772, 0.046), "tsslcd_tuned": (0.933, -0.001), "vanillalstm": (0.618, 0.040), "residualvanillalstm": (0.630, 0.080)}
TARGET = {"P": ("P",), "N": ("N",), "all": ("P", "N", "M")}


def merged_run(client, main_keys, extra_keys):
    data = ft.load_run(client, main_keys) if main_keys else None
    if extra_keys:
        extra = ft.load_run(client, extra_keys)
        if data is None:
            return extra
        merged = dict(data)
        conds = [str(c) for c in data["conditions"]]
        for c in [str(c) for c in extra["conditions"]]:
            if c in conds and c != "real":
                same = all(np.array_equal(data[f"starts|{r}"], extra[f"starts|{r}"]) for r in [str(x) for x in data["receivers"]])
                if not same:
                    continue
            if c != "real" or c not in conds:
                for k in extra:
                    if f"|{c}" in k and k.startswith(("cnt|", "cls|", "clsr|", "rmae|")):
                        merged[k] = extra[k]
            if c not in conds:
                conds.append(c)
        merged["conditions"] = np.array(conds)
        return merged
    return data


def cell(data_by_seed, cond, tname, horizons=HORIZONS):
    """-> dict with per-seed pooled (real, dis, diff) and the seed-averaged per-origin difference, or None."""
    shorts = TARGET[tname]
    seeds = sorted(data_by_seed)
    per_seed = {}
    arrays = []                                  # (weight, real(orig,h), dis(orig,h), starts)
    nowrap_min = None
    for s in seeds:
        d = data_by_seed[s]
        if cond not in [str(c) for c in d["conditions"]]:
            continue
        hz = [int(h) for h in d["horizons"]]
        js = [hz.index(h) for h in horizons]
        recs, ws = [], []
        for r in [str(x) for x in d["receivers"]]:
            cnt = d[f"cnt|{r}|{cond}"]
            n_bins = int(sum(cnt[{"P": 0, "N": 1, "M": 2}[k]] for k in shorts))
            if n_bins == 0:
                continue
            real = (sum(d[f"clsr|{r}|{cond}|{k}"] for k in shorts) / n_bins).astype(np.float64)[:, js]
            dis = (sum(d[f"cls|{r}|{cond}|{k}"] for k in shorts) / n_bins).astype(np.float64)[:, js]
            recs.append((real, dis, d[f"starts|{r}"], n_bins))
        if not recs or any(len(x[2]) != len(recs[0][2]) for x in recs):
            continue
        w = np.array([x[3] for x in recs], float)
        w /= w.sum()
        real = sum(wi * x[0] for wi, x in zip(w, recs))
        dis = sum(wi * x[1] for wi, x in zip(w, recs))
        starts = recs[0][2]
        main = starts >= ft.MAIN_MIN_START
        per_seed[s] = {"real": float(real[main].mean()), "dis": float(dis[main].mean()), "n_bins": int(sum(x[3] for x in recs)), "n": int(main.sum())}
        arrays.append((real[main].mean(axis=1), dis[main].mean(axis=1)))
    if not per_seed:
        return None
    n = min(len(a[0]) for a in arrays)
    real = np.mean([a[0][:n] for a in arrays], axis=0)
    dis = np.mean([a[1][:n] for a in arrays], axis=0)
    return {"seeds": sorted(per_seed), "per_seed": per_seed, "real": real, "dis": dis}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="frequency-ablation-2026-11-classes")
    ap.add_argument("--extra-prefix", default="frequency-ablation-2026-11-classes-af36")
    ap.add_argument("--out", default="evaluation/results")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    main_runs, extra_runs = ft.list_runs(client, a.prefix), ft.list_runs(client, a.extra_prefix)
    models = a.models or list(PAPER_MODELS + TEXT_MODELS + CONTROLS)
    rows, checks, completeness = [], [], []
    for model in models:
        for band in ("600_800", "2400_2600"):
            seeds_all = sorted({s for (m, b, s) in list(main_runs) + list(extra_runs) if m == model and b == band})
            data, partial = {}, []
            for s in seeds_all:
                try:
                    d = merged_run(client, main_runs.get((model, band, s)), extra_runs.get((model, band, s)))
                    hz = {int(h) for h in d["horizons"]}
                    if not set(HORIZONS) <= hz:
                        partial.append((s, sorted(hz)))
                        continue
                    data[s] = d
                except Exception as e:  # noqa: BLE001
                    partial.append((s, str(e)[:60]))
            completeness.append({"model": model, "band": band, "seeds_complete": ",".join(map(str, sorted(data))) or "-", "n_complete": len(data),
                                 "seeds_partial": ";".join(f"{s}:{h}" for s, h in partial) or "-"})
            if not data:
                continue
            for qname, cset, dist, tname in QUANTITIES:
                cond = f"{cset}_{dist}"
                c = cell(data, cond, tname)
                if c is None:
                    continue
                diff = c["dis"] - c["real"]
                lo, hi = ft.boot(diff)
                ps = c["per_seed"]
                seed_diffs = [v["dis"] - v["real"] for v in ps.values()]
                reals = [v["real"] for v in ps.values()]
                is_real = qname == "mae_real"
                rows.append({"band": band, "model": model, "quantity": qname, "disturbed_set": cset, "disturbance": dist, "target_class": tname,
                             "horizons": "t+4/12/24", "mean": float(np.mean(reals) if is_real else np.mean(seed_diffs)),
                             "seed_sd": float(np.std(reals if is_real else seed_diffs, ddof=1)) if len(ps) > 1 else np.nan,
                             "ci_low": np.nan if is_real else lo, "ci_high": np.nan if is_real else hi,
                             "mae_real": float(np.mean(reals)), "mae_disturbed": float(np.mean([v["dis"] for v in ps.values()])),
                             "n_seeds": len(ps), "seeds": ",".join(map(str, c["seeds"])), "n_target_bins": int(next(iter(ps.values()))["n_bins"]),
                             "n_origins": int(next(iter(ps.values()))["n"])})
            # checks: controls exactly zero; reference reproduction
            if model in CONTROLS:
                worst = 0.0
                for s, d in data.items():
                    for cnd in [str(x) for x in d["conditions"]]:
                        for r in [str(x) for x in d["receivers"]]:
                            for k in ("P", "N", "M"):
                                worst = max(worst, float(np.abs(d[f"cls|{r}|{cnd}|{k}"] - d[f"clsr|{r}|{cnd}|{k}"]).max()))
                checks.append({"check": "control: max |disturbed - real| over all conditions (class sums)", "model": model, "band": band, "value": worst, "reference": 0.0,
                               "difference": worst, "n_seeds": len(data)})
            if model in REF_ALL:
                bi = 0 if band == "600_800" else 1
                for qname, ref in (("all_24h", REF_ALL[model][bi][0]), ("all_52h", REF_ALL[model][bi][1])):
                    r = [x for x in rows if x["band"] == band and x["model"] == model and x["quantity"] == qname]
                    if r:
                        checks.append({"check": f"reference (brief, 5 seeds): {qname}", "model": model, "band": band, "value": r[0]["mean"], "reference": ref,
                                       "difference": r[0]["mean"] - ref, "n_seeds": r[0]["n_seeds"]})
            if band == "600_800" and model in REF_42 and 42 in data:
                c1 = {42: data[42]}
                for qname, cond, ref in (("P_only_shift144 seed 42", "P_only_shift144", REF_42[model][0]), ("N_only_shift144 seed 42", "N_only_shift144", REF_42[model][1])):
                    cc = cell(c1, cond, "all")
                    if cc:
                        v = cc["per_seed"][42]
                        checks.append({"check": f"reference (brief, seed 42): {qname}", "model": model, "band": band, "value": v["dis"] - v["real"], "reference": ref,
                                       "difference": v["dis"] - v["real"] - ref, "n_seeds": 1})
    # reproduction of the existing frozen-input ablation (frequency_ablation.csv, per-region deltas): the brief's reference values are UNWEIGHTED region means; the
    # class tables pool the target bins of all regions (weights = number of target bins), so the like-for-like reference is the same file weighted by n_bins
    try:
        old = pd.read_csv(io.BytesIO(ft.get(client, "analysis-outputs/frequency_ablation.csv")))
        old["horizon"] = pd.to_numeric(old["horizon"], errors="coerce")
        for r in [x for x in rows if x["quantity"] in ("all_24h", "all_52h")]:
            mode = "shift96" if r["quantity"] == "all_24h" else "shift210"
            sub = old[(old.model == r["model"]) & (old.band == r["band"]) & (old["mode"] == mode) & (old.horizon.isin(HORIZONS))]
            if sub.empty:
                continue
            reg = sub.groupby("region").delta.mean()
            nb = sub.groupby("region").n_bins.first()
            seeds_old = int(sub.n_seeds.iloc[0])
            checks.append({"check": f"existing ablation (frequency_ablation.csv), bin-weighted over regions: {r['quantity']}", "model": r["model"], "band": r["band"],
                           "value": r["mean"], "reference": float((reg * nb).sum() / nb.sum()), "difference": r["mean"] - float((reg * nb).sum() / nb.sum()),
                           "n_seeds": f"{r['n_seeds']} (existing {seeds_old})"})
            checks.append({"check": f"existing ablation (frequency_ablation.csv), unweighted region mean: {r['quantity']}", "model": r["model"], "band": r["band"],
                           "value": r["mean"], "reference": float(reg.mean()), "difference": r["mean"] - float(reg.mean()), "n_seeds": f"{r['n_seeds']} (existing {seeds_old})"})
    except Exception as e:  # noqa: BLE001
        print("existing-ablation reproduction skipped:", str(e)[:100])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "spectral_context_table_36h.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "spectral_context_table_36h_checks.csv", index=False)
    comp = pd.DataFrame(completeness)
    comp.to_csv(out / "spectral_context_table_36h_completeness.csv", index=False)
    lines = ["Spectral-context table at a 36-hour shift (MAE and differences in dB; mean over t+4/12/24; receivers pooled; seed mean [n seeds]; interval = block bootstrap over origins)", ""]
    for band in ("600_800", "2400_2600"):
        lines.append(f"{band}:")
        qs = ["mae_real", "all_24h", "all_36h"] + (["N_36h", "P_36h"] if band == "600_800" else [])
        head = f"  {'model':20s}" + "".join(f"{q:>26s}" for q in qs)
        lines.append(head)
        for model in PAPER_MODELS + TEXT_MODELS:
            cells = []
            for q in qs:
                r = table[(table.band == band) & (table.model == model) & (table.quantity == q)]
                if r.empty:
                    cells.append("-")
                    continue
                r = r.iloc[0]
                cells.append(f"{r['mean']:.3f} [{int(r.n_seeds)}]" if q == "mae_real" else f"{r['mean']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] [{int(r.n_seeds)}]")
            lines.append(f"  {model:20s}" + "".join(f"{c:>26s}" for c in cells))
        lines.append("")
    lines += ["completeness (complete = object holds t+4/12/24):", comp.to_string(index=False), "", "checks:", pd.DataFrame(checks).round(4).to_string(index=False)]
    (out / "spectral_context_table_36h.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if a.upload:
        for f in ("spectral_context_table_36h.csv", "spectral_context_table_36h_checks.csv", "spectral_context_table_36h_completeness.csv", "spectral_context_table_36h.txt"):
            client.put_object(Bucket="spectrum", Key="analysis-outputs/" + f, Body=(out / f).read_bytes())
            print("uploaded", f)


if __name__ == "__main__":
    main()
