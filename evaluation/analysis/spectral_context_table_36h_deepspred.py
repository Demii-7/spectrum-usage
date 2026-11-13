#!/usr/bin/env python3
"""3D-SwinSTB (``deepspred2d``) rows of the 36-hour spectral-context table (600-800 only; same layout and conventions as ``spectral_context_table_36h.csv``).

Reads ``frequency-ablation-2026-11-classes-deepspred/deepspred2d/600_800/seed-N/h<h>/per_origin.npz`` (``training.ray.frequency_ablation_deepspred``) and reuses ``cell`` and
``QUANTITIES`` of ``spectral_context_table_36h.py``: three-horizon mean over t+4/12/24, receivers pooled by target-bin count, seed mean, seed sd over per-seed values, interval =
moving-block bootstrap over origins (block 96) of the seed-averaged per-origin difference, seeds behind every cell. Quantities: mae_real, all_24h (all x shift96), all_36h (all x
shift144), N_36h / P_36h (N_only / P_only x shift144), P_only_36h on periodic and on non-periodic target bins, all_52h (all x shift210). Output files carry the suffix ``_deepspred``;
the paper table files of fork AJ are not touched.

    python evaluation/analysis/spectral_context_table_36h_deepspred.py [--out evaluation/results] [--upload]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frequency_ablation_classes_table as ft  # noqa: E402
import spectral_context_table_36h as sc  # noqa: E402

MODEL, BAND = "deepspred2d", "600_800"
PREFIX = "frequency-ablation-2026-11-classes-deepspred"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--out", default="evaluation/results")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    runs = ft.list_runs(client, a.prefix)
    seeds_all = sorted({s for (m, b, s) in runs if m == MODEL and b == BAND})
    data, partial = {}, []
    for s in seeds_all:
        try:
            d = sc.merged_run(client, runs.get((MODEL, BAND, s)), None)
            hz = {int(h) for h in d["horizons"]}
            if not set(sc.HORIZONS) <= hz:
                partial.append((s, sorted(hz)))
                continue
            data[s] = d
        except Exception as e:  # noqa: BLE001
            partial.append((s, str(e)[:60]))
    rows = []
    for qname, cset, dist, tname in sc.QUANTITIES:
        cond = f"{cset}_{dist}"
        c = sc.cell(data, cond, tname)
        if c is None:
            continue
        diff = c["dis"] - c["real"]
        lo, hi = ft.boot(diff)
        ps = c["per_seed"]
        seed_diffs = [v["dis"] - v["real"] for v in ps.values()]
        reals = [v["real"] for v in ps.values()]
        is_real = qname == "mae_real"
        rows.append({"band": BAND, "model": MODEL, "quantity": qname, "disturbed_set": cset, "disturbance": dist, "target_class": tname, "horizons": "t+4/12/24",
                     "mean": float(np.mean(reals) if is_real else np.mean(seed_diffs)),
                     "seed_sd": float(np.std(reals if is_real else seed_diffs, ddof=1)) if len(ps) > 1 else np.nan,
                     "ci_low": np.nan if is_real else lo, "ci_high": np.nan if is_real else hi, "mae_real": float(np.mean(reals)),
                     "mae_disturbed": float(np.mean([v["dis"] for v in ps.values()])), "n_seeds": len(ps), "seeds": ",".join(map(str, c["seeds"])),
                     "n_target_bins": int(next(iter(ps.values()))["n_bins"]), "n_origins": int(next(iter(ps.values()))["n"])})
    table = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "spectral_context_table_36h_deepspred.csv", index=False)
    lines = ["3D-SwinSTB (deepspred2d), 600-800: MAE and differences in dB; mean over t+4/12/24; receivers pooled by target-bin count; seed mean [n seeds]; interval = block bootstrap over origins", ""]
    for _, r in table.iterrows():
        val = f"{r['mean']:.3f}" if r["quantity"] == "mae_real" else f"{r['mean']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] (sd {r['seed_sd']:.3f})"
        lines.append(f"  {r['quantity']:28s} target {r['target_class']:3s} {val}  seeds {int(r['n_seeds'])} ({r['seeds']}), {int(r['n_target_bins'])} target bins")
    lines += ["", f"seeds complete: {sorted(data)}; partial: {partial or '-'}"]
    (out / "spectral_context_table_36h_deepspred.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if a.upload:
        for f in ("spectral_context_table_36h_deepspred.csv", "spectral_context_table_36h_deepspred.txt"):
            client.put_object(Bucket="spectrum", Key="analysis-outputs/" + f, Body=(out / f).read_bytes())
            print("uploaded", f)


if __name__ == "__main__":
    main()
