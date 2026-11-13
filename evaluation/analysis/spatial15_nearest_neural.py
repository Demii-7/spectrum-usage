#!/usr/bin/env python3
"""Nearest-cell readout for the NEURAL 4D finals (STS-PredNet, ResidualConvLSTM, DSwinLSTM-I), default map ``core4-p2``, 15 minutes.

Companion of ``spatial15_nearest_readout.py`` (linear 4D finals). The neural finals were rescored with ``--readout nearest_cell`` under
``spatial15-origin-errors-oct2026-nearest``; the as-evaluated reference is the standard ``spatial15-origin-errors-oct2026`` (MAE only; MSE not
available for the neural models). Per model and band, on identical origins, seeds averaged per origin, moving-block bootstrap (block 96, 2,000 draws):
(1) nearest_cell minus as_evaluated receiver MAE (per receiver and pooled, mean over t+4/12/24/48 and per horizon);
(2) the gap to the per-receiver 2D ResidualLinearAR final and to persistence under both readouts (4D model minus the other model).

    python evaluation/analysis/spatial15_nearest_neural.py [--out evaluation/results/spatial15_nearest_neural] [--upload]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import neighbor_value as nv  # noqa: E402
import spatial15_nearest_readout as nr  # noqa: E402
from training.ray.storage import spectrum_s3_client  # noqa: E402

MODELS = ("stsprednet", "residualconvlstm", "dswinlstm_i")
VARIANT = "core4-p2"
PREFIX_AS, PREFIX_NEAR = "spatial15-origin-errors-oct2026", "spatial15-origin-errors-oct2026-nearest"
SEEDS = (40, 41, 42, 43, 44)


def cells(client, prefix, band, model):
    out = {}
    for s in SEEDS:
        c = nr.read_cell(client, prefix, band, VARIANT, model, s)
        if c is not None:
            out[s] = c
    return out


def series(cs, receiver):
    return nr.seed_mean_series(cs, receiver, "mean").to_frame("mae")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="evaluation/results/spatial15_nearest_neural")
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    client = spectrum_s3_client()
    paired, gaps = [], []
    for band in nr.BANDS:
        for model in MODELS:
            as_c, near_c = cells(client, PREFIX_AS, band, model), cells(client, PREFIX_NEAR, band, model)
            if not as_c or not near_c:
                print(f"[skip] {band} {model}: as_evaluated seeds {sorted(as_c)}, nearest seeds {sorted(near_c)}")
                continue
            paired += nr.paired_rows(band, VARIANT, model, as_c, near_c)
            seeds_common = sorted(set(as_c) & set(near_c))
            for receiver_4d, receiver in nv.RECEIVERS.items():
                refs = {"2d residuallinearar2d": nv.per_receiver_series(client, band, receiver, "2d", "residuallinearar2d"),
                        "persistence": nv.persistence_series(client, band, receiver)}
                for readout, cs in (("as_evaluated", {s: as_c[s] for s in seeds_common}), ("nearest_cell", {s: near_c[s] for s in seeds_common})):
                    ours = series(cs, receiver_4d)
                    for name, other in refs.items():
                        if other is None:
                            continue
                        delta, n, ma, mb = nv.paired(ours, other)
                        draws = nv.block_bootstrap_delta(delta)
                        gaps.append({"band": band, "model": model, "variant": VARIANT, "readout": readout, "receiver": receiver_4d, "versus": name,
                                     "n_seeds": len(seeds_common), "n_origins": n, "mae_4d": ma, "mae_other": mb, "delta_4d_minus_other": float(delta.mean()),
                                     "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5)), "draws": draws})
    paired = pd.DataFrame(paired)
    g = pd.DataFrame(gaps)
    pooled = []
    for keys, sub in g.groupby(["band", "model", "readout", "versus"]):
        d = np.mean(np.stack(sub["draws"].to_numpy()), axis=0)
        pooled.append({"band": keys[0], "model": keys[1], "variant": VARIANT, "readout": keys[2], "receiver": "pooled", "versus": keys[3],
                       "n_seeds": int(sub.n_seeds.iloc[0]), "n_origins": int(sub.n_origins.mean()), "mae_4d": sub.mae_4d.mean(), "mae_other": sub.mae_other.mean(),
                       "delta_4d_minus_other": sub.delta_4d_minus_other.mean(), "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5))})
    g = pd.concat([g.drop(columns="draws"), pd.DataFrame(pooled)], ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(a.out + "_paired.csv", index=False)
    g.to_csv(a.out + "_gaps.csv", index=False)
    lines = ["Neural 4D finals, core4-p2: nearest-cell minus as-evaluated receiver MAE (pooled, mean over t+4/12/24/48), and the gap to per-receiver models", ""]
    pp = paired[(paired.receiver == "pooled") & (paired.horizon == "mean")]
    for _, r in pp.iterrows():
        lines.append(f"{r.band} {r.model:18s} seeds {r.n_seeds}: as_evaluated {r.mae_as_evaluated:.3f}  nearest_cell {r.mae_nearest_cell:.3f}  "
                     f"delta {r.delta_nearest_minus_as_evaluated:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]")
    lines.append("")
    gp = g[(g.receiver == "pooled")]
    for _, r in gp.sort_values(["band", "model", "versus", "readout"]).iterrows():
        lines.append(f"{r.band} {r.model:18s} vs {r.versus:22s} {r.readout:13s}: {r.delta_4d_minus_other:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  "
                     f"(4D {r.mae_4d:.3f}, other {r.mae_other:.3f})")
    Path(a.out + ".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if a.upload:
        for suffix in ("_paired.csv", "_gaps.csv", ".txt"):
            f = a.out + suffix
            client.put_object(Bucket="spectrum", Key="analysis-outputs/" + Path(f).name, Body=open(f, "rb").read())
            print("uploaded", f)


if __name__ == "__main__":
    main()
