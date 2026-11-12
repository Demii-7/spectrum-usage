"""Centred against as-trained, paired, for the frozen neural 4D finals (15 minutes, October test week; task 3 of the neural-4D brief).

Is the neural models' receiver-level error a level failure? Each tuning final (STS-PredNet, ResidualConvLSTM: tuned; DSwinLSTM-I: inherited:1min-r2
configuration) is scored at the four core receivers on the ``core4-p2`` map, readout unchanged (IDW over the grid cells with the map's own power),
twice on IDENTICAL origins (``training.ray.spatial15_centered``): as trained, and with per-window centring (the lookback mean of every bin removed from
the input before prediction and restored afterwards; ``own`` = the receiver's own window mean is added to its sampled forecast (the brief's
definition), ``map`` = the map's window mean is added back to the forecast map before it is read at the receivers).

Per origin the error is the mean over horizons of the MAE over the 200 bins, averaged over the seeds that exist; ``delta`` = centred minus as trained
(negative: centring helps = the model had a level problem), moving-block bootstrap over origins (block 96, 2,000 draws), receivers resampled
independently and pooled as in ``map_construction_vs_per_receiver``. Horizon sets: ``h4`` = t+4/12/24/48 (the paper's four), ``h3`` = t+4/12/24 (1, 3, 6 h).

    python evaluation/analysis/spatial15_centered_table.py --out evaluation/results/spatial15_centered.csv [--upload]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import map_construction_vs_per_receiver as M  # noqa: E402
from training.ray.storage import spectrum_s3_client  # noqa: E402

PREFIX = "spatial15-centered-oct2026"
MODELS = ("stsprednet", "residualconvlstm", "dswinlstm_i")
CONFIG = {"stsprednet": "tuned", "residualconvlstm": "tuned", "dswinlstm_i": "inherited:1min-r2"}
SEEDS = (40, 41, 42, 43, 44)
HORIZON_SETS = {"h4": (4, 12, 24, 48), "h3": (4, 12, 24)}
VARIANTS = {"own": "mae_center_own", "map": "mae_center_map"}


def load(client, band: str, model: str, receiver_4d: str, key: str, horizons: tuple[int, ...]):
    """Seed-averaged per-origin mean-over-horizons error series for one array key, and the seeds used."""
    frames, seeds = [], []
    for seed in SEEDS:
        try:
            with np.load(io.BytesIO(M.read(client, f"{PREFIX}/{band}/core4-p2/{model}/seed-{seed}/per_origin.npz")), allow_pickle=True) as npz:
                hs = list(npz["horizons"])
                pick = [hs.index(h) for h in horizons]
                frames.append(pd.Series(npz[f"{receiver_4d}|{key}"][:, pick].mean(axis=1), index=M.utc(npz[f"{receiver_4d}|times"])))
                seeds.append(seed)
        except Exception:
            continue
    if not frames:
        return None, []
    return pd.concat(frames, axis=1).mean(axis=1, skipna=False).dropna().rename("mae").to_frame(), seeds


def compare(client) -> pd.DataFrame:
    rows = []
    for band in M.BANDS:
        for model in MODELS:
            for hname, horizons in HORIZON_SETS.items():
                for variant, array_key in VARIANTS.items():
                    entries, per_receiver = [], {}
                    for r4 in M.RECEIVERS:
                        base, seeds = load(client, band, model, r4, "mae", horizons)
                        cent, seeds_c = load(client, band, model, r4, array_key, horizons)
                        if base is None or cent is None:
                            break
                        delta, n, mean_c, mean_b = M.paired(cent, base)
                        draws = M.block_bootstrap(delta)
                        entries.append((n, mean_b, mean_c, float(delta.mean()), draws))
                        per_receiver[r4] = (n, mean_b, mean_c, float(delta.mean()), np.percentile(draws, 2.5), np.percentile(draws, 97.5))
                    if len(entries) != len(M.RECEIVERS):
                        continue
                    common = dict(band=band, model=model, config_label=CONFIG[model], map_variant="core4-p2", readout="IDW over grid cells, map's own power (2)",
                                  centring=variant, horizons=hname, n_seeds=len(seeds), seeds=",".join(map(str, seeds)))
                    for r4, (n, mb, mc, d, lo, hi) in per_receiver.items():
                        rows.append({**common, "receiver": r4, "n_origins": n, "mae_as_trained": mb, "mae_centered": mc, "delta_centered_minus_as_trained": d,
                                     "ci_low": float(lo), "ci_high": float(hi)})
                    pooled = np.mean(np.stack([e[4] for e in entries]), axis=0)
                    rows.append({**common, "receiver": "pooled", "n_origins": int(sum(e[0] for e in entries)),
                                 "mae_as_trained": float(np.mean([e[1] for e in entries])), "mae_centered": float(np.mean([e[2] for e in entries])),
                                 "delta_centered_minus_as_trained": float(np.mean([e[3] for e in entries])),
                                 "ci_low": float(np.percentile(pooled, 2.5)), "ci_high": float(np.percentile(pooled, 97.5))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="evaluation/results/spatial15_centered.csv")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    client = spectrum_s3_client()
    table = compare(client)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    if len(table):
        for band in M.BANDS:
            for variant in VARIANTS:
                d = table[(table.band == band) & (table.centring == variant) & (table.horizons == "h4") & (table.receiver == "pooled")]
                if d.empty:
                    continue
                print(f"\n{band}, centring '{variant}', pooled over the four receivers, mean over t+4/12/24/48 (dB; delta = centred minus as trained; 95% block-bootstrap)")
                for _, r in d.iterrows():
                    print(f"  {r.model:18s} {r.config_label:18s} seeds {r.n_seeds}: as trained {r.mae_as_trained:.3f}  centred {r.mae_centered:.3f}  {r.delta_centered_minus_as_trained:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  n={r.n_origins}")
    if a.upload:
        for f in (out,):
            client.put_object(Bucket=M.BUCKET, Key="analysis-outputs/" + f.name, Body=f.read_bytes())
            print("uploaded", f.name)


if __name__ == "__main__":
    main()
