#!/usr/bin/env python3
"""Does the four-receiver map help the neural 4D model, compared with ITSELF trained on one receiver alone? (15 minutes, October test week)

For each band, core receiver R and seed (40, 41, 42) of a 4D model (STS-PredNet; the DSwinLSTM-I rows are added when they exist):

  core4  the tuned winner trained on the four-receiver map (``core4-p2``), scored at R: per-origin errors of the neighbour-ablation 'real' mode
         (``neighbor-ablation-oct2026``, the same weights and geometry) or, if present, ``spatial15-origin-errors-oct2026``
  solo   the same tuned winner trained on a map built from R alone (``solo-<R>``: grid frame unchanged, so the IDW map is constant in space),
         scored at R: ``spatial15-solo-origin-errors-oct2026/<band>/solo-<R>/<model>/seed-N/per_origin.npz``

Both are scored at the physical receiver with ``sample_forecast_grid`` (IDW with the map's own power; unchanged readout), on IDENTICAL origins (paired on
the origin time). Per origin the error is the mean over horizons of the MAE over the 200 bins, averaged over the seeds that exist for BOTH models.
delta = core4 - solo (dB; < 0: the four-receiver map helps). Moving-block bootstrap over origins (block 96 = one day, 2,000 draws); the pooled row averages
the per-receiver bootstrap draws (receivers resampled independently), as in ``map_construction_vs_per_receiver``. Two horizon sets: ``t+4/12/24/48`` and
``t+4/12/24`` (1, 3, 6 h). Output: one long CSV; rows for a single seed or fewer than expected seeds are marked in ``seed_status``.

    python evaluation/analysis/solo_vs_core4.py --out evaluation/results/solo_vs_core4.csv [--upload]
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

BANDS = M.BANDS
RECEIVERS = ("humanities", "madsen", "moran", "sage point")
SOLO_PREFIX = "spatial15-solo-origin-errors-oct2026"
NBR_PREFIX = "neighbor-ablation-oct2026"
SEEDS = {"stsprednet": (40, 41, 42), "dswinlstm_i": (42,)}
HORIZON_SETS = {"t4_12_24_48": (4, 12, 24, 48), "t4_12_24": (4, 12, 24)}
CONFIG = {"stsprednet": "tuned 15-minute winner (16 trials, search seed 42)",
          "dswinlstm_i": "see CAMPAIGN_STATUS (inherited:1min-r2 or the tuned label of the 15-minute search)"}


def solo_name(receiver: str) -> str:
    return "solo-" + receiver.replace(" ", "")


def series(client, key: str, receiver: str, mode: str | None, horizons: tuple[int, ...]):
    mid = f"|{mode}" if mode else ""
    try:
        with np.load(io.BytesIO(M.read(client, key)), allow_pickle=True) as npz:
            all_h = list(npz["horizons"])
            pick = [all_h.index(h) for h in horizons]
            return pd.Series(npz[f"{receiver}{mid}|mae"][:, pick].mean(axis=1), index=M.utc(npz[f"{receiver}{mid}|times"]))
    except Exception:
        return None


def core4_series(client, band: str, model: str, seed: int, receiver: str, horizons):
    s = series(client, f"{M.ORIGIN_PREFIX}/{band}/core4-p2/{model}/seed-{seed}/per_origin.npz", receiver, None, horizons)
    if s is None:
        s = series(client, f"{NBR_PREFIX}/{band}/{model}/seed-{seed}/per_origin.npz", receiver, "real", horizons)
    return s


def compare(client, models) -> pd.DataFrame:
    rows = []
    for model in models:
        for hname, horizons in HORIZON_SETS.items():
            for band in BANDS:
                per_receiver_draws, per_seed_rows = {}, []
                for receiver in RECEIVERS:
                    core, solo, used = [], [], []
                    for seed in SEEDS[model]:
                        c = core4_series(client, band, model, seed, receiver, horizons)
                        s = series(client, f"{SOLO_PREFIX}/{band}/{solo_name(receiver)}/{model}/seed-{seed}/per_origin.npz", receiver, None, horizons)
                        if c is None or s is None:
                            continue
                        core.append(c)
                        solo.append(s)
                        used.append(seed)
                        common = c.index.intersection(s.index)
                        d = (c.loc[common] - s.loc[common]).to_numpy()
                        per_seed_rows.append({"band": band, "model": model, "horizon_set": hname, "receiver": receiver, "seed": seed,
                                              "n_origins": len(common), "mae_core4": float(c.loc[common].mean()), "mae_solo": float(s.loc[common].mean()),
                                              "delta_core4_minus_solo": float(d.mean()), "ci_low": np.nan, "ci_high": np.nan,
                                              "n_seeds": 1, "seed_status": "single seed"})
                    if not used:
                        continue
                    a = pd.concat(core, axis=1).mean(axis=1, skipna=False).dropna().rename("mae").to_frame()
                    b = pd.concat(solo, axis=1).mean(axis=1, skipna=False).dropna().rename("mae").to_frame()
                    delta, n, mean_a, mean_b = M.paired(a, b)
                    draws = M.block_bootstrap(delta)
                    per_receiver_draws[receiver] = draws
                    status = "complete" if len(used) == len(SEEDS[model]) else f"partial seeds {used}"
                    rows.append({"band": band, "model": model, "horizon_set": hname, "receiver": receiver, "seed": "mean", "n_origins": n,
                                 "mae_core4": mean_a, "mae_solo": mean_b, "delta_core4_minus_solo": float(delta.mean()),
                                 "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5)),
                                 "n_seeds": len(used), "seed_status": status})
                rows.extend(per_seed_rows)
                got = [r for r in rows if r["band"] == band and r["model"] == model and r["horizon_set"] == hname and r["seed"] == "mean"]
                if got:
                    pooled = np.mean([per_receiver_draws[r["receiver"]] for r in got], axis=0)
                    rows.append({"band": band, "model": model, "horizon_set": hname, "receiver": "pooled", "seed": "mean",
                                 "n_origins": int(np.mean([r["n_origins"] for r in got])),
                                 "mae_core4": float(np.mean([r["mae_core4"] for r in got])), "mae_solo": float(np.mean([r["mae_solo"] for r in got])),
                                 "delta_core4_minus_solo": float(np.mean([r["delta_core4_minus_solo"] for r in got])),
                                 "ci_low": float(np.percentile(pooled, 2.5)), "ci_high": float(np.percentile(pooled, 97.5)),
                                 "n_seeds": int(min(r["n_seeds"] for r in got)),
                                 "seed_status": "complete" if all(r["seed_status"] == "complete" for r in got) and len(got) == 4 else "partial"})
    out = pd.DataFrame(rows)
    out["configuration"] = out["model"].map(CONFIG)
    out["readout"] = "sample_forecast_grid, IDW with the map's own power (unchanged)"
    out["map_core4"] = "core4-p2"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="evaluation/results/solo_vs_core4.csv")
    p.add_argument("--models", nargs="+", default=["stsprednet"])
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    client = spectrum_s3_client()
    table = compare(client, a.models)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    show = table[(table.seed == "mean") & (table.horizon_set == "t4_12_24_48")]
    print("four-receiver minus solo (dB, < 0: the four-receiver map helps), mean over seeds, mean over t+4/12/24/48:")
    print(show[["band", "model", "receiver", "n_seeds", "n_origins", "mae_core4", "mae_solo", "delta_core4_minus_solo", "ci_low", "ci_high",
                "seed_status"]].round(3).to_string(index=False))
    if a.upload:
        client.put_object(Bucket="spectrum", Key="analysis-outputs/" + Path(a.out).name, Body=Path(a.out).read_bytes())
        print("uploaded", a.out)


if __name__ == "__main__":
    main()
