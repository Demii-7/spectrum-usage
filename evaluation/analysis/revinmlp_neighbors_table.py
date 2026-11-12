"""RevIN-MLP with neighbour inputs versus plain RevIN-MLP at the four physical receivers (15 minutes, October test week, seeds 40-42).

Both are the tuned direct finals of ``training.ray.tune_direct`` (``revinmlp`` and ``revinmlp_nbr``, each with its own 16-candidate search at seed 42);
the per-origin errors (``training.ray.origin_errors`` for plain, ``training.ray.neighbor_origin_errors`` for the neighbour model) are matched on the origin
time (and horizon, receiver, seed), so the comparison is on identical forecast times and receivers. Per seed the error per origin is the MAE over the 200
bins; seeds are averaged per origin, then paired differences (neighbours minus plain, dB; negative = neighbours help) with a moving-block bootstrap over
origins (block 96 = one day, 2,000 draws, seed 0), pooled over the receivers as in ``map_construction_vs_per_receiver``. ``mean_4h`` = mean over horizons
4/12/24/48.

    python evaluation/analysis/revinmlp_neighbors_table.py --out evaluation/results/revinmlp_neighbors.csv
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

BANDS, CORE, HORIZONS, SEEDS = ("600_800", "2400_2600"), ("humanities", "madsen", "moran", "sagepoint"), (4, 12, 24, 48), (40, 41, 42)
PREFIX = "origin-errors-oct2026"
BLOCK, DRAWS = 96, 2000
RNG = np.random.default_rng(0)


def read(client, band, receiver, model, h, seed):
    try:
        raw = client.get_object(Bucket="spectrum", Key=f"{PREFIX}/15min/{band}/{receiver}/direct/{model}/h{h}/seed-{seed}/per_origin.csv")["Body"].read()
        return pd.read_csv(io.BytesIO(raw)).set_index("time").mae_db
    except Exception:
        return None


def seed_mean(client, band, receiver, model, h):
    series = [s for s in (read(client, band, receiver, model, h, seed) for seed in SEEDS) if s is not None]
    if not series:
        return None, 0
    return pd.concat(series, axis=1).mean(axis=1, skipna=False).dropna(), len(series)


def bootstrap(delta):
    n = len(delta)
    block = BLOCK if n >= 2 * BLOCK else max(1, n // 4)
    starts = RNG.integers(0, n - block + 1, size=(DRAWS, int(np.ceil(n / block))))
    return delta[(starts[:, :, None] + np.arange(block)[None, None, :]).reshape(DRAWS, -1)[:, :n]].mean(axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="evaluation/results/revinmlp_neighbors.csv")
    args = p.parse_args()
    client = spectrum_s3_client()
    rows = []
    for band in BANDS:
        per_h = {}
        for h in HORIZONS:
            entries = {}
            for r in CORE:
                nb, n_nb = seed_mean(client, band, r, "revinmlp_nbr", h)
                pl, n_pl = seed_mean(client, band, r, "revinmlp", h)
                if nb is None or pl is None:
                    continue
                common = nb.index.intersection(pl.index)
                x, y = nb.loc[common].to_numpy(), pl.loc[common].to_numpy()
                entries[r] = (x, y, n_nb, n_pl)
            per_h[h] = entries
            for r, (x, y, n_nb, n_pl) in entries.items():
                d = bootstrap(x - y)
                rows.append({"band": band, "horizon": h, "receiver": r, "n_origins": len(x), "seeds_neighbors": n_nb, "seeds_plain": n_pl,
                             "mae_neighbors": x.mean(), "mae_plain": y.mean(), "delta": (x - y).mean(),
                             "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
            if len(entries) == len(CORE):
                draws = np.mean([bootstrap(e[0] - e[1]) for e in entries.values()], axis=0)
                rows.append({"band": band, "horizon": h, "receiver": "pooled", "n_origins": sum(len(e[0]) for e in entries.values()),
                             "seeds_neighbors": min(e[2] for e in entries.values()), "seeds_plain": min(e[3] for e in entries.values()),
                             "mae_neighbors": np.mean([e[0].mean() for e in entries.values()]), "mae_plain": np.mean([e[1].mean() for e in entries.values()]),
                             "delta": np.mean([(e[0] - e[1]).mean() for e in entries.values()]),
                             "ci_low": np.percentile(draws, 2.5), "ci_high": np.percentile(draws, 97.5)})
        # mean over the four horizons (origins matched per horizon; draws averaged, as in map_construction_vs_per_receiver)
        if all(len(per_h[h]) == len(CORE) for h in HORIZONS):
            pooled = [r for r in rows if r["band"] == band and r["receiver"] == "pooled"]
            rows.append({"band": band, "horizon": "mean_4h", "receiver": "pooled", "n_origins": sum(r["n_origins"] for r in pooled),
                         "seeds_neighbors": min(r["seeds_neighbors"] for r in pooled), "seeds_plain": min(r["seeds_plain"] for r in pooled),
                         "mae_neighbors": np.mean([r["mae_neighbors"] for r in pooled]), "mae_plain": np.mean([r["mae_plain"] for r in pooled]),
                         "delta": np.mean([r["delta"] for r in pooled]), "ci_low": np.nan, "ci_high": np.nan})
    frame = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print(frame.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
