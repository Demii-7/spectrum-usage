#!/usr/bin/env python3
"""Does the readout of the forecast map at the receivers change the 4D linear models' receiver errors? (15 minutes, October test week)

The paper's receiver errors read the forecast map with an IDW-weighted mean over ALL grid cells (``sample_forecast_grid``, the grid's own power;
kriging maps are read with power 2). ``smoothing_diagnostics.py`` shows that this read-out alone loses 0.46 dB (600-800) and 0.65 dB (2400-2600) on
the default map even for a perfect forecaster. Here the finished linear 4D cells are re-scored with the nearest-grid-node readout
(``--readout nearest_cell`` of ``training.ray.spatial15_origins``; per-origin errors under the new S3 prefix
``spatial15-origin-errors-oct2026-nearest``) and, for identical origins and MSE, with the as-evaluated readout re-run under
``spatial15-origin-errors-oct2026-asevaluated`` (the existing ``spatial15-origin-errors-oct2026`` holds MAE only).

Outputs (``--out`` stem):
* ``<out>.csv`` : band, variant, model, seed, readout, receiver, horizon, MAE, MSE, n_origins (receiver ``pooled`` = mean of the four receivers;
  horizon ``mean`` = mean of t+4/12/24/48).
* ``<out>_paired.csv`` : nearest_cell minus as_evaluated per (band, variant, model, receiver, horizon): seed-averaged per-origin MAE on identical
  origins, moving-block bootstrap over origins (block 96 = one day, 2,000 resamples); the pooled value averages the per-receiver draws.
* ``<out>.txt`` : per band the ranking of the map constructions by receiver MAE under both readouts.
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

BUCKET = "spectrum"
PREFIXES = {"as_evaluated": "spatial15-origin-errors-oct2026-asevaluated", "nearest_cell": "spatial15-origin-errors-oct2026-nearest"}
BANDS = ("600_800", "2400_2600")
VARIANTS = ("core4-p1", "core4-p2", "core4-p4", "core4-p16", "core4-p2-linear", "core4-p2-g5", "core4-p2-g20", "core4-krige")
SEEDS = (40, 41, 42, 43, 44)
RECEIVERS = ("humanities", "madsen", "moran", "sage point")
HORIZONS = (4, 12, 24, 48)
BLOCK, DRAWS = 96, 2000
RNG = np.random.default_rng(0)


def block_bootstrap(delta: np.ndarray) -> np.ndarray:
    """Moving-block bootstrap means of a paired per-origin difference (block 96)."""
    n = len(delta)
    blocks = int(np.ceil(n / BLOCK))
    starts = RNG.integers(0, max(n - BLOCK + 1, 1), size=(DRAWS, blocks))
    draws = np.empty(DRAWS)
    for d in range(DRAWS):
        draws[d] = np.concatenate([delta[s:s + BLOCK] for s in starts[d]])[:n].mean()
    return draws


def read_cell(client, prefix: str, band: str, variant: str, model: str, seed: int):
    try:
        key = f"{prefix}/{band}/{variant}/{model}/seed-{seed}/per_origin.npz"
        body = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        with np.load(io.BytesIO(body), allow_pickle=True) as npz:
            return {k: npz[k] for k in npz.files}
    except Exception:
        return None


def rows_for(cell, band, variant, model, seed, readout) -> list[dict]:
    horizons = [int(h) for h in cell["horizons"]]
    rows = []
    pooled = {"mae": [], "mse": []}
    for receiver in RECEIVERS:
        mae, mse = cell[f"{receiver}|mae"], cell.get(f"{receiver}|mse")
        n = len(mae)
        pooled["mae"].append(mae)
        if mse is not None:
            pooled["mse"].append(mse)
        for column, h in [(i, h) for i, h in enumerate(horizons)] + [(None, "mean")]:
            rows.append({"band": band, "variant": variant, "model": model, "seed": seed, "readout": readout, "receiver": receiver, "horizon": h,
                         "MAE": float(mae.mean() if column is None else mae[:, column].mean()),
                         "MSE": float((mse.mean() if column is None else mse[:, column].mean()) if mse is not None else np.nan), "n_origins": n})
    mae_all = np.mean(pooled["mae"], axis=0)
    mse_all = np.mean(pooled["mse"], axis=0) if pooled["mse"] else None
    for column, h in [(i, h) for i, h in enumerate(horizons)] + [(None, "mean")]:
        rows.append({"band": band, "variant": variant, "model": model, "seed": seed, "readout": readout, "receiver": "pooled", "horizon": h,
                     "MAE": float(mae_all.mean() if column is None else mae_all[:, column].mean()),
                     "MSE": float((mse_all.mean() if column is None else mse_all[:, column].mean()) if mse_all is not None else np.nan),
                     "n_origins": len(mae_all)})
    return rows


def seed_mean_series(cells: dict[int, dict], receiver: str, horizon) -> pd.Series | None:
    """Seed-averaged per-origin MAE (mean over horizons, or one horizon) at one receiver, indexed by origin time."""
    frames = []
    for cell in cells.values():
        horizons = [int(h) for h in cell["horizons"]]
        mae = cell[f"{receiver}|mae"]
        value = mae.mean(axis=1) if horizon == "mean" else mae[:, horizons.index(int(horizon))]
        frames.append(pd.Series(value, index=pd.to_datetime(pd.Series(cell[f"{receiver}|times"]), utc=True)))
    if not frames:
        return None
    return pd.concat(frames, axis=1).mean(axis=1, skipna=False).dropna()


def paired_rows(band, variant, model, as_cells, near_cells) -> list[dict]:
    seeds = sorted(set(as_cells) & set(near_cells))
    as_cells, near_cells = {s: as_cells[s] for s in seeds}, {s: near_cells[s] for s in seeds}
    out = []
    for horizon in (*HORIZONS, "mean"):
        draws_all, deltas, a_all, b_all, n_all = [], [], [], [], []
        for receiver in RECEIVERS:
            a, b = seed_mean_series(as_cells, receiver, horizon), seed_mean_series(near_cells, receiver, horizon)
            common = a.index.intersection(b.index)
            delta = (b.loc[common] - a.loc[common]).to_numpy()
            draws = block_bootstrap(delta)
            draws_all.append(draws)
            deltas.append(delta.mean())
            a_all.append(a.loc[common].mean())
            b_all.append(b.loc[common].mean())
            n_all.append(len(common))
            out.append({"band": band, "variant": variant, "model": model, "receiver": receiver, "horizon": horizon, "n_seeds": len(seeds),
                        "n_origins": len(common), "mae_as_evaluated": a_all[-1], "mae_nearest_cell": b_all[-1], "delta_nearest_minus_as_evaluated": deltas[-1],
                        "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5))})
        pooled = np.mean(draws_all, axis=0)
        out.append({"band": band, "variant": variant, "model": model, "receiver": "pooled", "horizon": horizon, "n_seeds": len(seeds),
                    "n_origins": int(np.mean(n_all)), "mae_as_evaluated": float(np.mean(a_all)), "mae_nearest_cell": float(np.mean(b_all)),
                    "delta_nearest_minus_as_evaluated": float(np.mean(deltas)), "ci_low": float(np.percentile(pooled, 2.5)),
                    "ci_high": float(np.percentile(pooled, 97.5))})
    return out


def ranking_text(paired: pd.DataFrame, model: str) -> list[str]:
    lines = []
    for band in BANDS:
        sub = paired[(paired.band == band) & (paired.model == model) & (paired.receiver == "pooled") & (paired.horizon == "mean")]
        if sub.empty:
            continue
        sub = sub.set_index("variant")
        rank_a = sub["mae_as_evaluated"].rank().astype(int)
        rank_n = sub["mae_nearest_cell"].rank().astype(int)
        lines.append(f"{band} ({model}; pooled over receivers, mean over t+4/12/24/48, seeds averaged per origin):")
        table = pd.DataFrame({"seeds": sub["n_seeds"], "MAE as_evaluated": sub["mae_as_evaluated"].round(3), "rank": rank_a,
                              "MAE nearest_cell": sub["mae_nearest_cell"].round(3), "rank ": rank_n,
                              "nearest - as_eval [95% CI]": [f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]" for d, lo, hi in
                                                             zip(sub["delta_nearest_minus_as_evaluated"], sub["ci_low"], sub["ci_high"])]})
        lines += [table.sort_values("MAE as_evaluated").to_string(), ""]
        panel = [v for v in ("core4-p1", "core4-p2", "core4-p4", "core4-p16", "core4-krige") if v in sub.index]
        order_a = sorted(panel, key=lambda v: sub.loc[v, "mae_as_evaluated"])
        order_n = sorted(panel, key=lambda v: sub.loc[v, "mae_nearest_cell"])
        lines.append(f"  panel (b) order by receiver MAE: as_evaluated {' < '.join(order_a)} | nearest_cell {' < '.join(order_n)} "
                     f"-> {'unchanged' if order_a == order_n else 'CHANGES'}")
        lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="evaluation/results/spatial15_nearest_readout")
    p.add_argument("--models", nargs="+", default=["linearar4d"])
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    long_rows, paired = [], []
    for band in BANDS:
        for model in a.models:
            for variant in VARIANTS:
                cells = {readout: {} for readout in PREFIXES}
                for readout, prefix in PREFIXES.items():
                    for seed in SEEDS:
                        cell = read_cell(client, prefix, band, variant, model, seed)
                        if cell is not None:
                            cells[readout][seed] = cell
                            long_rows += rows_for(cell, band, variant, model, seed, readout)
                if cells["as_evaluated"] and cells["nearest_cell"]:
                    paired += paired_rows(band, variant, model, cells["as_evaluated"], cells["nearest_cell"])
                print(f"[nearest] {band} {variant} {model}: seeds as_evaluated {sorted(cells['as_evaluated'])} nearest {sorted(cells['nearest_cell'])}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(long_rows)
    paired = pd.DataFrame(paired)
    long.to_csv(str(out) + ".csv", index=False)
    paired.to_csv(str(out) + "_paired.csv", index=False)
    text = ["Readout of the forecast map at the receivers: as_evaluated (IDW over all grid cells, the paper) vs nearest_cell (nearest grid node). "
            "MAE in dB, same origins and seeds for both.", ""]
    for model in a.models:
        text += ranking_text(paired, model)
    Path(str(out) + ".txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    if a.upload:
        for suffix in (".csv", "_paired.csv", ".txt"):
            path = Path(str(out) + suffix)
            client.put_object(Bucket=BUCKET, Key=f"analysis-outputs/{path.name}", Body=path.read_bytes())
            print("uploaded", path, path.stat().st_size)


if __name__ == "__main__":
    main()
