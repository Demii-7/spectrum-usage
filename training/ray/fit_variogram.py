"""Fit the frozen kriging variogram of each band from the TRAINING-range receiver series (no test data), once, before any scoring.

For each band (600_800, 2400_2600) the four core receivers' 15-minute training run (2026-10-15 14:34 to 10-22 13:13, the files under
``data/powder15/<node>/20261015T1434Z/<band>/``) is aligned on common timestamps; the empirical semivariogram
``gamma_ij = 0.5 * mean((z_i - z_j)^2)`` is pooled over time and the 200 one-MHz bins, and an exponential model with a nugget is fitted by
relative least squares (``training.common.kriging.fit_exponential_variogram``).

Which series: the fit is made on the NORMALIZED series (each receiver's per-bin training mean removed and divided by its per-bin training
standard deviation, i.e. gamma = 1 - correlation of the receivers' time fluctuations), because the raw-dB semivariogram of four receivers
is dominated by receiver-specific level offsets that do not follow distance (first fit, recorded under ``raw_db``: 600-800 gamma between
13 and 31 dB^2 at nearly equal distances; 2400-2600 gamma between 1 and 53 dB^2, so the best raw fit is a pure nugget with sill 0 and
would give every grid cell the plain average of the receivers). The weights from the normalized variogram are applied to the raw dB
values, as with IDW. This choice was made before any scoring and uses no test data.

Four receivers give only 6 pair distances for 3 parameters, so the script also bootstraps the fit over time blocks (block = 96 steps =
one day, 200 draws) and reports how much the kriging weights on the 10x10 grid move between draws. It also reports the kriging prediction
variance of the core-four map on the six-receiver frame grid (in units of the normalized variance, so a variance of 1 is "no better than
the mean").

Output: JSON (``--out``, also uploaded to ``kriging/variograms.json`` in the bucket) that is then committed as
``evaluation/analysis/kriging_variograms.json``; ``training.ray.spatial15`` reads it for the ``core4-krige`` variant.

    python -m training.ray.fit_variogram --out /tmp/kriging_variograms.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK, DRAWS = 96, 200


def load_core_series(repo_root: str, band: str):
    """(time, 4 receivers, bins) dB of the core receivers' training run, plus their coordinates and names."""
    from training.ray.powder15 import CORE_NODES, OUT_NAME, OUT_ROOT, TRAIN_RUN, ensure_powder15
    from training.ray.spatial15 import SIX

    ensure_powder15(band, repo_root, CORE_NODES)
    frames = []
    for node in CORE_NODES:
        frame = pd.read_csv(Path(repo_root) / OUT_ROOT / node / TRAIN_RUN / band / OUT_NAME)
        frame.index = pd.to_datetime(frame.pop("timestamp_utc"), utc=True)
        frames.append(frame)
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    common = common.sort_values()
    values = np.stack([frame.loc[common].to_numpy(dtype=np.float64) for frame in frames], axis=1)
    entries = {e["name"]: e for e in json.loads((Path(repo_root) / "data/locations/powder.json").read_text())["endpoints"]}
    names = [SIX[node] for node in CORE_NODES]
    lons = np.asarray([entries[n]["longitude"] for n in names])
    lats = np.asarray([entries[n]["latitude"] for n in names])
    return values, names, lons, lats, (str(common[0]), str(common[-1]))


def grid_points(repo_root: str, grid: dict):
    from training.common.map_builder import _local_xy
    from training.ray.spatial15 import GRID_DEFAULT, six_receiver_frame

    frame = six_receiver_frame(repo_root)
    cfg = {**GRID_DEFAULT, **frame, **grid}
    gx, gy = np.meshgrid(np.linspace(-cfg["width_meters"] / 2, cfg["width_meters"] / 2, int(cfg["width"])),
                         np.linspace(-cfg["height_meters"] / 2, cfg["height_meters"] / 2, int(cfg["height"])))
    return cfg, gx, gy, _local_xy


def fit_band(repo_root: str, band: str, rng: np.random.Generator) -> dict:
    from training.common.kriging import empirical_semivariogram, fit_exponential_variogram, kriging_weights

    raw_values, names, lons, lats, span = load_core_series(repo_root, band)
    cfg, gx, gy, local_xy = grid_points(repo_root, {})
    sx, sy = local_xy(lons, lats, cfg["origin_lon"], cfg["origin_lat"])
    raw_distances, raw_gammas = empirical_semivariogram(raw_values, sx, sy)
    try:
        raw_fit = fit_exponential_variogram(raw_distances, raw_gammas)
    except Exception as error:              # a degenerate raw fit is reported, not fatal
        raw_fit = {"error": str(error)}
    mean, std = raw_values.mean(axis=0, keepdims=True), raw_values.std(axis=0, keepdims=True)
    values = (raw_values - mean) / np.maximum(std, 1e-9)               # normalized per receiver and bin (training statistics)
    distances, gammas = empirical_semivariogram(values, sx, sy)
    fit = fit_exponential_variogram(distances, gammas)
    variogram = {k: float(fit[k]) for k in ("nugget", "sill", "range")} | {"model": "exponential"}
    weights, variance = kriging_weights(sx, sy, gx, gy, variogram, return_variance=True)
    n = values.shape[0]
    draws, weight_shift = [], []
    starts_per_block = int(np.ceil(n / BLOCK))
    for _ in range(DRAWS):
        starts = rng.integers(0, max(n - BLOCK, 1), size=starts_per_block)
        index = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in starts])[:n]
        d, g = empirical_semivariogram(values[index], sx, sy)
        try:
            b = fit_exponential_variogram(d, g)
        except Exception:
            continue
        draws.append([b["nugget"], b["sill"], b["range"]])
        wb = kriging_weights(sx, sy, gx, gy, {"model": "exponential", "nugget": b["nugget"], "sill": b["sill"], "range": b["range"]})
        weight_shift.append(float(np.abs(wb - weights).max(axis=-1).mean()))        # mean over cells of the largest weight change
    draws = np.asarray(draws)
    pct = lambda col: [float(v) for v in np.percentile(draws[:, col], [2.5, 50, 97.5])]
    total = variogram["nugget"] + variogram["sill"]
    return {
        "basis": "normalized (per receiver and bin, training mean and standard deviation); weights applied to raw dB",
        "raw_db": {"empirical_distance_m": [float(v) for v in raw_distances], "empirical_semivariance_db2": [float(v) for v in raw_gammas],
                   "exponential_fit": raw_fit},
        "variogram": variogram,
        "fit_relative_rss": fit["relative_rss"],
        "n_pairs": int(len(distances)), "n_parameters": 3,
        "receivers": names, "training_span": span, "n_time_steps": int(values.shape[0]), "n_bins": int(values.shape[2]),
        "empirical": {"distance_m": [float(v) for v in distances], "semivariance_db2": [float(v) for v in gammas],
                      "fitted_db2": [float(v) for v in (variogram["nugget"] + variogram["sill"] * (1 - np.exp(-distances / variogram["range"])))]},
        "bootstrap": {"draws": int(len(draws)), "block": BLOCK, "nugget_p2.5_p50_p97.5": pct(0), "sill_p2.5_p50_p97.5": pct(1),
                      "range_p2.5_p50_p97.5": pct(2),
                      "mean_max_weight_change_p50_p95": [float(np.percentile(weight_shift, 50)), float(np.percentile(weight_shift, 95))]},
        "kriging_variance_core4_grid": {"unit": "dB^2", "total_variance_nugget_plus_sill": total, "mean": float(variance.mean()),
                                        "min": float(variance.min()), "max": float(variance.max()),
                                        "share_of_cells_above_half_total": float((variance > 0.5 * total).mean()),
                                        "cells": [[float(v) for v in row] for row in variance]},
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
    p.add_argument("--out", default="evaluation/analysis/kriging_variograms.json")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    p.add_argument("--upload-key", default="kriging/variograms.json", help="bucket key to upload to ('' to skip)")
    args = p.parse_args(argv)
    os.chdir(args.repo_root)
    sys.path.insert(0, args.repo_root)
    rng = np.random.default_rng(0)
    result = {"model": "exponential with nugget, gamma(0) = 0; relative least squares; raw dB pooled over time and bins; training range only",
              "bands": {band: fit_band(args.repo_root, band, rng) for band in args.bands}}
    text = json.dumps(result, indent=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text, flush=True)
    if args.upload_key:
        from training.ray.storage import spectrum_s3_client

        spectrum_s3_client().put_object(Bucket=os.environ.get("SPECTRUM_S3_BUCKET", "spectrum"), Key=args.upload_key, Body=text.encode())
        print(f"[fit_variogram] uploaded {args.upload_key}", flush=True)


if __name__ == "__main__":
    main()
