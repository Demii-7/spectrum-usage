#!/usr/bin/env python3
"""Model-free smoothing diagnostics of the eight ``s2`` map constructions of the 15-minute spatial section (paper 5.4, Reviewer 2).

No model is involved. For each band and map construction (core4-p1/p2/p4/p16, core4-p2-linear, core4-p2-g5, core4-p2-g20, core4-krige), on the
four core receivers' 15-minute series (test week 2026-10-27T20:00Z to 2026-11-03T20:00Z; the training run 20261015T1434Z for the variance statistics):

* ``roundtrip_mae`` / ``roundtrip_mse`` (dB, dB^2): build the map from the receivers' TRUE values at every step and read it back at the four receivers.
  The error a perfect map forecaster would still have at the receivers. Readouts: ``as_evaluated`` (``sample_forecast_grid`` with the grid's own power,
  what the paper's numbers use), ``nearest_cell`` (the grid node nearest to the receiver, ``_nearest_grid_indices``), ``sharp`` (``sample_forecast_grid``
  with power 16 regardless of the map's power).
* ``own_weight``: the share of the as-evaluated round-trip value that comes from the receiver's own measurement (map weights composed with the readout
  weights; both depend only on coordinates). 1: the readout returns the receiver's own value, 0.25: a flat mean of the four. Obtained by pushing unit
  impulses through ``build_idw_maps`` and ``sample_forecast_grid`` (no reimplementation). For ``core4-p2-linear`` the map mixes LINEAR power, so the
  value is the share of the weights (as if in dB), not of the dB value.
* ``variance_retained_<period>``: per bin, temporal variance of each map cell divided by the mean temporal variance of the four receivers, averaged over
  bins; readout ``all_cells`` (mean over cells) or ``nearest_cells`` (the four cells nearest to the receivers). Periods: ``train``, ``test``.
* ``diff_ratio_<period>``: mean absolute 15-minute change of the map cells over that of the receivers (``all_cells`` / ``nearest_cells``).
* ``persistence_map_mae_t+h`` / ``persistence_receiver_mae_t+h`` (and MSE): parameter-free forecast ``x_hat(t+h) = x(t)`` at h = 4, 12, 24, 48 steps
  (1, 3, 6, 12 h): on the map (all cells against the map's own future) and at the receivers (the as-evaluated read-out of the map at t against the TRUE
  future receiver value); ``persistence_raw_mae_t+h`` is the receivers' own persistence (no map), the reference.

Output (long format): band, variant, statistic, readout, receiver, value, n (receiver ``all`` = pooled over the four receivers / all cells).

    python evaluation/analysis/smoothing_diagnostics.py --out evaluation/results/smoothing_diagnostics [--upload]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common.spatial_ablation_eval import _coordinates, _nearest_grid_indices  # noqa: E402
from training.common.spatial_checkpoint_evaluation import (  # noqa: E402
    FrozenGrid, build_idw_maps, derive_frozen_grid, load_aligned_receiver_csvs, sample_forecast_grid)
from training.ray import spatial15  # noqa: E402

BANDS = ("600_800", "2400_2600")
TRAIN_RUN, TEST_RUN = "20261015T1434Z", "20261025T1355Z"
TEST_START, TEST_END = "2026-10-27T20:00:00Z", "2026-11-03T20:00:00Z"
READOUTS = ("as_evaluated", "nearest_cell", "sharp")
SHARP_POWER = 16.0
HORIZONS = (4, 12, 24, 48)
STEP = pd.Timedelta(minutes=15)
MAX_MISSING_GAP = 16                      # as in spatial15.experiment_yaml
LOOKBACK_MEAN_RECEIVER_MAE = {"600_800": 1.215, "2400_2600": 1.216}      # evaluation/results/spatial15_s2.csv (4D lookback mean at the receivers)
NAMES = tuple(spatial15.SIX[node] for node in spatial15.CORE)            # humanities, madsen, moran, sage point


# ------------------------------------------------------------------------------------------------ loading
def load_series(root: Path, band: str, run: str, start: str | None = None, end: str | None = None):
    """(values (T, R, F) float32, timestamps, frequencies, longitudes, latitudes) of the four core receivers, aligned and bounded-imputed as in the evaluation."""
    from training.common.data_sources import _read_csv

    paths = [root / "data/powder15" / node / run / band / "power_15min.csv" for node in spatial15.CORE]
    frequencies = [float(c) for c in _read_csv(paths[0])[0].columns]
    coordinates = _coordinates(root / "data/locations/powder.json", "endpoints", list(NAMES))
    series = load_aligned_receiver_csvs(paths, frequencies, coordinates, receiver_names=NAMES, max_missing_gap=MAX_MISSING_GAP)
    stamps = series.timestamps
    keep = np.ones(len(stamps), dtype=bool)
    if start is not None:
        keep &= stamps >= pd.Timestamp(start)
    if end is not None:
        keep &= stamps <= pd.Timestamp(end)
    return series.values_dbm[keep], stamps[keep], series.frequencies, series.longitudes, series.latitudes


def grid_for_variant(variant: dict[str, Any], frame: dict[str, float], band: str, root: Path, lons, lats) -> FrozenGrid:
    """The frozen grid of one ``s2`` variant exactly as the evaluation builds it (``grid_for`` + the kriging variogram of the band)."""
    cfg = spatial15.grid_for(variant, frame)
    if cfg.get("method") == "kriging":
        cfg = {**cfg, "variogram": spatial15.load_variogram(str(root), band)}
    return derive_frozen_grid(
        lons, lats, height=int(cfg.get("height", 10)), width=int(cfg.get("width", 10)), power=float(cfg.get("power", 2.0)),
        width_meters=cfg.get("width_meters"), height_meters=cfg.get("height_meters"), origin_lon=cfg.get("origin_lon"),
        origin_lat=cfg.get("origin_lat"), power_domain=str(cfg.get("power_domain", "db")), method=str(cfg.get("method", "idw")),
        variogram=cfg.get("variogram"))


# ------------------------------------------------------------------------------------------------ readouts and weights
def read_maps(maps: np.ndarray, grid: FrozenGrid, lons, lats, readout: str) -> np.ndarray:
    """Maps (..., H, W, F) -> values at the receivers (..., R, F) under one readout."""
    if readout == "as_evaluated":
        return sample_forecast_grid(maps, grid, lons, lats)[0]
    if readout == "sharp":
        return sample_forecast_grid(maps, dataclasses.replace(grid, power=SHARP_POWER), lons, lats)[0]
    if readout == "nearest_cell":
        index = _nearest_grid_indices(grid, lons, lats, grid.origin_longitude, grid.origin_latitude)
        return np.stack([maps[..., i, j, :] for i, j in index], axis=-2)
    raise ValueError(readout)


def composite_weights(grid: FrozenGrid, lons, lats) -> np.ndarray:
    """(R, R): entry [k, q] = weight of receiver k in the as-evaluated round-trip value at receiver q (unit impulses through the real code)."""
    r = len(lons)
    weight_grid = dataclasses.replace(grid, power_domain="db") if grid.power_domain != "db" else grid     # weights are domain-agnostic; see module doc
    impulses = np.eye(r, dtype=np.float32)[:, :, None]                                                  # (impulse k, receiver, 1 bin)
    maps = build_idw_maps(impulses, lons, lats, weight_grid)
    return sample_forecast_grid(maps, weight_grid, lons, lats)[0][:, :, 0].astype(np.float64)


def nearest_cells(grid: FrozenGrid, lons, lats) -> list[tuple[int, int]]:
    return _nearest_grid_indices(grid, lons, lats, grid.origin_longitude, grid.origin_latitude)


# ------------------------------------------------------------------------------------------------ statistics
def roundtrip(values: np.ndarray, maps: np.ndarray, grid: FrozenGrid, lons, lats) -> dict[str, dict[str, np.ndarray]]:
    """{readout: {'mae': (R,), 'mse': (R,)}} over all steps and bins, per receiver."""
    out = {}
    for readout in READOUTS:
        error = read_maps(maps, grid, lons, lats, readout).astype(np.float64) - values.astype(np.float64)
        out[readout] = {"mae": np.nanmean(np.abs(error), axis=(0, 2)), "mse": np.nanmean(error ** 2, axis=(0, 2))}
    return out


def variance_retained(values: np.ndarray, maps: np.ndarray, cells: list[tuple[int, int]]) -> tuple[float, float]:
    """Mean over bins of (cell temporal variance / mean receiver temporal variance): over all cells, over the four nearest-node cells."""
    with np.errstate(all="ignore"):
        receiver_var = np.nanvar(values.astype(np.float64), axis=0).mean(axis=0)           # (F,)
        cell_var = np.nanvar(maps.astype(np.float64), axis=0)                              # (H, W, F)
        ratio = cell_var / receiver_var
        nearest = np.stack([ratio[i, j] for i, j in cells])                                # (R, F)
    return float(np.nanmean(ratio)), float(np.nanmean(nearest))


def diff_ratio(values: np.ndarray, maps: np.ndarray, cells: list[tuple[int, int]], stamps: pd.DatetimeIndex) -> tuple[float, float]:
    """Mean absolute 15-minute change of the map cells over that of the receivers (consecutive steps only)."""
    ok = np.flatnonzero(np.diff(stamps.asi8) == STEP.value)
    with np.errstate(all="ignore"):
        receivers = np.nanmean(np.abs(values[ok + 1].astype(np.float64) - values[ok].astype(np.float64)))
        change = np.abs(maps[ok + 1].astype(np.float64) - maps[ok].astype(np.float64))     # (T', H, W, F)
        allcells = np.nanmean(change)
        nearest = np.nanmean(np.stack([change[:, i, j, :] for i, j in cells]))
    return float(allcells / receivers), float(nearest / receivers)


def origin_pairs(stamps: pd.DatetimeIndex, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(t, t + horizon): indices of the steps t for which a step exactly ``horizon`` x 15 minutes later exists; the same set for every construction."""
    target = stamps + horizon * STEP
    position = stamps.get_indexer(target)
    return np.flatnonzero(position >= 0), position[position >= 0]


def persistence(values: np.ndarray, maps: np.ndarray, grid: FrozenGrid, lons, lats, stamps: pd.DatetimeIndex) -> dict[str, Any]:
    """Persistence errors on the map, at the receivers via the as-evaluated read-out, and raw at the receivers; per horizon."""
    out = {}
    read = read_maps(maps, grid, lons, lats, "as_evaluated").astype(np.float64)
    values64, maps64 = values.astype(np.float64), maps.astype(np.float64)
    for h in HORIZONS:
        t, f = origin_pairs(stamps, h)
        with np.errstate(all="ignore"):
            map_err = maps64[t] - maps64[f]
            rec_err = read[t] - values64[f]
            raw_err = values64[t] - values64[f]
        out[h] = {"n_origins": len(t),
                  "map_mae": float(np.nanmean(np.abs(map_err))), "map_mse": float(np.nanmean(map_err ** 2)),
                  "receiver_mae": np.nanmean(np.abs(rec_err), axis=(0, 2)), "receiver_mse": np.nanmean(rec_err ** 2, axis=(0, 2)),
                  "raw_mae": np.nanmean(np.abs(raw_err), axis=(0, 2)), "raw_mse": np.nanmean(raw_err ** 2, axis=(0, 2))}
    return out


# ------------------------------------------------------------------------------------------------ driver
def diagnostics_for_band(root: Path, band: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test, test_stamps, _, lons, lats = load_series(root, band, TEST_RUN, TEST_START, TEST_END)
    train, train_stamps, _, _, _ = load_series(root, band, TRAIN_RUN)
    finite = np.isfinite(test).all(axis=(1, 2))
    frame = spatial15.six_receiver_frame(str(root))
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"n_steps_test": int(len(test_stamps)), "n_steps_test_all_finite": int(finite.sum()),
                            "n_steps_train": int(len(train_stamps)), "bins": int(test.shape[2])}

    def add(variant, statistic, readout, receiver, value, n):
        rows.append({"band": band, "variant": variant, "statistic": statistic, "readout": readout, "receiver": receiver,
                     "value": float(value), "n": int(n)})

    for variant in spatial15.variants(["s2"]):
        name = variant["name"]
        grid = grid_for_variant(variant, frame, band, root, lons, lats)
        cells = nearest_cells(grid, lons, lats)
        meta.setdefault("grid_power", {})[name] = (grid.power, grid.power_domain, grid.method)
        test_maps = build_idw_maps(test, lons, lats, grid)
        train_maps = build_idw_maps(train, lons, lats, grid)
        test_maps[~np.isfinite(test).all(axis=(1, 2))] = np.nan          # a step with any missing receiver value is dropped from every statistic
        train_maps[~np.isfinite(train).all(axis=(1, 2))] = np.nan
        n_values = int(np.isfinite(test).sum() // test.shape[1])
        trip = roundtrip(test, test_maps, grid, lons, lats)
        for readout, stats in trip.items():
            for stat in ("mae", "mse"):
                for k, receiver in enumerate(NAMES):
                    add(name, f"roundtrip_{stat}", readout, receiver, stats[stat][k], n_values)
                add(name, f"roundtrip_{stat}", readout, "all", stats[stat].mean(), n_values * len(NAMES))
        composite = composite_weights(grid, lons, lats)
        for k, receiver in enumerate(NAMES):
            add(name, "own_weight", "as_evaluated", receiver, composite[k, k], 1)
        add(name, "own_weight", "as_evaluated", "all", np.mean(np.diag(composite)), len(NAMES))
        for period, values, maps, stamps in (("train", train, train_maps, train_stamps), ("test", test, test_maps, test_stamps)):
            v_all, v_near = variance_retained(values, maps, cells)
            d_all, d_near = diff_ratio(values, maps, cells, stamps)
            for stat, (a, b) in ((f"variance_retained_{period}", (v_all, v_near)), (f"diff_ratio_{period}", (d_all, d_near))):
                add(name, stat, "all_cells", "all", a, int(maps.shape[0]))
                add(name, stat, "nearest_cells", "all", b, int(maps.shape[0]))
        for h, stats in persistence(test, test_maps, grid, lons, lats, test_stamps).items():
            add(name, f"persistence_map_mae_t+{h}", "map", "all", stats["map_mae"], stats["n_origins"])
            add(name, f"persistence_map_mse_t+{h}", "map", "all", stats["map_mse"], stats["n_origins"])
            for kind in ("receiver", "raw"):
                for stat in ("mae", "mse"):
                    for k, receiver in enumerate(NAMES):
                        add(name, f"persistence_{kind}_{stat}_t+{h}", "as_evaluated" if kind == "receiver" else "none", receiver,
                            stats[f"{kind}_{stat}"][k], stats["n_origins"])
                    add(name, f"persistence_{kind}_{stat}_t+{h}", "as_evaluated" if kind == "receiver" else "none", "all",
                        stats[f"{kind}_{stat}"].mean(), stats["n_origins"])
            meta.setdefault("n_origins", {})[h] = stats["n_origins"]
    return rows, meta


def checks(table: pd.DataFrame, meta: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    for band in BANDS:
        sub = table[(table["band"] == band) & (table["receiver"] == "all")]

        def val(variant, statistic, readout):
            hit = sub[(sub["variant"] == variant) & (sub["statistic"] == statistic) & (sub["readout"] == readout)]
            return float(hit["value"].iloc[0]) if len(hit) else float("nan")

        p2 = val("core4-p2", "roundtrip_mae", "as_evaluated")
        lines.append(f"[{band}] core4-p2 as_evaluated round-trip MAE {p2:.3f} dB vs 4D lookback mean at the receivers {LOOKBACK_MEAN_RECEIVER_MAE[band]:.3f} dB: "
                     f"{'clearly below' if p2 < 0.8 * LOOKBACK_MEAN_RECEIVER_MAE[band] else 'NOT clearly below, investigate'}")
        krige = val("core4-krige", "roundtrip_mae", "as_evaluated")
        lines.append(f"[{band}] core4-krige as_evaluated round-trip MAE {krige:.3f} dB (600-800: every model scores 2.4-2.5 dB on this map)")
        for variant in ("core4-p2", "core4-p16"):
            near, sharp = val(variant, "roundtrip_mae", "nearest_cell"), val(variant, "roundtrip_mae", "sharp")
            lines.append(f"[{band}] {variant}: nearest_cell {near:.3f} vs sharp (power 16) {sharp:.3f} dB, difference {near - sharp:+.3f}")
        asev = val("core4-p2", "roundtrip_mae", "as_evaluated")
        near = val("core4-p2", "roundtrip_mae", "nearest_cell")
        lines.append(f"[{band}] READOUT EFFECT on the default map core4-p2: as_evaluated - nearest_cell = {asev - near:+.3f} dB"
                     + ("  (> 0.05 dB: the paper's receiver errors include a readout penalty)" if asev - near > 0.05 else ""))
        lines.append(f"[{band}] time steps: test week {meta[band]['n_steps_test']} ({meta[band]['n_steps_test_all_finite']} with all bins finite), "
                     f"training run {meta[band]['n_steps_train']}; persistence origins per horizon {meta[band]['n_origins']}; same for every construction")
    return lines


def text_report(table: pd.DataFrame, lines: list[str]) -> str:
    out = ["Smoothing diagnostics of the s2 map constructions (15 minutes; four core receivers; model-free). MAE in dB, pooled over receivers and bins.", ""]
    for band in BANDS:
        sub = table[(table["band"] == band) & (table["receiver"] == "all")]
        variants = list(dict.fromkeys(sub["variant"]))

        def col(statistic, readout):
            s = sub[(sub["statistic"] == statistic) & (sub["readout"] == readout)].set_index("variant")["value"]
            return [s.get(v, np.nan) for v in variants]

        frame = pd.DataFrame({
            "rt_asEval": col("roundtrip_mae", "as_evaluated"), "rt_nearest": col("roundtrip_mae", "nearest_cell"), "rt_sharp": col("roundtrip_mae", "sharp"),
            "own_w": col("own_weight", "as_evaluated"),
            "var_all": col("variance_retained_test", "all_cells"), "var_near": col("variance_retained_test", "nearest_cells"),
            "dRatio_all": col("diff_ratio_test", "all_cells"), "dRatio_near": col("diff_ratio_test", "nearest_cells"),
            "persMap_h4": col("persistence_map_mae_t+4", "map"), "persMap_h48": col("persistence_map_mae_t+48", "map"),
            "persRec_h4": col("persistence_receiver_mae_t+4", "as_evaluated"), "persRec_h48": col("persistence_receiver_mae_t+48", "as_evaluated"),
            "persRaw_h4": col("persistence_raw_mae_t+4", "none"), "persRaw_h48": col("persistence_raw_mae_t+48", "none"),
        }, index=variants)
        out += [f"== {band}", frame.round(3).to_string(), ""]
        out.append("Proposed panel (b) columns (round-trip MAE as evaluated | variance retained, all cells, test week):")
        for variant in ("core4-p1", "core4-p2", "core4-p4", "core4-p16", "core4-krige"):
            if variant in frame.index:
                out.append(f"  {variant:12s} {frame.loc[variant, 'rt_asEval']:.3f} dB | {frame.loc[variant, 'var_all']:.3f}")
        out.append("")
    return "\n".join(out + ["Checks:"] + lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="evaluation/results/smoothing_diagnostics")
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    root = Path(a.root).resolve()
    rows, metas = [], {}
    for band in BANDS:
        band_rows, metas[band] = diagnostics_for_band(root, band)
        rows += band_rows
        print(f"[smoothing] {band}: {len(band_rows)} rows, {metas[band]}", flush=True)
    table = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(str(a.out) + ".csv", index=False)
    lines = checks(table, metas)
    report = text_report(table, lines)
    Path(str(a.out) + ".txt").write_text(report)
    print(report)
    if a.upload:
        from training.ray.storage import spectrum_s3_client

        client = spectrum_s3_client()
        for suffix in (".csv", ".txt"):
            path = Path(str(a.out) + suffix)
            client.put_object(Bucket="spectrum", Key=f"analysis-outputs/{path.name}", Body=path.read_bytes())
            print("uploaded", path, path.stat().st_size)


if __name__ == "__main__":
    main()
