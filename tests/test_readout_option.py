"""The opt-in readout of the forecast map at the receivers: the default path is unchanged (it IS sample_forecast_grid)."""
from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common.spatial_checkpoint_evaluation import (  # noqa: E402
    build_idw_maps, derive_frozen_grid, read_forecast_maps, sample_forecast_grid)

LONS = np.array([-111.8465, -111.8400, -111.8325, -111.8440])
LATS = np.array([40.7585, 40.7681, 40.7701, 40.7648])


def _maps(power=2.0, t=6, f=4):
    grid = derive_frozen_grid(LONS, LATS, height=10, width=10, power=power)
    values = np.random.default_rng(0).normal(-100, 5, size=(t, 4, f)).astype(np.float32)
    return grid, build_idw_maps(values, LONS, LATS, grid), values


def test_default_readout_is_exactly_sample_forecast_grid():
    grid, maps, _ = _maps()
    ref = sample_forecast_grid(maps, grid, LONS, LATS)
    got = read_forecast_maps(maps, grid, LONS, LATS)
    explicit = read_forecast_maps(maps, grid, LONS, LATS, "as_evaluated")
    for a, b, c in zip(ref, got, explicit):
        assert np.array_equal(a, b) and np.array_equal(a, c)


def test_sharp_readout_is_power_16():
    grid, maps, _ = _maps(power=1.0)
    ref = sample_forecast_grid(maps, dataclasses.replace(grid, power=16.0), LONS, LATS)[0]
    assert np.array_equal(read_forecast_maps(maps, grid, LONS, LATS, "sharp")[0], ref)


def test_nearest_cell_returns_the_nearest_node_value_and_keeps_metadata():
    grid, maps, _ = _maps()
    sampled, inside, outside = read_forecast_maps(maps, grid, LONS, LATS, "nearest_cell")
    ref = sample_forecast_grid(maps, grid, LONS, LATS)
    assert sampled.shape == ref[0].shape and np.array_equal(inside, ref[1]) and np.array_equal(outside, ref[2])
    from training.common.map_builder import _local_xy

    x, y = _local_xy(LONS, LATS, grid.origin_longitude, grid.origin_latitude)
    for q in range(4):
        i, j = np.unravel_index(int(np.hypot(grid.x - x[q], grid.y - y[q]).argmin()), grid.shape)
        assert np.array_equal(sampled[..., q, :], maps[..., i, j, :])


def test_leading_dimensions_are_preserved():
    grid, maps, _ = _maps()
    stacked = np.stack([maps, maps], axis=1)            # (T, 2 horizons, H, W, F), as the forecast maps are
    assert read_forecast_maps(stacked, grid, LONS, LATS, "nearest_cell")[0].shape == (maps.shape[0], 2, 4, maps.shape[-1])


def test_unknown_readout_is_rejected():
    grid, maps, _ = _maps()
    with pytest.raises(ValueError):
        read_forecast_maps(maps, grid, LONS, LATS, "bilinear")


def test_evaluate_condition_default_is_as_evaluated():
    from training.common.spatial_ablation_eval import evaluate_condition

    assert inspect.signature(evaluate_condition).parameters["readout"].default == "as_evaluated"


def test_per_origin_errors_keep_mae_and_add_mse():
    from training.ray.spatial15_origins import per_origin_errors

    rng = np.random.default_rng(1)
    pred = rng.normal(size=(5, 4, 2, 3)).astype(np.float32)
    target = rng.normal(size=(5, 4, 2, 3)).astype(np.float32)
    npz = {"predictions_normalized": pred, "targets_dbm": target, "origin_timestamps": np.array([f"t{i}" for i in range(5)])}
    out = per_origin_errors(npz, 1, None)
    assert np.allclose(out["mae"], np.abs(pred[:, :, 1] - target[:, :, 1]).mean(axis=2))
    assert np.allclose(out["mse"], ((pred[:, :, 1] - target[:, :, 1]) ** 2).mean(axis=2))
