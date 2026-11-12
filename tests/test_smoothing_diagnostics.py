"""Unit tests of the model-free smoothing diagnostics (synthetic receivers; no cluster data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "analysis"))

import smoothing_diagnostics as sd  # noqa: E402
from training.common.spatial_checkpoint_evaluation import build_idw_maps, derive_frozen_grid  # noqa: E402

LON0, LAT0 = -111.84, 40.76
# four receivers at the corners of a rectangle: with a 2x2 grid the grid nodes ARE the receivers
CORNER_LONS = np.array([LON0 - 0.004, LON0 + 0.004, LON0 - 0.004, LON0 + 0.004])
CORNER_LATS = np.array([LAT0 - 0.003, LAT0 - 0.003, LAT0 + 0.003, LAT0 + 0.003])
# four scattered receivers for a 10x10 grid
LONS = np.array([-111.8465, -111.8400, -111.8325, -111.8440])
LATS = np.array([40.7585, 40.7681, 40.7701, 40.7648])


def _values(t=40, r=4, f=5, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(size=(t, 1, f)), axis=0)
    return (-100.0 + base + rng.normal(scale=2.0, size=(t, r, f))).astype(np.float32)


def test_roundtrip_is_exactly_zero_when_receivers_sit_on_nodes():
    grid = derive_frozen_grid(CORNER_LONS, CORNER_LATS, height=2, width=2, power=2.0)
    values = _values()
    maps = build_idw_maps(values, CORNER_LONS, CORNER_LATS, grid)
    for readout in ("sharp", "nearest_cell", "as_evaluated"):
        read = sd.read_maps(maps, grid, CORNER_LONS, CORNER_LATS, readout)
        assert np.allclose(read, values, atol=1e-4), readout
    trip = sd.roundtrip(values, maps, grid, CORNER_LONS, CORNER_LATS)
    assert trip["sharp"]["mae"].max() < 1e-4 and trip["nearest_cell"]["mae"].max() < 1e-4


def test_own_weight_is_one_on_nodes_and_rows_sum_to_one():
    grid = derive_frozen_grid(CORNER_LONS, CORNER_LATS, height=2, width=2, power=2.0)
    composite = sd.composite_weights(grid, CORNER_LONS, CORNER_LATS)
    assert np.allclose(np.diag(composite), 1.0, atol=1e-5)
    scattered = derive_frozen_grid(LONS, LATS, height=10, width=10, power=2.0)
    composite = sd.composite_weights(scattered, LONS, LATS)
    assert np.allclose(composite.sum(axis=0), 1.0, atol=1e-4)                       # every receiver's value is a convex combination of the four
    assert (np.diag(composite) > 0.25).all() and (np.diag(composite) < 1.0).all()


def test_own_weight_grows_with_the_idw_power():
    own = []
    for power in (1.0, 2.0, 4.0, 16.0):
        grid = derive_frozen_grid(LONS, LATS, height=10, width=10, power=power)
        own.append(np.diag(sd.composite_weights(grid, LONS, LATS)).mean())
    assert own == sorted(own) and own[0] < own[-1]


def test_constant_in_space_field_keeps_all_variance_and_unit_diff_ratio():
    t = 30
    common = np.cumsum(np.random.default_rng(1).normal(size=(t, 1, 3)), axis=0).astype(np.float32)
    values = np.repeat(common, 4, axis=1)                                          # the same series at every receiver
    grid = derive_frozen_grid(LONS, LATS, height=10, width=10, power=2.0)
    maps = build_idw_maps(values, LONS, LATS, grid)
    cells = sd.nearest_cells(grid, LONS, LATS)
    v_all, v_near = sd.variance_retained(values, maps, cells)
    stamps = pd.date_range("2026-10-27T20:00:00Z", periods=t, freq="15min")
    d_all, d_near = sd.diff_ratio(values, maps, cells, stamps)
    assert abs(v_all - 1.0) < 1e-3 and abs(v_near - 1.0) < 1e-3
    assert abs(d_all - 1.0) < 1e-3 and abs(d_near - 1.0) < 1e-3


def test_smoothing_reduces_variance_of_independent_receivers():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(200, 4, 3)).astype(np.float32)                        # independent series: averaging must shrink the variance
    grid = derive_frozen_grid(LONS, LATS, height=10, width=10, power=1.0)
    maps = build_idw_maps(values, LONS, LATS, grid)
    v_all, v_near = sd.variance_retained(values, maps, sd.nearest_cells(grid, LONS, LATS))
    assert v_all < 0.8


def test_origin_pairs_require_the_exact_future_step():
    stamps = pd.DatetimeIndex(pd.date_range("2026-10-27T20:00:00Z", periods=20, freq="15min")).delete([10])      # one missing step
    t, f = sd.origin_pairs(stamps, 4)
    assert all(stamps[b] - stamps[a] == 4 * sd.STEP for a, b in zip(t, f))
    assert len(t) == 14      # 16 possible origins of 20 steps, minus the origin that is the gap and the one whose t+4 steps falls on the gap
