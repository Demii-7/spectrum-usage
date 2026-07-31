import numpy as np
import pandas as pd
import pytest

from training.common.spatial_checkpoint_evaluation import (
    ReceiverSeries, aggregate_region_metrics, build_idw_maps, derive_frozen_grid,
    normalize_with_checkpoint, sample_forecast_grid, select_window_origins,
)
from training.common.spatial_ablation_eval import (
    _finite_origin_mask,
    _map_metric_rows,
)


COORDINATES = (np.array([-73.0, -72.999]), np.array([40.0, 40.001]))


def test_identity_permutation_is_exact_and_non_bijection_is_rejected():
    grid = derive_frozen_grid(*COORDINATES)
    values = np.array([[[1.0], [9.0]]], dtype=np.float32)
    expected = build_idw_maps(values, *COORDINATES, grid)
    actual = build_idw_maps(values, *COORDINATES, grid, coordinate_permutation=[0, 1])
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="bijection"):
        build_idw_maps(values, *COORDINATES, grid, coordinate_permutation=[0, 0])


def test_subset_uses_fixed_baseline_grid():
    grid = derive_frozen_grid(*COORDINATES)
    subset = build_idw_maps(np.array([[[3.0]]]), COORDINATES[0][:1], COORDINATES[1][:1], grid)
    assert grid.shape == (10, 10)
    assert subset.shape == (1, 10, 10, 1)
    np.testing.assert_allclose(subset, 3.0)


def test_sampling_extrapolates_and_reports_distance_without_clamping():
    grid = derive_frozen_grid(*COORDINATES)
    maps = grid.x[..., None].astype(np.float32)
    outside_lon = grid.origin_longitude + 0.01
    sampled, inside, distance = sample_forecast_grid(maps, grid, [grid.origin_longitude, outside_lon],
                                                      [grid.origin_latitude] * 2)
    assert inside.tolist() == [True, False]
    assert distance[0] == 0 and distance[1] > 0
    assert sampled.shape == (2, 1)
    assert sampled[1, 0] != maps[0, -1, 0]


def test_checkpoint_normalization_does_not_change_targets():
    raw = np.array([[[1.0, 12.0]]], dtype=np.float32)
    target = raw.copy()
    normalized = normalize_with_checkpoint(raw, {"mean_dbm": [1.0, 10.0], "std_dbm": [2.0, 2.0]})
    np.testing.assert_allclose(normalized, [[[0.0, 1.0]]])
    np.testing.assert_array_equal(raw, target)


def test_window_selection_requires_exact_minute_continuity_and_keeps_targets_raw():
    timestamps = pd.DatetimeIndex(pd.to_datetime([
        "2024-01-01 00:00Z", "2024-01-01 00:01Z", "2024-01-01 00:02Z",
        "2024-01-01 00:04Z", "2024-01-01 00:05Z", "2024-01-01 00:06Z",
    ]))
    raw = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    series = ReceiverSeries(raw, timestamps, np.array([100.0]), ("r",), np.array([0.0]), np.array([0.0]))
    origins, inputs, targets = select_window_origins(series, 2, [1], model_values=raw + 100)
    np.testing.assert_array_equal(origins, [1, 4])
    np.testing.assert_array_equal(inputs[:, -1, 0, 0], [101, 104])
    np.testing.assert_array_equal(targets[:, 0, 0, 0], [2, 5])


def test_region_metrics_denormalize_predictions_and_flag_noise_floor():
    predictions = np.array([[[0.0, 1.0, 2.0]]], dtype=np.float32)
    targets = np.array([[[10.0, 14.0, 16.0]]], dtype=np.float32)
    rows = aggregate_region_metrics(predictions, targets, [100, 200, 300], [
        {"region_id": "signal", "start_mhz": 100, "end_mhz": 200},
        {"region_id": "floor", "start_mhz": 300, "end_mhz": 300, "is_noise_floor": True},
    ], normalization={"mean_dbm": [10, 10, 10], "std_dbm": [2, 2, 2]})
    assert rows[0]["mae_db"] == 1.0
    assert rows[1]["rmse_db"] == 2.0
    assert rows[1]["is_noise_floor"] is True


def test_finite_origin_mask_rejects_input_history_and_target_outages():
    timestamps = pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC")
    values = np.ones((6, 2, 1), dtype=np.float32)
    values[1, 1, 0] = np.nan
    series = ReceiverSeries(
        values,
        timestamps,
        np.array([100.0]),
        ("base", "guest"),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    origins = np.array([1, 3])
    targets = np.ones((2, 1, 1, 1), dtype=np.float32)
    targets[1] = np.nan
    mask = _finite_origin_mask(series, origins, [0, 1], targets, lookback=2)
    assert mask.tolist() == [False, False]


def test_map_metrics_report_full_grid_and_receiver_grid_scopes():
    predicted = np.zeros((2, 1, 2, 2, 1), dtype=np.float32)
    target = np.zeros_like(predicted)
    predicted[:, 0, 0, 0, 0] = 2.0
    rows = _map_metric_rows(
        predicted,
        target,
        np.array([100.0]),
        [{"region_id": "R1", "start_mhz": 100, "end_mhz": 100}],
        [1],
        normalization=None,
        grid=None,
        receiver_indices=[(0, 0)],
    )
    assert {row["metric_scope"] for row in rows} == {
        "full_grid_idw",
        "receiver_grid_idw",
    }
    values = {row["metric_scope"]: row["mae_db"] for row in rows}
    assert values["full_grid_idw"] == 0.5
    assert values["receiver_grid_idw"] == 2.0
