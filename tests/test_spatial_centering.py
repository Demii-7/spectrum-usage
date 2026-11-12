import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("yaml")
from training.common.spatial_ablation_eval import center_window_batch, own_window_levels  # noqa: E402


def test_center_window_batch_removes_the_lookback_mean_and_returns_it():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(5, 8, 3, 4)).astype(np.float32)
    centered, level = center_window_batch(windows)
    assert centered.shape == windows.shape and level.shape == (5, 1, 3, 4)
    assert np.allclose(centered.mean(axis=1), 0, atol=1e-6)
    assert np.allclose(centered + level, windows, atol=1e-6)


def test_own_window_levels_use_rows_origin_minus_lookback_plus_one_to_origin():
    values = np.arange(20, dtype=np.float64).reshape(20, 1, 1) * np.ones((20, 2, 3))
    levels = own_window_levels(values, np.array([4, 9]), lookback=5)
    assert levels.shape == (2, 2, 3)
    assert np.allclose(levels[0], 2.0) and np.allclose(levels[1], 7.0)      # rows 0..4 and 5..9


def test_centering_is_linear_in_the_map_so_it_equals_centering_the_receiver_series():
    rng = np.random.default_rng(1)
    series = rng.normal(size=(30, 4, 6))                                      # T, receivers, bins
    weights = rng.random((7, 4)); weights /= weights.sum(axis=1, keepdims=True)     # 7 cells, idw-like weights summing to 1
    window = series[10:20]
    map_window = np.einsum("cr,trf->tcf", weights, window)
    centered_map, _ = center_window_batch(map_window[None])
    centered_series, _ = center_window_batch(window[None])
    rebuilt = np.einsum("cr,trf->tcf", weights, centered_series[0])
    assert np.allclose(centered_map[0], rebuilt, atol=1e-9)
