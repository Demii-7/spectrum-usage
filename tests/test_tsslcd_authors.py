import numpy as np
import pandas as pd

from training.ray import tsslcd_authors as T


def test_savgol_off_is_identity_and_polynomials_are_reproduced():
    t = np.arange(200, dtype=float)[:, None]
    x = np.hstack([0.01 * t ** 2 - 0.3 * t + 5, 2 * t])
    assert T.smooth_series(x, "off") is x
    for kind in ("centered", "causal"):
        np.testing.assert_allclose(T.smooth_series(x, kind), x, atol=1e-6)


def test_causal_savgol_never_uses_future_samples_but_centered_does():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = x.copy()
    y[200:] += 10.0                                       # change only the future of row 150
    causal_a, causal_b = T.smooth_series(x, "causal"), T.smooth_series(y, "causal")
    np.testing.assert_allclose(causal_a[:200], causal_b[:200])
    centered_a, centered_b = T.smooth_series(x, "centered"), T.smooth_series(y, "centered")
    assert not np.allclose(centered_a[190:200], centered_b[190:200])


def test_causal_savgol_leaves_the_first_window_unfiltered():
    x = np.random.default_rng(1).normal(size=(120, 2))
    out = T.smooth_series(x, "causal")
    np.testing.assert_array_equal(out[:T.SAVGOL_WINDOW - 1], x[:T.SAVGOL_WINDOW - 1])
    assert not np.allclose(out[T.SAVGOL_WINDOW:], x[T.SAVGOL_WINDOW:])


def test_segments_and_windows_never_cross_a_gap():
    valid = np.array([1] * 150 + [0] * 5 + [1] * 130 + [0] + [1] * 20, dtype=bool)
    segments = T.contiguous_segments(valid, 120)
    assert segments == [(0, 150), (155, 285)]
    starts = T.window_starts(segments, 120)
    assert starts.min() == 0 and starts.max() == 165
    for s in starts:
        assert any(a <= s and s + 120 <= b for a, b in segments)
    assert len(T.window_starts(segments, 120, stride=7)) < len(starts)


def test_minmax_is_fit_on_the_given_rows_and_inverts():
    train = np.array([[0.0, 10.0], [4.0, 30.0]])
    scaler = T.MinMax().fit(train)
    np.testing.assert_allclose(scaler.transform(np.array([[2.0, 20.0], [8.0, 50.0]])), [[0.5, 0.5], [2.0, 2.0]])
    np.testing.assert_allclose(scaler.inverse(scaler.transform(train)), train)


def test_moving_average_modes_and_identity():
    x = np.arange(10, dtype=float)[:, None]
    assert T.moving_average(x, 1, "centered") is x
    causal = T.moving_average(x, 3, "causal")[:, 0]
    np.testing.assert_allclose(causal[2:], x[2:, 0] - 1.0)                       # mean of (t-2, t-1, t)
    centered = T.moving_average(x, 3, "centered")[:, 0]
    np.testing.assert_allclose(centered[1:-1], x[1:-1, 0])                       # mean of (t-1, t, t+1)


def test_eval_smoothing_is_applied_inside_each_segment_only():
    series = np.concatenate([np.zeros((5, 1)), np.full((5, 1), 10.0)])
    segments = np.array([0] * 5 + [1] * 5)
    out = T.smooth_windows(series, segments, 3, "centered")
    assert out[:5].max() == 0.0 and out[5:].min() == 10.0                        # no leakage across the boundary


def test_score_grid_scores_both_targets_and_skips_smoothing_without_consecutive_windows():
    rng = np.random.default_rng(2)
    n, d = 40, 5
    truth = rng.normal(size=(n, len(T.HORIZONS), d))
    noisy_gt = truth + rng.normal(scale=0.5, size=truth.shape)                  # the "smoothed" model target
    seg = np.zeros(n, dtype=int)
    smooth = [T.parse_eval_smooth("1:1:centered"), T.parse_eval_smooth("6:3:causal")]
    grid = T.score_grid(truth, noisy_gt, truth, seg, smooth, consecutive=True)
    assert grid["raw_target__off__mean_horizon_mae_db"] == 0.0                   # predictions equal the raw truth
    assert grid["model_target__off__mean_horizon_mae_db"] > 0.2
    assert {k.split("__")[1] for k in grid} == {"off", "p6g3cau"}
    assert {k.split("__")[0] for k in grid} == {"model_target", "raw_target"}
    thin = T.score_grid(truth, noisy_gt, truth, seg, smooth, consecutive=False)
    assert {k.split("__")[1] for k in thin} == {"off"}


def test_parse_eval_smooth_validates():
    assert T.parse_eval_smooth("6:3:causal") == (6, 3, "causal")
    for bad in ("6:3:mean", "0:3:causal"):
        try:
            T.parse_eval_smooth(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_load_partition_joins_receivers_on_common_minutes_and_fills_short_gaps(tmp_path):
    start = pd.Timestamp(T.RANGES["train"][0])
    for k, site in enumerate(("a-nuc1", "b-nuc1")):
        folder = tmp_path / "data" / "powder" / site / T.RUNS["train"] / "600_800"
        folder.mkdir(parents=True)
        minutes = pd.date_range(start, periods=30, freq="min")
        minutes = minutes.delete([10, 11] if k == 0 else [20])                   # a short gap in each receiver
        frame = pd.DataFrame({"timestamp_utc": minutes.strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "600.5": np.arange(len(minutes), dtype=float) + k,
                              "601.5": np.arange(len(minutes), dtype=float) * 2})
        frame.to_csv(folder / T.RAW_NAME, index=False)
    values, grid, valid = T.load_partition(str(tmp_path), "train", sites=("a-nuc1", "b-nuc1"))
    assert values.shape[1] == 4                                                    # 2 receivers x 2 bins, receiver-major
    assert valid[:30].all() and not valid[30:].any()                               # filled inside, nothing outside the data
