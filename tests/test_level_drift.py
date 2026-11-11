import numpy as np
import pandas as pd

from training.ray import level_diagnostics_origins as L


def test_window_means_matches_manual_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 4))
    starts = np.array([0, 3, 17, 30])
    got = L.window_means(data, starts, 10)
    want = np.stack([data[s:s + 10].mean(axis=0) for s in starts])
    np.testing.assert_allclose(got, want)


def test_level_distance_is_absolute_and_scales_by_std():
    means = np.array([[1.0, -2.0], [3.0, 5.0]])
    mu = np.array([2.0, 1.0])
    d = L.level_distance(means, mu)
    np.testing.assert_allclose(d, [[1.0, 3.0], [1.0, 4.0]])
    np.testing.assert_allclose(L.level_distance(means, mu, np.array([2.0, 4.0])), [[0.5, 0.75], [0.5, 1.0]])


def test_quantile_dict_keys_and_values():
    q = L.quantile_dict(np.arange(101.0))
    assert q["q10"] == 10.0 and q["q50"] == 50.0 and q["q90"] == 90.0 and abs(q["q99"] - 99.0) < 1e-9 and q["n"] == 101


def test_day_index_floors_whole_days():
    times = pd.DatetimeIndex(["2026-10-27T20:00:00Z", "2026-10-28T19:59:00Z", "2026-10-28T20:00:00Z", "2026-11-03T20:00:00Z"])
    assert L.day_index(times, "2026-10-27T20:00:00Z").tolist() == [0, 0, 1, 7]


def test_ols_from_statistics_matches_polyfit_and_pools_rows():
    rng = np.random.default_rng(1)
    d = rng.uniform(0, 5, size=(30, 20))
    excess = 0.4 * d + 0.1 + rng.normal(scale=0.3, size=d.shape)
    stats = L.regression_stats(d, excess)
    got = L.ols_from_stats(*(float(stats[k].sum()) for k in ("n", "sx", "sy", "sxy", "sxx", "syy")))
    slope, intercept = np.polyfit(d.ravel(), excess.ravel(), 1)
    assert abs(got["slope"] - slope) < 1e-9 and abs(got["intercept"] - intercept) < 1e-9
    r = np.corrcoef(d.ravel(), excess.ravel())[0, 1]
    assert abs(got["r2"] - r * r) < 1e-9 and got["n"] == 600


def test_ols_degenerate_returns_nan():
    out = L.ols_from_stats(5.0, 10.0, 3.0, 6.0, 20.0, 2.0)          # x constant (all 2.0): zero variance
    assert np.isnan(out["slope"])


def test_analysis_helpers_bootstrap_and_line_slope():
    import importlib.util
    spec = importlib.util.spec_from_file_location("lda", "evaluation/analysis/level_drift_analysis.py")
    lda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lda)
    rng = np.random.default_rng(2)
    blocks = []
    for _ in range(7):                                    # seven day blocks of (window, bin) points with a true slope of 0.5
        x = rng.uniform(0, 4, 400)
        y = 0.5 * x + rng.normal(scale=0.2, size=400)
        blocks.append([400, x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()])
    blocks = np.array(blocks)
    slope = lda.ols(*blocks.sum(axis=0))[0]
    lo, hi = lda.bootstrap_slope(blocks, draws=300, rng=np.random.default_rng(3))
    assert abs(slope - 0.5) < 0.02 and lo <= slope <= hi and hi - lo < 0.1
    s, se = lda.line_slope(np.arange(7), 0.1 * np.arange(7) + np.array([0, .01, -.01, 0, .01, -.01, 0]))
    assert abs(s - 0.1) < 0.01 and se < 0.01
    assert abs(lda.gap(1.15, 1.0) - 0.15) < 1e-12
