import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "analysis"))
import forecastability_atlas as fa  # noqa: E402


def _ar1(n, phi, bins=3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, bins))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 1, bins)
    return x


def test_autocorrelation_recovers_ar1_coefficient_and_handles_nan():
    x = _ar1(6000, 0.8)
    x[100:110] = np.nan
    ac = fa.autocorrelation(x, 1)
    assert np.all(np.abs(ac - 0.8) < 0.05)
    assert np.all(np.isnan(fa.autocorrelation(x[:50], 100)))


def test_daily_profile_r2_is_high_for_a_daily_cycle_and_low_for_noise():
    idx = pd.date_range("2026-10-27", periods=96 * 4, freq="15min", tz="UTC")
    phase = 2 * np.pi * (idx.hour * 60 + idx.minute) / 1440
    cycle = pd.DataFrame({"a": np.sin(phase) + 0.05 * np.random.default_rng(0).normal(size=len(idx))}, index=idx)
    noise = pd.DataFrame({"a": np.random.default_rng(1).normal(size=len(idx))}, index=idx)
    assert fa.daily_profile_r2(cycle, 96)[0] > 0.9
    assert fa.daily_profile_r2(noise, 96)[0] < 0.05   # cross-validated: no overfit credit for noise


def test_headroom_is_small_for_iid_noise_and_large_for_a_smooth_trend():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (4000, 4))
    r = fa.causal_vs_oracle(noise, 5)
    assert 0 < 1 - np.mean(r["oracle_mae"]) / np.mean(r["causal_mae"]) < 0.2
    trend = np.linspace(0, 100, 4000)[:, None] + rng.normal(0, 0.01, (4000, 4))
    t = fa.causal_vs_oracle(trend, 5)
    assert 1 - np.mean(t["oracle_mae"]) / np.mean(t["causal_mae"]) > 0.8


def test_residual_ar_model_beats_the_window_mean_on_a_persistent_series():
    x = _ar1(3000, 0.95, bins=3)
    g = fa.residual_ar_gain(x)
    assert np.all(g["model_mae"] < g["window_mean_mae"])
    assert np.all(g["oracle_mae_test"] > 0)


def test_block_db_averages_in_linear_power_and_drops_sparse_blocks():
    idx = pd.date_range("2026-10-27T20:00:00Z", periods=30, freq="min")
    f = pd.DataFrame({"a": [-100.0] * 29 + [-90.0]}, index=idx)
    out = fa.block_db(f, 15)
    assert out.iloc[1, 0] > -100.0 + 0.5
    f.iloc[0:8] = np.nan                       # first block has only 7 of 15 valid minutes -> dropped
    assert np.isnan(fa.block_db(f, 15).iloc[0, 0])


def test_summarize_cell_returns_all_measures_on_a_synthetic_trace():
    n = 60 * 24 * 3
    idx = pd.date_range("2026-10-27T00:00:00Z", periods=n, freq="min")
    rng = np.random.default_rng(0)
    cols = {f"{600 + i}.5": -100 + 3 * np.sin(2 * np.pi * np.arange(n) / 1440) * (i % 2) + rng.normal(0, 1, n)
            for i in range(6)}
    cell, bins = fa.summarize_cell(pd.DataFrame(cols, index=idx),
                                   {"testbed": "t", "site": "s", "run": "r", "band": "b"})
    for key in ("hours", "active_frac", "E_actual_median", "r15_k2_headroom_mae", "r1_k30_headroom_mse",
                "r15_realized_gain_mae", "daily_r2_median", "autocorr_1440min_median"):
        assert key in cell, key
    assert len(bins) == 6 and "headroom_mae_15min" in bins[0]
    assert cell["hours"] == 72.0 and 0 <= cell["active_frac"] <= 1
