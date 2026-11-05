import numpy as np
import pandas as pd

from training.ray.electrosense import (
    SEQ_LEN, baseline_metrics, convert_raw, predict_baseline, split_borders, standardized_from_per_frequency,
    window_ranges,
)


def test_split_borders_match_time_series_library_for_2872_rows():
    b = split_borders(2872)
    assert b["train"] == (0, 2010)
    assert b["validation"] == (2010 - 192, 2010 + 288)   # 288 = 2872 - 2010 - 574
    assert b["test"] == (2872 - 574 - 192, 2872)


def _raw(n=300, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(10, 2, (n, channels)), columns=["500", "650", "800"][:channels])
    frame.insert(0, "date", pd.date_range("2021-05-28", periods=n, freq="15min").strftime("%Y/%m/%d %H:%M"))
    return frame


def test_convert_raw_stamps_rows_as_minutes_and_fits_stats_on_train_only():
    raw = _raw(400)
    series, stats = convert_raw(raw)
    assert list(series.columns) == ["timestamp_utc", "500.0", "650.0", "800.0"]
    assert series["timestamp_utc"].iloc[1] == "2021-01-01T00:01:00Z"
    train_end = stats["borders"]["train"][1]
    assert np.allclose(stats["train_mean"], raw.iloc[:train_end, 1:].astype(float).mean(axis=0))
    assert np.allclose(stats["train_std"], raw.iloc[:train_end, 1:].astype(float).std(axis=0, ddof=0))
    ranges = window_ranges(stats)
    assert ranges["train"]["start"] == "2021-01-01T00:00:00Z"
    assert ranges["test"]["end"] == "2021-01-01T06:39:00Z"  # last of 400 rows = minute 399


def test_baselines_on_a_constant_series_are_exact_and_seasonal_uses_the_day_before():
    n = 800
    t = np.arange(n)
    wave = np.stack([np.sin(2 * np.pi * t / 96), np.cos(2 * np.pi * t / 96)], axis=1) * 3 + 10
    raw = pd.DataFrame(wave, columns=["500", "800"])
    raw.insert(0, "date", "x")
    series, stats = convert_raw(raw)
    values = series.drop(columns=["timestamp_utc"]).to_numpy()
    seasonal = baseline_metrics(values, stats, "seasonal_naive", horizons=(12, 96), seq_len=SEQ_LEN)
    assert seasonal["mse_raw_h12"] < 1e-9 and seasonal["mse_raw_h96"] < 1e-9   # a pure daily cycle is predicted exactly
    last = baseline_metrics(values, stats, "last_value", horizons=(12,), seq_len=SEQ_LEN)
    assert last["mse_raw_h12"] > 0.1
    ctx = values[:SEQ_LEN]
    assert predict_baseline("seasonal_mean", ctx, 192, np.zeros(2)).shape == (192, 2)
    assert predict_baseline("seasonal_naive", ctx, 192, np.zeros(2)).shape == (192, 2)


def test_standardized_metrics_divide_by_train_std_per_bin():
    stats = {"frequencies": [500.0, 800.0], "train_std": [2.0, 0.5]}
    per_freq = pd.DataFrame({
        "split": ["2d_test"] * 4 + ["2d_validation"] * 2,
        "frequency_mhz": [500.0, 800.0, 500.0, 800.0, 500.0, 800.0],
        "horizon": [12, 12, 24, 24, 12, 12],
        "mae_db": [1.0, 1.0, 2.0, 2.0, 9.0, 9.0],
        "mse_db2": [4.0, 1.0, 16.0, 4.0, 99.0, 99.0],
    })
    out = standardized_from_per_frequency(per_freq, stats)
    assert abs(out["mse_std_h12"] - np.mean([4.0 / 4.0, 1.0 / 0.25])) < 1e-9     # (1 + 4) / 2
    assert abs(out["mae_std_h12"] - np.mean([0.5, 2.0])) < 1e-9
    assert abs(out["mse_raw_h12"] - 2.5) < 1e-9                                  # validation rows ignored
    assert abs(out["mse_std_mean"] - np.mean([out["mse_std_h12"], out["mse_std_h24"]])) < 1e-9


def test_tsl_command_uses_the_paper_protocol_and_pins_run_identity():
    from training.ray.tsl_runs import parse_metrics, run_command

    cmd = run_command("DLinear", "spec25", 48, 2022, "/data/", "800", training=True, inverse=False, has_seed_flag=True)
    joined = " ".join(cmd)
    for expected in ("--seq_len 192", "--pred_len 48", "--enc_in 25", "--dec_in 25", "--c_out 25", "--features M",
                     "--data custom", "--target 800", "--model DLinear", "--is_training 1", "--seed 2022",
                     "--model_id spec25_192_48_s2022"):
        assert expected in joined, expected
    assert "--inverse" not in joined
    raw = " ".join(run_command("DLinear", "spec4", 12, 1, "/d/", "800", training=False, inverse=True,
                               has_seed_flag=False))
    assert "--is_training 0" in raw and "--inverse" in raw and "--seed" not in raw and "--enc_in 4" in raw
    assert parse_metrics("noise\nmse:0.2718, mae:0.3105\n") == (0.2718, 0.3105)
    assert parse_metrics("no metrics here") is None


def test_expand_variants_adds_one_job_per_variant_with_overrides():
    import pytest

    from training.ray.electrosense import expand_variants

    jobs = [{"model": "residuallinearar1d", "seed": 40}, {"model": "residuallinearar2d", "seed": 40}]
    assert expand_variants(jobs, None) == jobs
    out = expand_variants(jobs, ['seasonal={"baseline":"seasonal","seasonal_period":96}', 'norm={"window_norm":"std"}'])
    assert len(out) == 4 and {j["variant"] for j in out} == {"seasonal", "norm"}
    assert out[0]["model_overrides"] == {"baseline": "seasonal", "seasonal_period": 96} and "variant" not in jobs[0]
    with pytest.raises(ValueError):
        expand_variants(jobs, ["badtoken"])
