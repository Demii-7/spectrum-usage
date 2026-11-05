import numpy as np
import pandas as pd
import pytest

from training.ray.powder15 import (
    HORIZONS, LOOKBACK, OUT_NAME, apply_powder15, block_average_db, convert_trace, fill_gaps, set_lookback_and_horizon,
)


def _minute_frame(n=90, value=-100.0, start="2026-10-27T20:00:00Z"):
    idx = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame({"600.5": np.full(n, value), "601.5": np.full(n, value + 5)}, index=idx)


def test_block_average_is_done_in_linear_power():
    idx = pd.date_range("2026-10-27T20:00:00Z", periods=15, freq="min")
    values = np.array([-100.0] * 14 + [-90.0])            # one minute 10 dB higher
    frame = pd.DataFrame({"a": values}, index=idx)
    out = block_average_db(frame).iloc[0, 0]
    expected = 10 * np.log10(np.mean(10 ** (values / 10)))
    assert out == pytest.approx(expected) and out > np.mean(values)   # linear mean > dB mean


def test_blocks_missing_too_many_minutes_are_dropped_then_interpolated():
    frame = _minute_frame(60)
    frame.iloc[16:30] = np.nan                             # second block nearly empty
    blocks = block_average_db(frame)
    assert np.isnan(blocks.iloc[1, 0]) and not np.isnan(blocks.iloc[0, 0])
    filled, share = fill_gaps(blocks)
    assert np.isfinite(filled.to_numpy()).all() and share == pytest.approx(0.25)
    with pytest.raises(ValueError):
        fill_gaps(blocks, max_gap=0)                       # the gap is longer than allowed -> refuse


def test_convert_trace_writes_pipeline_csv(tmp_path):
    frame = _minute_frame(60)
    raw = frame.copy(); raw.insert(0, "timestamp_utc", frame.index.strftime("%Y-%m-%dT%H:%M:%SZ"))
    raw_path = tmp_path / "raw.csv"; raw.to_csv(raw_path, index=False)
    out_path = tmp_path / "x" / OUT_NAME
    report = convert_trace(raw_path, out_path)
    out = pd.read_csv(out_path)
    assert list(out.columns) == ["timestamp_utc", "600.5", "601.5"] and len(out) == 4 == report["blocks"]
    assert out["timestamp_utc"].iloc[1] == "2026-10-27T20:15:00Z"
    assert np.allclose(out["600.5"], -100.0)


def test_apply_powder15_swaps_paths_windows_and_keeps_ranges():
    base = {"data": {"files": [
        {"path": "data/powder/humanities-nuc1/20261015T1434Z/600_800/power_1mhz_avg_per_minute.csv", "partition": "train"},
        {"path": "data/powder/humanities-nuc1/20261025T1355Z/600_800/power_1mhz_avg_per_minute.csv", "partition": "test"}],
        "split": {"ranges": {"train": {"start": "a", "end": "b"}}}, "map": {"name": "m"}},
        "windowing": {"lookback": 60, "horizons": [1, 15, 60]}, "preprocessing": {"normalize": True, "impute": True}}
    out = apply_powder15(base, "2400_2600", "/repo", list(HORIZONS))
    assert out["data"]["files"][0]["path"] == "data/powder15/humanities-nuc1/20261015T1434Z/2400_2600/power_15min.csv"
    assert out["windowing"] == {"lookback": LOOKBACK, "horizons": sorted(HORIZONS)}
    assert out["preprocessing"]["impute"] is False and "map" not in out["data"]
    assert out["data"]["split"] == base["data"]["split"]
    assert base["windowing"]["lookback"] == 60              # input untouched
    model_cfg = {"vanillalstm": {"model": {"input_sequence_length": 60, "prediction_horizon": 1}, "train": {"test_stride": 5}}}
    direct = set_lookback_and_horizon(model_cfg, "vanillalstm", 24)
    assert direct["vanillalstm"]["model"] == {"input_sequence_length": LOOKBACK, "prediction_horizon": 24}
    assert direct["vanillalstm"]["train"]["test_stride"] == 1
