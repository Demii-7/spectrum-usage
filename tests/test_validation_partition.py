from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from training.common.data import ChunkSpec, load_chunk


def _write_csv(path, start, rows):
    values = np.arange(rows * 4, dtype=float).reshape(rows, 4) - 100.0
    frame = pd.DataFrame(values, columns=["100.5", "101.5", "102.5", "103.5"])
    frame.insert(0, "timestamp_utc", pd.date_range(start, periods=rows, freq="min", tz="UTC"))
    frame.to_csv(path, index=False)


def _config(train_path, later_path, validation_files):
    files = [{"path": str(train_path), "partition": "train"},
             {"path": str(later_path), "partition": "test"}]
    if validation_files:
        files.append({"path": str(later_path), "partition": "validation"})
    return {
        "data": {
            "representation": "2d",
            "files": files,
            "concat": "rows",
            "reference_site": "test",
            "split": {"ranges": {
                "train": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:09:00Z"},
                "validation": {"start": "2026-01-02T00:00:00Z", "end": "2026-01-02T00:05:00Z"},
                "test": {"start": "2026-01-02T00:10:00Z", "end": "2026-01-02T00:19:00Z"},
            }},
            "chunks": [{"id": "test", "start_mhz": 100.0, "end_mhz": 104.0}],
        },
        "windowing": {"lookback": 2, "horizons": [1]},
        "training": {"device": "cpu", "models": ["vanillalstm"]},
        "preprocessing": {"normalize": True, "impute": False, "max_missing_gap": 0},
    }


def test_validation_and_test_can_both_come_from_a_later_run(tmp_path):
    train_path, later_path = tmp_path / "train.csv", tmp_path / "later.csv"
    _write_csv(train_path, "2026-01-01", 10)
    _write_csv(later_path, "2026-01-02", 20)

    loaded = load_chunk(_config(train_path, later_path, True), ChunkSpec("test", 100.0, 104.0), 0.1)

    assert loaded.splits[loaded.train_split].raw_dbm.shape[0] == 10
    assert loaded.splits[loaded.validation_split].raw_dbm.shape[0] == 6
    assert loaded.splits[loaded.test_split].raw_dbm.shape[0] == 10
    # Normalization is fitted on the train range only.
    train_values = loaded.splits[loaded.train_split].raw_dbm
    np.testing.assert_allclose(loaded.normalization["mean_dbm"], train_values.mean(axis=0), atol=1e-5)


def test_without_validation_files_validation_is_cut_from_train_files(tmp_path):
    train_path, later_path = tmp_path / "train.csv", tmp_path / "later.csv"
    _write_csv(train_path, "2026-01-01", 10)
    _write_csv(later_path, "2026-01-02", 20)
    config = _config(train_path, later_path, False)
    # With no validation files, the validation range must fall inside the train files.
    config["data"]["split"]["ranges"]["train"] = {
        "start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:05:00Z"}
    config["data"]["split"]["ranges"]["validation"] = {
        "start": "2026-01-01T00:06:00Z", "end": "2026-01-01T00:09:00Z"}

    loaded = load_chunk(config, ChunkSpec("test", 100.0, 104.0), 0.1)

    assert loaded.splits[loaded.validation_split].raw_dbm.shape[0] == 4


def test_validation_files_require_explicit_split_ranges(tmp_path):
    train_path, later_path = tmp_path / "train.csv", tmp_path / "later.csv"
    _write_csv(train_path, "2026-01-01", 10)
    _write_csv(later_path, "2026-01-02", 20)
    config = deepcopy(_config(train_path, later_path, True))
    del config["data"]["split"]

    with pytest.raises(ValueError, match="requires data.split.ranges"):
        load_chunk(config, ChunkSpec("test", 100.0, 104.0), 0.1)
