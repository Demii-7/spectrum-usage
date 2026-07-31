from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

from training.common.preprocessing import SequenceSegment
from training.common.data_sources import LoadedSource
from training.common.evaluation_integrated import calculate_errors_and_export_arrays
from training.common.transfer_1d import (
    EXPECTED_BINS,
    ROOT,
    TRANSFER_MODELS,
    apply_source_normalization,
    flatten_source,
    load_transfer_manifest,
    preflight_checkpoints,
    repository_path,
    resolve_checkpoint,
    row_normalization,
    target_window_starts,
    transfer_tasks,
    validate_target_frequencies,
)


def test_transfer_task_catalog_reuses_all_12_stable_tasks():
    tasks = {task.task_id: task for task in transfer_tasks()}
    assert len(tasks) == 12
    assert tasks["reference-powder-600-800"].target_end == pd.Timestamp(
        "2026-07-05 19:39", tz="UTC"
    )


def test_manifest_is_complete_and_uses_only_seed_42_1d_checkpoints():
    manifest = load_transfer_manifest(ROOT / "training/configs/transfer_1d_manifest.json")
    assert tuple(manifest["models"]) == TRANSFER_MODELS
    assert set(manifest["tasks"]) == {task.task_id for task in transfer_tasks()}
    paths = [manifest["config"], *manifest["models"].values()]
    paths.extend(path for task in manifest["tasks"].values() for path in task["files"])
    assert all(not Path(path).is_absolute() for path in paths)
    assert all(repository_path(path).is_absolute() for path in paths)
    assert all("checkpoints/1d/" in path and "seed-42" in path
               for path in manifest["models"].values())


def test_config_matches_scalar_seed_42_winners():
    config = yaml.safe_load((ROOT / "training/configs/transfer_1d.yaml").read_text())
    assert config["data"]["representation"] == "1d"
    assert config["windowing"] == {"lookback": 60, "horizons": [1, 15, 60]}
    for model_name in ("vanillalstm1d", "residualvanillalstm"):
        model = config[model_name]["model"]
        assert model["input_size"] == 1
        assert model["hidden_size"] == 8
        assert model["num_layers"] == 1
    assert config["linearar1d"]["model"]["ridge_alpha"] == 0.01
    assert config["residuallinearar1d"]["model"]["ridge_alpha"] == 0.01


def test_source_normalization_aligns_shifted_bins_by_position():
    mean = np.arange(EXPECTED_BINS, dtype=np.float32)
    std = np.full(EXPECTED_BINS, 2.0, dtype=np.float32)
    target = np.ones((1, EXPECTED_BINS), dtype=np.float32) + mean
    np.testing.assert_allclose(
        apply_source_normalization(target, {"mean_dbm": mean, "std_dbm": std}),
        np.full((1, EXPECTED_BINS), 0.5),
    )


def test_flattened_rows_keep_independent_frequency_segments_and_stats():
    data = np.zeros((3, EXPECTED_BINS), dtype=np.float32)
    data[:, 0] = [10.0, 12.0, 14.0]
    data[:, 1] = [20.0, 22.0, 24.0]
    source = LoadedSource(
        data=data,
        frequencies=np.arange(EXPECTED_BINS, dtype=np.float32),
        timestamps=pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
        files=[],
        segments=(SequenceSegment(0, 3, "site"),),
        feature_labels=[],
    )
    flattened = flatten_source(source)
    assert flattened.data.shape == (3 * EXPECTED_BINS, 1)
    assert flattened.segments[0] == SequenceSegment(0, 3, "site:frequency_0")
    assert flattened.segments[1] == SequenceSegment(3, 6, "site:frequency_1")
    normalized, indices = row_normalization(
        {"mean_dbm": np.arange(EXPECTED_BINS), "std_dbm": np.ones(EXPECTED_BINS)},
        flattened.segments,
        np.array([1, 4, 3 * EXPECTED_BINS - 1]),
    )
    np.testing.assert_array_equal(indices, [0, 1, 199])
    np.testing.assert_array_equal(normalized["mean_dbm"].ravel(), [0, 1, 199])


def test_metrics_denormalize_each_flattened_target_by_frequency_index():
    segments = (SequenceSegment(0, 3, "site:frequency_0"),
                SequenceSegment(3, 6, "site:frequency_1"))
    normalization = {"mean_dbm": np.full(EXPECTED_BINS, 10.0),
                    "std_dbm": np.full(EXPECTED_BINS, 2.0)}
    normalization["mean_dbm"][1] = 20.0
    rows = np.array([1, 4])
    selected, _ = row_normalization(normalization, segments, rows)
    _, _, absolute, squared = calculate_errors_and_export_arrays(
        prediction_normalized=np.array([[2.0], [1.0]], dtype=np.float32),
        target_raw_model_layout=np.array([[12.0], [24.0]], dtype=np.float32),
        normalization=selected,
    )
    np.testing.assert_allclose(absolute, [[2.0], [2.0]])
    np.testing.assert_allclose(squared, [[4.0], [4.0]])


def test_target_windows_never_cross_flattened_series():
    timestamps = pd.DatetimeIndex(
        list(pd.date_range("2026-01-01", periods=8, freq="min", tz="UTC")) * 2
    )
    segments = (SequenceSegment(0, 8, "site:frequency_0"),
                SequenceSegment(8, 16, "site:frequency_1"))
    starts = target_window_starts(
        timestamps,
        segments,
        lookback=2,
        max_horizon=2,
        stride=1,
        target_start=pd.Timestamp("2026-01-01 00:02", tz="UTC"),
        target_end=pd.Timestamp("2026-01-01 00:06", tz="UTC"),
    )
    assert starts.tolist() == [0, 1, 2, 8, 9, 10]


def test_frequency_policy_requires_exactly_200_one_mhz_bins():
    task = next(task for task in transfer_tasks()
                if task.task_id == "new-bands-powder-2400-2600")
    assert validate_target_frequencies(task, np.arange(2400.5, 2600.0, 1.0)) \
        == "source_checkpoint_stats_by_bin_position"
    with pytest.raises(ValueError, match="exactly 200"):
        validate_target_frequencies(task, np.arange(2400.0, 2601.0, 1.0))


def test_seed_42_checkpoints_strict_load_with_checked_in_config():
    manifest = load_transfer_manifest(ROOT / "training/configs/transfer_1d_manifest.json")
    checkpoints = {name: resolve_checkpoint(path)
                   for name, path in manifest["models"].items()}
    config = yaml.safe_load((ROOT / manifest["config"]).read_text())
    preflight_checkpoints(config, checkpoints)
