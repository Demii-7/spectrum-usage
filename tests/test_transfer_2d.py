import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from training.common.preprocessing import SequenceSegment
from training.common.transfer_2d import (
    ROOT, TRANSFER_MODELS, apply_source_normalization, load_transfer_manifest,
    preflight_checkpoints, repository_path, target_window_starts, transfer_tasks,
    resolve_checkpoint, validate_target_frequencies,
)


def test_transfer_task_catalog_has_exact_stable_intervals():
    tasks = {task.task_id: task for task in transfer_tasks()}
    assert len(tasks) == 12
    reference = tasks["reference-powder-600-800"]
    assert reference.raw_start == pd.Timestamp("2026-07-03 18:39", tz="UTC")
    assert reference.raw_end == pd.Timestamp("2026-07-05 20:39", tz="UTC")
    assert reference.target_start == pd.Timestamp("2026-07-03 19:39", tz="UTC")
    assert reference.target_end == pd.Timestamp("2026-07-05 19:39", tz="UTC")


def test_manifest_rejects_unstable_task_id(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "config": "x.yaml", "models": {
        m: "x.pt" for m in TRANSFER_MODELS},
                                "tasks": {"typo": {"files": []}}}))
    with pytest.raises(ValueError, match="stable transfer task IDs"):
        load_transfer_manifest(path)


def test_source_normalization_is_applied_without_target_fit():
    target = np.array([[12.0, 26.0], [1000.0, -1000.0]], dtype=np.float32)
    result = apply_source_normalization(target, {"mean_dbm": [10.0, 20.0], "std_dbm": [2.0, 3.0]})
    np.testing.assert_allclose(result[0], [1.0, 2.0])
    np.testing.assert_allclose(result[1], [495.0, -340.0])


def test_target_windows_do_not_cross_sites_and_align_frequency_shift_targets():
    timestamps = pd.DatetimeIndex(list(pd.date_range("2026-01-01", periods=8, freq="min", tz="UTC")) * 2)
    segments = (SequenceSegment(0, 8, "site-a"), SequenceSegment(8, 16, "site-b"))
    starts = target_window_starts(timestamps, segments, lookback=2, max_horizon=2, stride=1,
                                  target_start=pd.Timestamp("2026-01-01 00:02", tz="UTC"),
                                  target_end=pd.Timestamp("2026-01-01 00:06", tz="UTC"))
    assert starts.tolist() == [0, 1, 2, 8, 9, 10]
    shifted = apply_source_normalization(np.array([[2400.0, 2401.0]], dtype=np.float32),
                                         {"mean_dbm": [600.0, 601.0], "std_dbm": [2.0, 2.0]})
    np.testing.assert_array_equal(shifted, [[900.0, 900.0]])


def test_checked_in_manifest_is_complete_and_repository_relative():
    manifest = load_transfer_manifest(ROOT / "training/configs/transfer_2d_manifest.json")
    assert set(manifest["tasks"]) == {task.task_id for task in transfer_tasks()}
    paths = [manifest["config"], *manifest["models"].values()]
    paths.extend(path for task in manifest["tasks"].values() for path in task["files"])
    assert all(not pd.io.common.is_url(path) and not Path(path).is_absolute()
               for path in paths)
    assert all("seed-42" in path for path in manifest["models"].values())
    assert all(repository_path(path).is_absolute() for path in paths)


def test_checked_in_config_matches_seed_42_architectures():
    config = yaml.safe_load((ROOT / "training/configs/transfer_2d.yaml").read_text())
    assert config["windowing"] == {"lookback": 60, "horizons": [1, 15, 60]}
    assert config["autoformer_csa"]["model"] | {} == {
        "input_sequence_length": 60, "prediction_horizon": 60, "label_len": 30,
        "d_model": 64, "d_ff": 256, "encoder_layers": 2, "decoder_layers": 1,
        "n_heads": 8, "moving_avg": 25, "dropout": 0.05, "factor": 3,
        "csam_kernel_size": 7, "output_attention": False,
    }
    assert config["linearar2d"]["model"]["ridge_alpha"] == 0.01
    assert config["residuallinearar2d"]["model"]["ridge_alpha"] == 0.0001
    assert config["lstmattn"]["model"]["attention_size"] == 128
    assert config["lstmattn"]["model"]["prediction_horizon"] == 1
    assert config["residualvanillalstm"]["model"]["hidden_size"] == 8
    assert config["temporalconvnet"]["model"]["hidden_channels"] == [8] * 6
    assert config["temporalconvnet"]["model"]["feature_mode"] == "independent"
    assert config["vanillalstm"]["model"]["hidden_size"] == 128
    assert config["vanillalstm"]["model"]["prediction_horizon"] == 1


def test_frequency_policy_requires_200_ordered_one_mhz_bins():
    task = next(task for task in transfer_tasks() if task.task_id == "new-bands-powder-2400-2600")
    assert validate_target_frequencies(task, np.arange(2400.5, 2600.0, 1.0)) == \
        "source_checkpoint_stats_by_bin_position"
    with pytest.raises(ValueError, match="exactly 200 ordered"):
        validate_target_frequencies(task, np.arange(2400.0, 2601.0, 1.0))


def test_extracted_seed_42_checkpoints_strict_load_with_checked_in_config():
    manifest = load_transfer_manifest(ROOT / "training/configs/transfer_2d_manifest.json")
    checkpoints = {name: resolve_checkpoint(path) for name, path in manifest["models"].items()}
    config = yaml.safe_load((ROOT / manifest["config"]).read_text())
    preflight_checkpoints(config, checkpoints)
