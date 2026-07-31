from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from training.common.transfer_4d import (
    DEFAULT_MANIFEST,
    ROOT,
    TRANSFER_MODELS,
    apply_checkpoint_normalization,
    build_target_maps,
    campaign_cells,
    compare_transfer_outputs,
    derive_task_grid,
    load_checkpoint_model,
    load_transfer_manifest,
    resolve_checkpoint,
    transfer_tasks,
)


def test_manifest_defines_exactly_48_seed_42_cells():
    manifest = load_transfer_manifest(DEFAULT_MANIFEST)
    tasks = transfer_tasks(manifest)
    cells = campaign_cells()
    assert len(tasks) == 8
    assert len(cells) == 48
    assert {cell["seed"] for cell in cells} == {42}
    assert {cell["model"] for cell in cells} == set(TRANSFER_MODELS)


def test_residual_convlstm_uses_available_checkpoint_seed():
    manifest = load_transfer_manifest(DEFAULT_MANIFEST)
    entry = manifest["checkpoints"]["residualconvlstm"]
    assert entry["source"] == "archive"
    assert entry["checkpoint_seed"] == 41


def test_task_semantics_and_intervals_are_explicit():
    tasks = {task.task_id: task for task in transfer_tasks()}
    point = tasks["new-point-powder-600-800"]
    assert point.specification["option"] == 2
    assert point.temporal_overlap is True
    assert len(point.input_receivers) == 7
    assert point.score_receivers == ("guesthouse",)
    assert point.raw_start == pd.Timestamp("2026-06-28T04:37:00Z")
    assert point.raw_end == pd.Timestamp("2026-07-03T02:54:00Z")
    reference = tasks["reference-powder-600-800"]
    assert len(reference.input_receivers) == 6
    new_bands = [task for task in tasks.values() if task.evaluation == "new_band"]
    assert {task.frequency_start_mhz for task in new_bands} == {800, 2400, 3500, 5725}
    assert all(task.raw_start == pd.Timestamp("2026-07-03T18:39:00Z") for task in new_bands)
    ara = [task for task in tasks.values() if task.platform == "ara"]
    assert len(ara) == 2
    assert all(task.geometry_receivers == ("ames", "horticulture") for task in ara)
    assert all(task.score_receivers == ("ames", "horticulture") for task in ara)


def test_manifest_paths_are_repository_relative_and_match_requested_data():
    manifest = load_transfer_manifest(DEFAULT_MANIFEST)
    values = [manifest["config"]]
    for entry in manifest["tasks"].values():
        values.extend(entry["files"].values())
        values.append(entry["locations"])
    assert all(not Path(value).is_absolute() for value in values)
    assert all((ROOT / value).is_absolute() for value in values)
    point_files = manifest["tasks"]["new-point-powder-600-800"]["files"]
    assert "20260628T0436Z" in point_files["guesthouse"]
    assert "20260628T0436Z" in point_files["humanities"]
    assert "20260628T0437Z" in point_files["cpg"]


def test_checkpoint_statistics_are_applied_by_bin_position():
    target = np.array([[[[800.5, 801.5]]]], dtype=np.float32)
    normalized = apply_checkpoint_normalization(
        target, {"mean_dbm": [600.5, 601.5], "std_dbm": [2.0, 4.0]}
    )
    np.testing.assert_allclose(normalized, [[[[100.0, 50.0]]]])


def test_powder_grid_is_frozen_to_six_geometry_receivers():
    task = next(task for task in transfer_tasks() if task.evaluation == "new_point")
    coordinates = {
        "cpg": (-111.84, 40.76),
        "ebc": (-111.83, 40.77),
        "humanities": (-111.85, 40.765),
        "madsen": (-111.835, 40.75),
        "moran": (-111.837, 40.78),
        "sagepoint": (-111.82, 40.762),
        "guesthouse": (-100.0, 50.0),
    }
    first = derive_task_grid(task, coordinates)
    coordinates["guesthouse"] = (0.0, 0.0)
    second = derive_task_grid(task, coordinates)
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.origin_longitude == second.origin_longitude
    assert first.origin_latitude == second.origin_latitude


def test_future_target_maps_have_full_forecast_shape():
    task = next(task for task in transfer_tasks() if task.task_id == "new-site-ara-600-800")
    coordinates = {"ames": (-93.659232, 42.0116096),
                   "horticulture": (-93.6179377, 42.0613321)}
    grid = derive_task_grid(task, coordinates)
    future = np.zeros((3, 3, 2, 200), dtype=np.float32)
    maps = build_target_maps(future, task, coordinates, grid)
    assert maps.shape == (3, 3, 10, 10, 200)


def test_physical_and_full_grid_metrics_use_distinct_targets():
    predicted_maps = np.ones((2, 3, 10, 10, 2), dtype=np.float32)
    target_maps = np.zeros_like(predicted_maps)
    predicted_receivers = np.ones((2, 3, 1, 2), dtype=np.float32)
    raw_receivers = np.ones_like(predicted_receivers)
    target_map_receivers = np.zeros_like(predicted_receivers)
    map_rows, receiver_rows = compare_transfer_outputs(
        predicted_maps, target_maps, predicted_receivers, raw_receivers,
        target_map_receivers, [1, 15, 60], ["guesthouse"],
    )
    full_grid = next(row for row in map_rows if row["metric_scope"] == "full_grid_idw")
    receiver_grid = next(row for row in map_rows if row["metric_scope"] == "receiver_grid_idw")
    physical = receiver_rows[0]
    assert full_grid["mae_db"] == 1.0
    assert receiver_grid["mae_db"] == 1.0
    assert physical["mae_db"] == 0.0


def test_resolved_checkpoint_model_config_and_strict_loading(tmp_path):
    config = yaml.safe_load((ROOT / "training/configs/transfer_4d_models.yaml").read_text())
    assert config["convlstm"]["model"]["hidden_channels"] == [32, 64]
    assert config["residualconvlstm"]["model"]["hidden_channels"] == [16, 32]
    assert config["linearar4d"]["model"]["ridge_alpha"] == 0.01
    assert config["residuallinearar4d"]["model"]["ridge_alpha"] == 0.01
    assert config["dswinlstm_i"]["model"]["prediction_horizon"] == 60
    assert config["dswinlstm_i"]["model"]["decoder_feedback"] == "pixel_feedback"
    assert config["lookbackmean4d"]["model"]["prediction_horizon"] == 1

    pytest.importorskip("torch")
    archive = ROOT / "checkpoints.tgz"
    if not archive.is_file():
        pytest.skip("local checkpoint archive is unavailable")
    manifest = load_transfer_manifest(DEFAULT_MANIFEST)
    checkpoint = resolve_checkpoint(
        manifest["checkpoints"]["convlstm"], archive=archive, cache_dir=tmp_path
    )
    model, normalization = load_checkpoint_model(checkpoint, "convlstm", config)
    assert model.training is False
    assert np.asarray(normalization["mean_dbm"]).squeeze().shape == (200,)
