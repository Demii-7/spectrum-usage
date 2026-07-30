from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from training.common.data import (
    ChunkSpec,
    apply_2d_spectral_mask,
    load_chunk,
)
from training.common.train_integrated import train_model


def _write_csv(path, start, values):
    frame = pd.DataFrame(values, columns=["100.5", "101.5", "102.5", "103.5"])
    frame.insert(0, "timestamp_utc", pd.date_range(start, periods=len(frame), freq="min", tz="UTC"))
    frame.to_csv(path, index=False)


def _config(train_path, test_path):
    return {
        "data": {
            "representation": "2d",
            "files": [
                {"path": str(train_path), "partition": "train"},
                {"path": str(test_path), "partition": "test"},
            ],
            "concat": "rows",
            "reference_site": "test",
            "split": {"ranges": {
                "train": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:05:00Z"},
                "validation": {"start": "2026-01-01T00:06:00Z", "end": "2026-01-01T00:09:00Z"},
                "test": {"start": "2026-01-02T00:00:00Z", "end": "2026-01-02T00:09:00Z"},
            }},
            "chunks": [{"id": "test", "start_mhz": 100.0, "end_mhz": 104.0}],
        },
        "windowing": {"lookback": 2, "horizons": [1]},
        "training": {"device": "cpu", "models": ["vanillalstm"]},
        "preprocessing": {"normalize": True, "impute": False, "max_missing_gap": 0},
        "vanillalstm": {
            "model": {"input_sequence_length": 2, "prediction_horizon": 1},
            "train": {"val_fraction": 0.1},
        },
    }


def test_low_tail_mask_is_deterministic_and_preserves_target_region():
    data = np.arange(48, dtype=np.float32).reshape(12, 4) - 120
    config = {
        "frequency_ranges": [[101.5, 102.5]],
        "replacement": "low_tail_gaussian",
        "calibration_frequency_ranges": [[100.5, 100.5]],
        "low_tail_quantile": 0.05,
        "seed": 9,
    }
    frequencies = np.array([100.5, 101.5, 102.5, 103.5])
    first = apply_2d_spectral_mask(
        data, frequencies, config, training_data=data[:8], split_seed_offset=0,
    )
    second = apply_2d_spectral_mask(
        data, frequencies, config, training_data=data[:8], split_seed_offset=0,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 1:3], data[:, 1:3])
    assert not np.array_equal(first[:, [0, 3]], data[:, [0, 3]])
    assert np.std(first[:, [0, 3]]) > 0


def test_masked_loader_preserves_raw_targets_and_uses_condition_normalization(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_values = np.arange(40, dtype=np.float32).reshape(10, 4) - 120
    test_values = np.arange(40, 80, dtype=np.float32).reshape(10, 4) - 120
    _write_csv(train_path, "2026-01-01", train_values)
    _write_csv(test_path, "2026-01-02", test_values)
    config = _config(train_path, test_path)
    unmasked = load_chunk(config, ChunkSpec("test", 100.0, 104.0), 0.1)

    masked_config = deepcopy(config)
    masked_config["data"]["mask"] = {
        "frequency_ranges": [[101.5, 102.5]],
        "replacement": "low_tail_gaussian",
        "calibration_frequency_ranges": [[100.5, 100.5]],
        "low_tail_quantile": 0.05,
        "seed": 3,
    }
    masked = load_chunk(masked_config, ChunkSpec("test", 100.0, 104.0), 0.1)

    for split_name in (masked.train_split, masked.validation_split, masked.test_split):
        np.testing.assert_array_equal(
            masked.splits[split_name].raw_dbm,
            unmasked.splits[split_name].raw_dbm,
        )
    np.testing.assert_allclose(
        masked.normalization["mean_dbm"],
        np.mean(
            masked.splits[masked.train_split].model_input
            * masked.normalization["std_dbm"]
            + masked.normalization["mean_dbm"],
            axis=0,
        ),
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        masked.splits[masked.train_split].model_input[:, 1:3],
        unmasked.splits[unmasked.train_split].model_input[:, 1:3],
    )
    assert not np.array_equal(
        masked.splits[masked.train_split].model_input[:, [0, 3]],
        unmasked.splits[unmasked.train_split].model_input[:, [0, 3]],
    )


def test_target_region_loss_runs_through_training_and_validation():
    class TinyForecaster(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, values):
            return self.layer(values[:, -1]).unsqueeze(1)

    rng = np.random.default_rng(4)
    train = rng.normal(size=(30, 4)).astype(np.float32)
    validation = rng.normal(size=(12, 4)).astype(np.float32)
    config = {
        "data": {"loss_frequency_ranges": [[101.5, 101.5]]},
        "windowing": {"lookback": 2, "horizons": [1, 2]},
        "training": {"device": "cpu"},
        "data_loader": {"num_workers": 0},
        "tiny": {
            "model": {"input_sequence_length": 2, "prediction_horizon": 1},
            "train": {
                "batch_size": 4,
                "epochs": 1,
                "val_fraction": 0.2,
                "train_stride": 1,
                "val_stride": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "optimizer": "adam",
                "early_stopping": False,
                "selection_metric": "val_mean_horizon_mae_db",
                "seed": 4,
            },
        },
    }
    _, results = train_model(
        "tiny",
        TinyForecaster(),
        train,
        config,
        val_data=validation,
        frequencies=[100.5, 101.5, 102.5, 103.5],
        normalization={"std_dbm": np.ones(4, dtype=np.float32)},
    )
    assert np.isfinite(results["best_selection_value"])
