import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

import models.ConvLSTM_FM as convlstm_fm
from models.ConvLSTM_FM import (
    ConvLSTMFMForecaster,
    masked_reconstruction_loss,
    pretrain_backbone,
)
from training.common.model_factory import build_model
from training.common.preprocessing import SequenceSegment
from training.common.train_integrated import train_model
from training.common.windowing import build_window_loaders


def config(pretrain_epochs=0):
    return {
        "convlstmfm": {
            "model": {
                "input_sequence_length": 3,
                "prediction_horizon": 1,
                "hidden_channels": [2],
                "num_layers": 1,
                "kernel_size": [[3, 3]],
                "pretrain_epochs": pretrain_epochs,
            }
        }
    }


def model():
    return ConvLSTMFMForecaster(
        {"model": {**config()["convlstmfm"]["model"], "input_channels": 1, "grid_height": 2, "grid_width": 2}}
    )


def test_conv3d_head_preserves_reconstruction_and_forecast_shapes():
    network = model()
    x = torch.randn(2, 3, 1, 2, 2)
    assert isinstance(network.head, torch.nn.Conv3d)
    assert network.reconstruct(x).shape == x.shape
    assert network(x).shape == (2, 1, 1, 2, 2)


def test_forecast_head_receives_full_sequence_and_selects_newest_output(monkeypatch):
    network = model()
    encoded = torch.zeros(2, 3, 2, 2, 2)

    class RecordingHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_shape = None

        def forward(self, sequence):
            self.input_shape = tuple(sequence.shape)
            outputs = torch.arange(sequence.shape[2], dtype=sequence.dtype).view(1, 1, -1, 1, 1)
            return outputs.expand(sequence.shape[0], 1, -1, sequence.shape[3], sequence.shape[4])

    head = RecordingHead()
    network.head = head
    monkeypatch.setattr(network, "_encode_sequence", lambda x: encoded)

    prediction = network(torch.zeros(2, 3, 1, 2, 2))

    assert head.input_shape == (2, 2, 3, 2, 2)
    assert prediction.shape == (2, 1, 1, 2, 2)
    assert torch.all(prediction == 2)


def test_masked_loss_is_mean_over_masked_scalars(monkeypatch):
    network = model()
    monkeypatch.setattr(network, "reconstruct", lambda x: torch.zeros_like(x))
    x = torch.ones(1, 2, 1, 2, 2)
    assert masked_reconstruction_loss(network, x, 1.0).item() == pytest.approx(1.0)


def test_pretraining_uses_loader_and_returns_serializable_metadata():
    network = model()
    x = torch.randn(4, 3, 1, 2, 2)
    loader = DataLoader(TensorDataset(x, torch.zeros(4, 1)), batch_size=2)
    metadata = pretrain_backbone(network, loader, epochs=1, mask_ratio=0.5)
    assert metadata["samples"] == 4
    assert len(metadata["epoch_losses"]) == 1
    json.dumps(metadata)


def test_pretraining_honors_model_specific_batch_size(monkeypatch):
    network = model()
    network.config["model"]["pretrain_batch_size"] = 3
    x = torch.randn(4, 3, 1, 2, 2)
    loader = DataLoader(TensorDataset(x, torch.zeros(4, 1)), batch_size=2)
    observed_batch_sizes = []

    def fake_loss(model, batch, mask_ratio, mask_mode):
        observed_batch_sizes.append(batch.shape[0])
        return next(model.parameters()).sum() * 0

    monkeypatch.setattr(convlstm_fm, "masked_reconstruction_loss", fake_loss)
    metadata = pretrain_backbone(network, loader, epochs=1)

    assert metadata["batch_size"] == 3
    assert sorted(observed_batch_sizes) == [1, 3]


def test_pretraining_loader_windows_do_not_cross_segments():
    data = np.concatenate([
        np.zeros((6, 2, 2, 1), dtype=np.float32),
        np.ones((6, 2, 2, 1), dtype=np.float32),
    ])
    segments = (
        SequenceSegment(0, 6, "first"),
        SequenceSegment(6, 12, "second"),
    )
    loader, _ = build_window_loaders(
        data=data,
        lookback=3,
        rollout_horizon=1,
        batch_size=2,
        val_fraction=0.25,
        train_stride=1,
        val_stride=1,
        segments=segments,
        val_data=data,
        val_segments=segments,
    )
    for x, _ in loader:
        for window in x:
            assert torch.all(window == 0) or torch.all(window == 1)


def test_factory_has_no_pretraining_side_effect_and_validates_freeze_conflict():
    data = torch.zeros(6, 2, 2, 1).numpy()
    network = build_model("convlstmfm", config(pretrain_epochs=1), data)
    assert all(parameter.requires_grad for parameter in network.encoder.parameters())

    conflicting = config(pretrain_epochs=1)
    conflicting["convlstmfm"]["model"]["freeze_backbone"] = True
    with pytest.raises(ValueError, match="freeze_backbone conflicts"):
        build_model("convlstmfm", conflicting, data)


def test_shared_trainer_runs_pretraining_then_freezes_backbone():
    settings = config(pretrain_epochs=1)
    settings.update({
        "windowing": {"horizons": [1]},
        "training": {"device": "cpu"},
    })
    model_cfg = settings["convlstmfm"]["model"]
    model_cfg.update({
        "pretrain_mask_ratio": 0.5,
        "freeze_backbone_after_pretrain": True,
    })
    settings["convlstmfm"]["train"] = {
        "batch_size": 2,
        "epochs": 1,
        "val_fraction": 0.2,
        "train_stride": 1,
        "val_stride": 1,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "validation_prediction_magnitude_threshold": 20.0,
        "early_stopping": False,
    }
    train = np.random.default_rng(1).normal(size=(8, 2, 2, 1)).astype(np.float32)
    validation = np.random.default_rng(2).normal(size=(5, 2, 2, 1)).astype(np.float32)
    network = build_model("convlstmfm", settings, train)
    network, results = train_model(
        "convlstmfm",
        network,
        train,
        settings,
        val_data=validation,
    )
    assert results["pretraining"]["epochs"] == 1
    assert all(not parameter.requires_grad for parameter in network.encoder.parameters())
    assert all(parameter.requires_grad for parameter in network.head.parameters())


def test_convlstm_fm_rejects_exploding_validation_epoch():
    settings = config(pretrain_epochs=0)
    settings.update({
        "windowing": {"horizons": [1]},
        "training": {"device": "cpu"},
    })
    settings["convlstmfm"]["train"] = {
        "batch_size": 2,
        "epochs": 1,
        "val_fraction": 0.2,
        "train_stride": 1,
        "val_stride": 1,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "validation_prediction_magnitude_threshold": 20.0,
        "early_stopping": False,
    }

    class ExplodingForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(100.0))

        def forward(self, x):
            return self.scale.expand(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])

    train = np.zeros((8, 2, 2, 1), dtype=np.float32)
    validation = np.zeros((5, 2, 2, 1), dtype=np.float32)
    with pytest.raises(RuntimeError, match="without an eligible validation epoch"):
        train_model(
            "convlstmfm",
            ExplodingForecaster(),
            train,
            settings,
            val_data=validation,
        )


def test_scratch_config_uses_same_architecture_without_pretraining():
    root = Path(__file__).parents[1]
    pretrained = yaml.safe_load((root / "training/configs/config_convlstm_fm.yaml").read_text())
    scratch = yaml.safe_load((root / "training/configs/config_convlstm_fm_scratch.yaml").read_text())
    pretrained_model = pretrained["convlstmfm"]["model"]
    scratch_model = scratch["convlstmfm"]["model"]

    for key in ("hidden_channels", "kernel_size", "num_layers", "cell_activation", "dropout"):
        assert scratch_model[key] == pretrained_model[key]
    assert scratch_model["pretrain_epochs"] == 0
    assert scratch_model["freeze_backbone_after_pretrain"] is False
