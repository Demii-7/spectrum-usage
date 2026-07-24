import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

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
