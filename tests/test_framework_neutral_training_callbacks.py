from pathlib import Path
from types import SimpleNamespace

import pytest

from training.common import specialized_models
from training.common.training_events import emit_training_event


def test_event_is_a_plain_mapping_and_callback_errors_propagate():
    events = []
    emit_training_event(
        events.append,
        model_name="example",
        chunk_id="chunk-1",
        stage="train",
        epoch=2,
        epochs=4,
        metrics={"val_loss": 0.25},
        selection_metric="val_loss",
        selection_mode="min",
        duration=1.5,
        prunable=True,
        is_best=True,
    )

    assert type(events[0]) is dict
    assert events[0] == {
        "model_name": "example",
        "chunk_id": "chunk-1",
        "stage": "train",
        "epoch": 2,
        "epochs": 4,
        "metrics": {"val_loss": 0.25},
        "selection_metric": "val_loss",
        "selection_mode": "min",
        "selection_value": 0.25,
        "is_best": True,
        "duration": 1.5,
        "prunable": True,
        "selection_eligible": True,
    }

    def fail(_event):
        raise RuntimeError("stop training")

    with pytest.raises(RuntimeError, match="stop training"):
        emit_training_event(
            fail,
            model_name="example",
            stage="train",
            epoch=1,
            epochs=1,
            metrics={"loss": 1.0},
            selection_metric="loss",
            selection_mode="min",
            duration=0.0,
            prunable=True,
        )


def test_convlstm_fm_pretraining_reports_each_epoch():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from models.ConvLSTM_FM import ConvLSTMFMForecaster, pretrain_backbone

    model = ConvLSTMFMForecaster({
        "model": {
            "input_channels": 1,
            "grid_height": 2,
            "grid_width": 2,
            "prediction_horizon": 1,
            "hidden_channels": [2],
            "num_layers": 1,
            "kernel_size": [[3, 3]],
        }
    })
    x = torch.randn(2, 2, 1, 2, 2)
    loader = DataLoader(TensorDataset(x, torch.zeros(2, 1)), batch_size=2)
    events = []

    pretrain_backbone(
        model, loader, epochs=1, mask_ratio=0.5,
        callback=events.append, chunk_id="maps",
    )

    assert len(events) == 1
    assert events[0]["stage"] == "pretrain"
    assert events[0]["chunk_id"] == "maps"
    assert events[0]["prunable"] is False
    assert "masked_reconstruction_loss" in events[0]["metrics"]


def test_specialized_dispatch_threads_callback_and_explicit_validation(monkeypatch, tmp_path):
    callback = lambda event: None
    calls = []

    def train_one_model(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        specialized_models,
        "_load_module",
        lambda model_name, phase: SimpleNamespace(train_one_model=train_one_model),
    )
    train_split = SimpleNamespace(model_input="train", segments=("train-segment",))
    validation_split = SimpleNamespace(model_input="validation", segments=("val-segment",))
    data = SimpleNamespace(
        splits={"train": train_split, "validation": validation_split},
        train_split="train",
        validation_split="validation",
        frequencies="frequencies",
        normalization="normalization",
    )
    chunk = SimpleNamespace(chunk_id="chunk")

    specialized_models.train_specialized_chunk(
        "stsprednet", {}, chunk, data, tmp_path, tmp_path, callback=callback,
    )

    assert calls[0][1]["callback"] is callback
    assert calls[0][1]["validation_data"] == "validation"
    assert calls[0][1]["validation_segments"] == ("val-segment",)


def test_tss_stage_reports_global_and_local_epoch(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    module = specialized_models._load_module("tss_lcd", "train")
    enc = torch.nn.Linear(1, 1)
    dec = torch.nn.Linear(1, 1)
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 1), torch.ones(2, 1)), batch_size=2
    )
    events = []
    config = {
        "autoencoder_epochs": 1,
        "autoencoder_learning_rate": 0.001,
        "early_stopping_patience": 2,
    }

    module.train_autoencoder(
        enc, dec, loader, loader, config, torch.device("cpu"),
        Path(tmp_path), Path(tmp_path), "chunk", callback=events.append,
        epoch_offset=5, total_epochs=9,
    )

    assert events[0]["stage"] == "autoencoder"
    assert events[0]["epoch"] == 6
    assert events[0]["epochs"] == 9
    assert events[0]["stage_epoch"] == 1
    assert events[0]["stage_epochs"] == 1
