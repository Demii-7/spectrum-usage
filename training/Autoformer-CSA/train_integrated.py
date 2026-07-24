"""Train Autoformer-CSA through the shared chunk-based data pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import AutoformerCSAForecaster, DotConfig  # noqa: E402
from training.common.config import load_config, unique_run_dir  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.data_loader import data_loader_kwargs  # noqa: E402
from training.common.forecasting import forecast  # noqa: E402
from training.common.results import prepare_output_dirs  # noqa: E402
from training.common.runtime import device_for, epoch_log_row, timestamp_utc  # noqa: E402
from training.common.windowing import build_window_loaders, make_window_starts  # noqa: E402
from training.common.preprocessing import SequenceSegment  # noqa: E402


MODEL_NAME = "autoformer_csa"


class MapWindowDataset(Dataset):
    """Windows from independent grid-cell streams shaped (G, T, F)."""

    def __init__(self, streams: np.ndarray, starts: list[tuple[int, int]], lookback: int, horizon: int):
        self.data = torch.from_numpy(streams).float()
        self.starts = starts
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        grid_index, start = self.starts[index]
        end = start + self.lookback
        return (
            self.data[grid_index, start:end],
            self.data[grid_index, end : end + self.horizon],
        )


def _map_streams(data: np.ndarray) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expected map data shaped (T,H,W,F), got {data.shape}")
    time, height, width, features = data.shape
    return np.transpose(data, (1, 2, 0, 3)).reshape(height * width, time, features)


def _model_config(config: dict[str, Any], n_features: int) -> DotConfig:
    autoformer = config[MODEL_NAME]
    model = dict(autoformer["model"])
    lookback = int(autoformer.get("seq_len", config["windowing"]["lookback"]))
    prediction_horizon = int(autoformer.get("pred_len", max(config["windowing"]["horizons"])))
    label_len = int(autoformer.get("label_len", lookback // 2))
    if not 0 < label_len <= lookback:
        raise ValueError("autoformer_csa.label_len must be between 1 and seq_len")
    if prediction_horizon != max(int(h) for h in config["windowing"]["horizons"]):
        raise ValueError(
            "autoformer_csa.pred_len must equal max(windowing.horizons) "
            "for direct multi-step evaluation"
        )
    return DotConfig(
        seq_len=lookback,
        label_len=label_len,
        pred_len=prediction_horizon,
        enc_in=n_features,
        dec_in=n_features,
        c_out=n_features,
        d_model=int(model["d_model"]),
        d_ff=int(model.get("d_ff", 4 * int(model["d_model"]))),
        e_layers=int(model.get("encoder_layers", model.get("e_layers", 2))),
        d_layers=int(model.get("decoder_layers", model.get("d_layers", 1))),
        n_heads=int(model["n_heads"]),
        moving_avg=int(model["moving_avg"]),
        dropout=float(model.get("dropout", 0.1)),
        factor=int(model.get("factor", 3)),
        output_attention=bool(model.get("output_attention", False)),
        csam_kernel_size=int(model.get("csam_kernel_size", 7)),
    )


def _build_model(config: dict[str, Any], n_features: int, device: torch.device) -> tuple[nn.Module, DotConfig]:
    model_config = _model_config(config, n_features)
    model = AutoformerCSAForecaster(model_config).to(device)
    return model, model_config


def _checkpoint_model_config(model_config: DotConfig) -> dict[str, dict[str, Any]]:
    return {
        "windowing": {
            "seq_len": model_config.seq_len,
            "label_len": model_config.label_len,
            "pred_len": model_config.pred_len,
        },
        "model": {
            "enc_in": model_config.enc_in,
            "dec_in": model_config.dec_in,
            "c_out": model_config.c_out,
            "d_model": model_config.d_model,
            "d_ff": model_config.d_ff,
            "encoder_layers": model_config.e_layers,
            "decoder_layers": model_config.d_layers,
            "n_heads": model_config.n_heads,
            "moving_avg": model_config.moving_avg,
            "dropout": model_config.dropout,
            "factor": model_config.factor,
            "csam_kernel_size": model_config.csam_kernel_size,
            "output_attention": model_config.output_attention,
        },
    }


def _loss_function(name: str) -> nn.Module | Any:
    normalized = str(name).lower()
    if normalized == "mse":
        return nn.MSELoss()
    if normalized == "mae":
        return nn.L1Loss()
    if normalized == "rmse":
        return lambda prediction, target: torch.sqrt(torch.mean((prediction - target) ** 2) + 1e-12)
    raise ValueError(f"Unsupported Autoformer-CSA loss: {name!r}")


def _optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    autoformer = config[MODEL_NAME]
    learning_rate = float(autoformer.get("learning_rate", 1e-4))
    weight_decay = float(autoformer.get("weight_decay", 0.0))
    name = str(autoformer.get("optimizer", "adam")).lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "nadam":
        return torch.optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported Autoformer-CSA optimizer: {name!r}")


def _scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer):
    autoformer = config[MODEL_NAME]
    name = str(autoformer.get("lr_scheduler", "none")).lower()
    if name == "none":
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(autoformer.get("lr_factor", 0.5)),
            patience=int(autoformer.get("lr_patience", 5)),
        )
    raise ValueError(f"Unsupported Autoformer-CSA lr_scheduler: {name!r}")


def _make_loaders(
    config: dict[str, Any],
    data: np.ndarray,
    segments,
):
    autoformer = config[MODEL_NAME]
    lookback = int(autoformer.get("seq_len", config["windowing"]["lookback"]))
    horizon = int(autoformer.get("pred_len", max(config["windowing"]["horizons"])))
    val_fraction = float(autoformer.get("val_fraction", 0.1))
    if data.ndim == 4:
        streams = _map_streams(data)
        split_index = int(streams.shape[1] * (1.0 - val_fraction))
        train_streams = streams[:, :split_index]
        val_streams = streams[:, split_index:]
        train_segments = tuple(
            SequenceSegment(segment.start, min(segment.end, split_index), segment.label)
            for segment in segments
            if segment.start < split_index and segment.start < min(segment.end, split_index)
        )
        val_segments = tuple(
            SequenceSegment(
                max(segment.start, split_index) - split_index,
                segment.end - split_index,
                segment.label,
            )
            for segment in segments
            if segment.end > split_index and max(segment.start, split_index) < segment.end
        )
        train_starts = make_window_starts(
            train_streams.shape[1], lookback, horizon,
            int(autoformer.get("train_stride", 1)), train_segments,
        )
        val_starts = make_window_starts(
            val_streams.shape[1], lookback, horizon,
            int(autoformer.get("val_stride", 1)), val_segments,
        )
        train_indices = [
            (grid, int(start))
            for grid in range(train_streams.shape[0])
            for start in train_starts
        ]
        val_indices = [
            (grid, int(start))
            for grid in range(val_streams.shape[0])
            for start in val_starts
        ]
        loader_config = data_loader_kwargs(config.get("data_loader"))
        return (
            DataLoader(
                MapWindowDataset(train_streams, train_indices, lookback, horizon),
                batch_size=int(autoformer.get("batch_size", 32)),
                shuffle=True,
                **loader_config,
            ),
            DataLoader(
                MapWindowDataset(val_streams, val_indices, lookback, horizon),
                batch_size=int(autoformer.get("batch_size", 32)),
                shuffle=False,
                **loader_config,
            ),
        )
    return build_window_loaders(
        data=data,
        lookback=lookback,
        rollout_horizon=horizon,
        batch_size=int(autoformer.get("batch_size", 32)),
        val_fraction=float(autoformer.get("val_fraction", 0.1)),
        train_stride=int(autoformer.get("train_stride", 1)),
        val_stride=int(autoformer.get("val_stride", 1)),
        segments=segments,
        data_loader_config=config.get("data_loader"),
    )


def _run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn,
    optimizer: torch.optim.Optimizer | None,
    clip_norm: float,
    prediction_horizon: int,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    samples = 0
    for x, target in loader:
        x = x.to(device)
        target = target.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        prediction = forecast(
            model=model,
            x=x,
            prediction_horizon=prediction_horizon,
            rollout_horizon=prediction_horizon,
            targets=target,
        )
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Autoformer-CSA output {tuple(prediction.shape)} does not match "
                f"target {tuple(target.shape)}"
            )
        loss = loss_fn(prediction, target)
        if optimizer is not None:
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        count = x.shape[0]
        total += float(loss.item()) * count
        samples += count
    return total / max(samples, 1)


def train_chunk(
    config: dict[str, Any],
    chunk,
    data,
    out: Path,
    checkpoints: Path,
) -> None:
    autoformer = config[MODEL_NAME]
    train_split = data.splits[data.train_split]
    device = device_for(config)
    model, model_config = _build_model(config, train_split.model_input.shape[-1], device)
    train_loader, val_loader = _make_loaders(config, train_split.model_input, train_split.segments)

    optimizer = _optimizer(config, model)
    scheduler = _scheduler(config, optimizer)
    loss_fn = _loss_function(autoformer.get("loss", "rmse"))
    epochs = int(autoformer.get("epochs", 20))
    patience = int(autoformer.get("patience", autoformer.get("early_stopping_patience", 6)))
    clip_norm = float(autoformer.get("gradient_clip", autoformer.get("gradient_clip_norm", 5.0)))
    early_stopping = bool(autoformer.get("early_stopping", True))

    seed = int(autoformer.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    log_rows: list[dict[str, Any]] = []
    started_at = timestamp_utc()
    started_counter = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started_at = timestamp_utc()
        epoch_counter = time.perf_counter()
        train_loss = _run_epoch(
            model, train_loader, device, loss_fn, optimizer, clip_norm, model_config.pred_len
        )
        with torch.no_grad():
            val_loss = _run_epoch(
                model, val_loader, device, loss_fn, None, 0.0, model_config.pred_len
            )
        if scheduler is not None:
            scheduler.step(val_loss)
        duration = time.perf_counter() - epoch_counter
        log_rows.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_start_time=epoch_started_at,
                epoch_end_time=timestamp_utc(),
                epoch_duration_sec=duration,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        print(
            f"{chunk.chunk_id} epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={duration:.1f}s"
        )

        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if early_stopping and no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Autoformer-CSA training finished without a finite validation result")
    model.load_state_dict(best_state)
    total_duration = time.perf_counter() - started_counter
    log_frame = pd.DataFrame(log_rows)
    log_frame.to_csv(out / f"{chunk.chunk_id}_training_log.csv", index=False)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "model_config": _checkpoint_model_config(model_config),
            "normalization": data.normalization,
            "frequencies": data.frequencies,
            "data_shape": list(train_split.model_input.shape),
            "common_config": config,
            "training_results": {
                "best_epoch": best_epoch,
                "best_val_loss": float(best_loss),
                "epochs_completed": len(log_rows),
                "training_start_time": started_at,
                "training_end_time": timestamp_utc(),
                "training_duration_sec": total_duration,
            },
        },
        checkpoints / f"{chunk.chunk_id}_{MODEL_NAME}.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Autoformer-CSA")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if MODEL_NAME not in config:
        raise ValueError("Configuration is missing the autoformer_csa section")

    if args.output_dir is not None:
        run_dir = args.output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        name = args.name or f"{MODEL_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir = unique_run_dir(Path("runs") / name)
    out, checkpoints = prepare_output_dirs(run_dir)
    config_source = args.config or ROOT / "training" / "common" / "config.yaml"
    shutil.copy2(config_source, run_dir / "config.yaml")

    val_fraction = float(config[MODEL_NAME].get("val_fraction", 0.1))
    for chunk in chunk_specs(config):
        print(
            f"Training Autoformer-CSA for {chunk.chunk_id} "
            f"({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)"
        )
        data = load_chunk(config, chunk, val_fraction=val_fraction)
        train_chunk(config, chunk, data, out, checkpoints)
    print(f"Training run written to {run_dir}")


if __name__ == "__main__":
    main()
