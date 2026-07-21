from __future__ import annotations

import argparse
from pathlib import Path
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import ConvLSTMPredictor  # noqa: E402
from training.common.config import load_config  # noqa: E402
from training.common.integrated import epoch_log_row, prepare_output_dirs, timestamp_utc  # noqa: E402
from training.common.windowing import make_window_starts, selected_horizon_index  # noqa: E402
from training.common.interpolated_map import (  # noqa: E402
    load_interpolated_map_npz,
    normalize_map_by_frequency,
)
from training.common.data import (
    chunk_specs,
    clean_interpolated_map,
    load_chunk,
    model_matrix_to_convlstm_frames,
)  # noqa: E402


MODEL_NAME = "convlstm"


class ConvLSTMWindowDataset(Dataset):
    def __init__(self, frames: np.ndarray, lookback: int, prediction_horizon: int, origins: np.ndarray):
        self.frames = torch.from_numpy(frames).float()
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.origins = origins.astype(np.int64)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int):
        origin = int(self.origins[idx])
        x = self.frames[origin - self.lookback + 1 : origin + 1].unsqueeze(1)
        y = self.frames[origin + 1 : origin + self.prediction_horizon + 1].unsqueeze(1)
        return x, y


class _MapWindowDataset(Dataset):
    """Dataset for interpolated-map mode: yields 4D windows (F, H, W)."""

    def __init__(self, data_4d: np.ndarray, lookback: int, prediction_horizon: int, origins: np.ndarray):
        self.data = torch.from_numpy(data_4d).float()
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.origins = origins.astype(np.int64)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int):
        origin = int(self.origins[idx])
        x = self.data[origin - self.lookback : origin]
        y = self.data[origin : origin + self.prediction_horizon]
        return x, y


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_map_for_path(path: str | Path, map_key: str) -> tuple[np.ndarray, dict[str, Any]]:
    data, metadata = load_interpolated_map_npz(path, map_key)
    print(f"[load_map_for_path] Loaded map: shape {data.shape}")
    data = clean_interpolated_map(data, train_ratio=0.8, fit_on_train_only=True)
    return data, metadata


def build_model_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    ccfg = config["convlstm"]
    reference_site = str(config["data"].get("reference_site", "CC2"))
    return {
        "data": {"n_nodes": 1, "n_bins_per_node": n_bins, "node_names": [reference_site]},
        "windowing": {
            "input_sequence_length": int(ccfg.get("input_sequence_length", config["windowing"]["lookback"])),
            "prediction_horizon": int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"]))),
        },
        "model": ccfg["model"],
    }


def build_map_model_config(config: dict[str, Any], n_freq: int, grid_h: int, grid_w: int) -> dict[str, Any]:
    """Build a model config for interpolated-map mode.

    Overrides data dimensions with map-specific values and sets
    ``input_channels`` to ``n_freq``.
    """
    ccfg = config["convlstm"]
    model_cfg = dict(ccfg["model"])
    model_cfg["input_channels"] = n_freq
    return {
        "data": {
            "n_nodes": 1,
            "n_bins_per_node": 1,
            "node_names": ["map"],
            "grid_height": grid_h,
            "grid_width": grid_w,
            "n_freq_bins": n_freq,
        },
        "windowing": {
            "input_sequence_length": int(ccfg.get("input_sequence_length", config["windowing"]["lookback"])),
            "prediction_horizon": int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"]))),
        },
        "model": model_cfg,
    }


def run_map_mode(config: dict[str, Any], out: Path, checkpoints: Path) -> None:
    data_cfg = config["data"]
    ccfg = config["convlstm"]
    map_cfg = ccfg.get("interpolated_map", {})
    train_map_path = data_cfg.get("train_map_path") or map_cfg.get("map_path")
    test_map_path = data_cfg.get("test_map_path") or train_map_path
    map_key = str(data_cfg.get("map_key") or map_cfg.get("map_key", "map_db"))
    if not train_map_path:
        raise ValueError("Map mode requires data.train_map_path or convlstm.interpolated_map.map_path")

    train_raw, train_meta = load_map_for_path(train_map_path, map_key)
    train_x, _, norm_stats = normalize_map_by_frequency(
        train_raw, None,
        enabled=bool(config.get("preprocessing", {}).get("normalize", True)),
    )

    T, F, H, W = train_x.shape
    lookback = int(ccfg.get("input_sequence_length", config["windowing"]["lookback"]))
    prediction_horizon = int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"])))
    batch_size = int(ccfg.get("batch_size", 32))
    if prediction_horizon <= 0:
        raise ValueError(f"prediction_horizon must be > 0, got {prediction_horizon}")
    if lookback <= 0:
        raise ValueError(f"lookback must be > 0, got {lookback}")

    origins = np.arange(lookback, T - prediction_horizon + 1, dtype=np.int64)
    if len(origins) < 2:
        raise ValueError(f"Not enough map timesteps ({T}) for lookback={lookback}, horizon={prediction_horizon}")
    val_count = max(1, int(len(origins) * 0.1))
    train_origins = origins[:-val_count]
    val_origins = origins[-val_count:]
    if len(train_origins) == 0:
        raise ValueError("No training windows were created")
    if len(val_origins) == 0:
        raise ValueError("No validation windows were created")
    train_loader = DataLoader(
        _MapWindowDataset(train_x, lookback, prediction_horizon, train_origins),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        _MapWindowDataset(train_x, lookback, prediction_horizon, val_origins),
        batch_size=batch_size,
        shuffle=False,
    )

    model_config = build_map_model_config(config, F, H, W)
    device = device_for()
    model = ConvLSTMPredictor(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(ccfg.get("learning_rate", 0.0002)),
        weight_decay=float(ccfg.get("weight_decay", 0.004)),
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    log_rows: list[dict[str, Any]] = []
    epoch_times: list[float] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()
    epochs = int(ccfg.get("epochs", 25))
    tf_start = float(ccfg.get("teacher_forcing_ratio_start", ccfg.get("teacher_forcing_ratio", 0.8)))
    tf_end = float(ccfg.get("teacher_forcing_ratio_end", ccfg.get("teacher_forcing_ratio", 0.2)))
    clip_norm = float(ccfg.get("gradient_clip_norm", 5.0))
    patience_left = int(ccfg.get("early_stopping_patience", 8))
    for epoch in range(1, epochs + 1):
        teacher_forcing_ratio = tf_start + (tf_end - tf_start) * (epoch - 1) / max(epochs - 1, 1)
        epoch_start_time = timestamp_utc()
        t_epoch = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, y_teacher=y, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(pred, y)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            assert y.shape[1] == prediction_horizon, f"Expected target timesteps {prediction_horizon}, got {y.shape[1]}"
            train_loss_sum += loss.item() * x.size(0)
            train_sample_count += x.size(0)
        train_loss = train_loss_sum / max(train_sample_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                window = x
                all_preds = []
                for h in range(prediction_horizon):
                    output = model(window)
                    next_pred = output[:, selected_horizon_index(1), :, :, :]
                    all_preds.append(next_pred)
                    window = torch.cat([window[:, 1:, :, :, :], next_pred.unsqueeze(1)], dim=1)
                pred = torch.cat(all_preds, dim=1)
                val_loss_sum += criterion(pred, y).item() * x.size(0)
                val_sample_count += x.size(0)
        val_loss = val_loss_sum / max(val_sample_count, 1)
        t_epoch = time.perf_counter() - t_epoch
        epoch_end_time = timestamp_utc()
        epoch_times.append(t_epoch)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta = avg_time * (epochs - epoch)
        log_rows.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=epoch_end_time,
                epoch_duration_sec=t_epoch,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        print(f"map epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={t_epoch:.1f}s avg={avg_time:.1f}s eta={eta:.0f}s")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(ccfg.get("early_stopping_patience", 8))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping triggered at epoch {epoch} (best epoch {best_epoch}, best val_loss {best_loss:.6f})")
                break
    total_time = time.perf_counter() - t_start
    print(f"Interpolated-map training done in {total_time:.1f}s ({total_time/60:.1f} min), best epoch {best_epoch}, best val_loss {best_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(checkpoints / "interpolated_map_training_log.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "common_config": config,
            "normalization_stats": norm_stats,
            "train_map_metadata": train_meta,
            "training_start_time": training_start_time,
            "training_end_time": timestamp_utc(),
            "training_duration_sec": total_time,
            "best_val_loss": best_loss,
            "best_epoch": best_epoch,
        },
        checkpoints / "interpolated_map_convlstm.pt",
    )
    print(f"Interpolated-map model saved to {checkpoints / 'interpolated_map_convlstm.pt'}")


def train_one_model(config: dict[str, Any], train_matrix: np.ndarray, segments, checkpoints: Path, out: Path, chunk_id: str) -> ConvLSTMPredictor:
    ccfg = config["convlstm"]
    lookback = int(ccfg.get("input_sequence_length", config["windowing"]["lookback"]))
    prediction_horizon = int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"])))
    batch_size = int(ccfg.get("batch_size", 32))
    epochs = int(ccfg.get("epochs", 25))
    val_fraction = float(ccfg.get("val_fraction", 0.1))
    clip_norm = float(ccfg.get("gradient_clip_norm", 5.0))
    if prediction_horizon <= 0:
        raise ValueError(f"prediction_horizon must be > 0, got {prediction_horizon}")
    if lookback <= 0:
        raise ValueError(f"lookback must be > 0, got {lookback}")

    frames = model_matrix_to_convlstm_frames(train_matrix)
    starts = make_window_starts(
        len(frames), lookback, prediction_horizon, 1, segments
    )
    origins = starts + lookback - 1
    if len(origins) < 2:
        raise ValueError(f"Not enough training rows for lookback={lookback} and horizon={prediction_horizon}")
    val_count = max(1, int(len(origins) * val_fraction)) if val_fraction > 0 else 0
    train_origins = origins[:-val_count] if val_count else origins
    val_origins = origins[-val_count:] if val_count else origins[-1:]
    if len(train_origins) == 0:
        raise ValueError("No training windows were created")
    if len(val_origins) == 0:
        raise ValueError("No validation windows were created")

    train_loader = DataLoader(
        ConvLSTMWindowDataset(frames, lookback, prediction_horizon, train_origins),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        ConvLSTMWindowDataset(frames, lookback, prediction_horizon, val_origins),
        batch_size=batch_size,
        shuffle=False,
    )

    model_config = build_model_config(config, train_matrix.shape[1])
    device = device_for()
    model = ConvLSTMPredictor(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(ccfg.get("learning_rate", 0.0002)),
        weight_decay=float(ccfg.get("weight_decay", 0.004)),
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    log_rows = []
    epoch_times: list[float] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()
    tf_start = float(ccfg.get("teacher_forcing_ratio_start", ccfg.get("teacher_forcing_ratio", 0.8)))
    tf_end = float(ccfg.get("teacher_forcing_ratio_end", ccfg.get("teacher_forcing_ratio", 0.2)))
    patience_left = int(ccfg.get("early_stopping_patience", 8))
    for epoch in range(1, epochs + 1):
        teacher_forcing_ratio = tf_start + (tf_end - tf_start) * (epoch - 1) / max(epochs - 1, 1)
        epoch_start_time = timestamp_utc()
        t_epoch = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x, y_teacher=y, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(pred, y)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            assert y.shape[1] == prediction_horizon, f"Expected target timesteps {prediction_horizon}, got {y.shape[1]}"
            train_loss_sum += loss.item() * x.size(0)
            train_sample_count += x.size(0)
        train_loss = train_loss_sum / max(train_sample_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                window = x
                all_preds = []
                for h in range(prediction_horizon):
                    output = model(window)
                    next_pred = output[:, selected_horizon_index(1), :, :, :]
                    all_preds.append(next_pred)
                    window = torch.cat([window[:, 1:, :, :, :], next_pred.unsqueeze(1)], dim=1)
                pred = torch.cat(all_preds, dim=1)
                val_loss_sum += criterion(pred, y).item() * x.size(0)
                val_sample_count += x.size(0)
        val_loss = val_loss_sum / max(val_sample_count, 1)
        t_epoch = time.perf_counter() - t_epoch
        epoch_end_time = timestamp_utc()
        epoch_times.append(t_epoch)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta = avg_time * (epochs - epoch)
        log_rows.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=epoch_end_time,
                epoch_duration_sec=t_epoch,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        print(f"{chunk_id} epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={t_epoch:.1f}s avg={avg_time:.1f}s eta={eta:.0f}s")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(ccfg.get("early_stopping_patience", 8))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"{chunk_id} early stopping triggered at epoch {epoch} (best epoch {best_epoch}, best val_loss {best_loss:.6f})")
                break
    total_time = time.perf_counter() - t_start
    print(f"{chunk_id} training done in {total_time:.1f}s ({total_time/60:.1f} min), best epoch {best_epoch}, best val_loss {best_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_training_log.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "common_config": config,
            "training_start_time": training_start_time,
            "training_end_time": timestamp_utc(),
            "training_duration_sec": total_time,
            "best_val_loss": best_loss,
            "best_epoch": best_epoch,
        },
        checkpoints / f"{chunk_id}_convlstm.pt",
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out, checkpoints = prepare_output_dirs(config, "ConvLSTM")
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        checkpoints = out / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)

    if config["data"].get("train_map_path") or config["convlstm"].get("interpolated_map", {}).get("enabled", False):
        print("Interpolated-map mode enabled — training on map data.")
        run_map_mode(config, out, checkpoints)
        return

    for chunk in chunk_specs(config):
        print(f"Training ConvLSTM for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        data = load_chunk(config, chunk)
        train = data.splits[data.train_split].model_input
        train_one_model(
            config, train, data.splits[data.train_split].segments,
            checkpoints, out, chunk.chunk_id,
        )


if __name__ == "__main__":
    main()
