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
from training.common.data import (
    chunk_specs,
    clean_interpolated_map,
    load_chunk,
    model_matrix_to_convlstm_frames,
)  # noqa: E402
from training.common.metrics import absolute_and_squared_errors_dbm  # noqa: E402
from training.common.results import append_metric_rows, load_band_definitions, output_dir  # noqa: E402
from training.common.windowing import aligned_history_matrix, selected_horizon_index, target_rows_for  # noqa: E402


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


def load_map_for_training(config: dict[str, Any]) -> np.ndarray:
    """Load and clean an interpolated .npz map for training.

    Expects ``config.convlstm.interpolated_map`` section with keys
    ``map_path``, ``map_key``, ``n_freq_bins``, ``grid_height``, ``grid_width``.

    Returns:
        Cleaned array of shape (T, F, H, W) with no NaN values.
    """
    map_cfg = config["convlstm"]["interpolated_map"]
    map_path = map_cfg["map_path"]
    map_key = map_cfg.get("map_key", "map_db")
    data = np.load(map_path)[map_key].astype(np.float32)
    data = data.transpose(0, 3, 1, 2)
    print(f"[load_map_for_training] Loaded map: shape {data.shape}")
    data = clean_interpolated_map(data, train_ratio=0.8, fit_on_train_only=True)
    return data


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


def train_one_model(config: dict[str, Any], train_matrix: np.ndarray, out: Path, chunk_id: str) -> ConvLSTMPredictor:
    ccfg = config["convlstm"]
    lookback = int(ccfg.get("input_sequence_length", config["windowing"]["lookback"]))
    prediction_horizon = int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"])))
    batch_size = int(ccfg.get("batch_size", 32))
    epochs = int(ccfg.get("epochs", 25))
    val_fraction = float(ccfg.get("val_fraction", 0.1))
    teacher_forcing_ratio = float(ccfg.get("teacher_forcing_ratio", 1.0))
    clip_norm = float(ccfg.get("gradient_clip_norm", 5.0))

    frames = model_matrix_to_convlstm_frames(train_matrix)
    origins = np.arange(lookback - 1, len(frames) - prediction_horizon, dtype=np.int64)
    if len(origins) < 2:
        raise ValueError(f"Not enough training rows for lookback={lookback} and horizon={prediction_horizon}")
    val_count = max(1, int(len(origins) * val_fraction)) if val_fraction > 0 else 0
    train_origins = origins[:-val_count] if val_count else origins
    val_origins = origins[-val_count:] if val_count else origins[-1:]

    train_loader = DataLoader(
        ConvLSTMWindowDataset(frames, lookback, prediction_horizon, train_origins),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
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
    best_state = None
    log_rows = []
    epoch_times: list[float] = []
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        train_loss = 0.0
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
            train_loss += loss.item() * x.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= max(len(val_loader.dataset), 1)
        t_epoch = time.perf_counter() - t_epoch
        epoch_times.append(t_epoch)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta = avg_time * (epochs - epoch)
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "time_sec": t_epoch})
        print(f"{chunk_id} epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={t_epoch:.1f}s avg={avg_time:.1f}s eta={eta:.0f}s")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    total_time = time.perf_counter() - t_start
    print(f"{chunk_id} training done in {total_time:.1f}s ({total_time/60:.1f} min)")

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_training_log.csv", index=False)
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": model_config, "common_config": config},
        out / "models" / f"{chunk_id}_convlstm.pt",
    )
    return model


def predict_for_targets(
    model: ConvLSTMPredictor,
    full_x: np.ndarray,
    target_rows: np.ndarray,
    horizon: int,
    lookback: int,
    batch_size: int,
) -> np.ndarray:
    origins = target_rows - horizon
    histories = aligned_history_matrix(full_x, origins, horizon=0, lookback=lookback)
    histories = histories[:, :, None, None, :].astype(np.float32)
    loader = DataLoader(torch.from_numpy(histories).float(), batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    preds = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            pred = model(x.to(device))
            preds.append(pred[:, selected_horizon_index(horizon), 0, 0, :].cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands: pd.DataFrame, out: Path):
    ccfg = config["convlstm"]
    lookback = int(ccfg.get("input_sequence_length", config["windowing"]["lookback"]))
    min_history = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    batch_size = int(ccfg.get("batch_size", 32))
    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm
    model = train_one_model(config, train, out, chunk.chunk_id)

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        for split_name in test_splits:
            split = data.splits[split_name]
            full_x = np.vstack([train, split.model_input]).astype(np.float32)
            full_raw = np.vstack([train_raw, split.raw_dbm]).astype(np.float32)
            history_offset = len(train)
            target_rows = target_rows_for(len(split.raw_dbm), history_offset, horizon, lookback, min_history)
            pred = predict_for_targets(model, full_x, target_rows, horizon, lookback, batch_size)
            target = full_raw[target_rows]
            _, abs_err, sq_err = absolute_and_squared_errors_dbm(pred, target, data.normalization)
            append_metric_rows(
                aggregate_rows,
                frequency_rows,
                band_rows,
                chunk_id=chunk.chunk_id,
                start_mhz=chunk.start_mhz,
                end_mhz=chunk.end_mhz,
                split_name=split_name,
                horizon=horizon,
                model=MODEL_NAME,
                target_rows=target_rows,
                history_offset=history_offset,
                freqs=data.frequencies,
                abs_err=abs_err,
                sq_err=sq_err,
                bands=bands,
            )
    return aggregate_rows, frequency_rows, band_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out = args.output_dir or output_dir(config, "ConvLSTM")
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)

    # Check for interpolated-map mode.
    map_cfg = config["convlstm"].get("interpolated_map", {})
    if map_cfg.get("enabled", False):
        print("Interpolated-map mode enabled — training on map data.")
        data_4d = load_map_for_training(config)
        T, F, H, W = data_4d.shape
        ccfg = config["convlstm"]
        lookback = int(ccfg.get("input_sequence_length", config["windowing"]["lookback"]))
        prediction_horizon = int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"])))
        # Build 4D windows for training.
        origins = np.arange(lookback, T - prediction_horizon, dtype=np.int64)
        if len(origins) < 2:
            raise ValueError(f"Not enough map timesteps ({T}) for lookback={lookback}, horizon={prediction_horizon}")
        val_count = max(1, int(len(origins) * 0.1))
        train_loader = DataLoader(
            _MapWindowDataset(data_4d, lookback, prediction_horizon, origins[:-val_count]),
            batch_size=int(ccfg.get("batch_size", 32)),
            shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            _MapWindowDataset(data_4d, lookback, prediction_horizon, origins[-val_count:]),
            batch_size=int(ccfg.get("batch_size", 32)),
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
        for epoch in range(1, int(ccfg.get("epochs", 25)) + 1):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x, y_teacher=y, teacher_forcing_ratio=float(ccfg.get("teacher_forcing_ratio", 1.0)))
                loss = criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(ccfg.get("gradient_clip_norm", 5.0)))
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= max(len(train_loader.dataset), 1)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    val_loss += criterion(pred, y).item() * x.size(0)
            val_loss /= max(len(val_loader.dataset), 1)
            print(f"map epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        torch.save(
            {"model_state_dict": model.state_dict(), "model_config": model_config, "common_config": config},
            out / "models" / "interpolated_map_convlstm.pt",
        )
        print(f"Interpolated-map model saved to {out / 'models' / 'interpolated_map_convlstm.pt'}")
        return

    bands = load_band_definitions(config)
    total_start = time.perf_counter()
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for chunk in chunk_specs(config):
        print(f"Training ConvLSTM for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        chunk_start = time.perf_counter()
        aggregate, frequency, band = evaluate_chunk(config, chunk, bands, out)
        print(f"  {chunk.chunk_id} total done in {time.perf_counter() - chunk_start:.1f}s")
        aggregate_rows.extend(aggregate)
        frequency_rows.extend(frequency)
        band_rows.extend(band)

    pd.DataFrame(aggregate_rows).to_csv(out / "aggregate_metrics.csv", index=False)
    pd.DataFrame(frequency_rows).to_csv(out / "per_frequency_metrics.csv", index=False)
    pd.DataFrame(band_rows).to_csv(out / "per_band_metrics.csv", index=False)
    total_run = time.perf_counter() - total_start
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")
    print(f"Total run time: {total_run:.1f}s ({total_run/60:.1f} min)")


if __name__ == "__main__":
    main()
