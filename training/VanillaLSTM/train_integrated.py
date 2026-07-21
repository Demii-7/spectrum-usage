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

from model import VanillaLSTMForecaster  # noqa: E402
from training.common.config import load_config  # noqa: E402
from training.common.integrated import epoch_log_row, prepare_output_dirs, timestamp_utc  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.data_loader import data_loader_kwargs  # noqa: E402
from training.common.windowing import make_window_starts  # noqa: E402


MODEL_NAME = "vanillalstm"


class VanillaWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, starts: np.ndarray, lookback: int, prediction_horizon: int):
        self.data = torch.from_numpy(data).float()
        self.starts = starts.astype(np.int64)
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        x = self.data[start : start + self.lookback]
        y = self.data[start + self.lookback : start + self.lookback + self.prediction_horizon]
        return x, y

#Looks up the device parameter in config file to sue gpu compute if available
def device_for(config: dict[str, Any]) -> torch.device:
    requested = str(config["vanillalstm"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    vcfg = config["vanillalstm"]
    return {
        "windowing": {
            "input_sequence_length": int(vcfg.get("input_sequence_length", config["windowing"]["lookback"])),
            "prediction_horizon": int(vcfg.get("prediction_horizon", max(config["windowing"]["horizons"]))),
        },
        "model": {
            "input_size": n_bins,
            "hidden_size": int(vcfg.get("hidden_size", 128)),
            "num_layers": int(vcfg.get("num_layers", 1)),
            "dropout": float(vcfg.get("dropout", 0.0)),
            "output_strategy": str(vcfg.get("output_strategy", "final_hidden")),
            "bidirectional": bool(vcfg.get("bidirectional", False)),
        },
    }

def autoregressive_rollout(
    model: VanillaLSTMForecaster,
    x: torch.Tensor,
    prediction_horizon: int,
) -> torch.Tensor:
    """
    Autoregressive validation rollout.

    Starts from ground-truth lookback x.
    Then repeatedly:
      1. predicts one step ahead
      2. appends the prediction
      3. shifts the lookback window
    """
    preds = []
    window = x

    for _ in range(prediction_horizon):
        output = model(window)

        # Use only one-step-ahead prediction for autoregressive rollout
        next_pred = output[:, 0, :]

        preds.append(next_pred)

        # Shift lookback left and append prediction
        window = torch.cat(
            [window[:, 1:, :], next_pred.unsqueeze(1)],
            dim=1,
        )

    return torch.stack(preds, dim=1)
    
def train_one_model(config: dict[str, Any], train_matrix: np.ndarray, segments, checkpoints: Path, out: Path, chunk_id: str) -> VanillaLSTMForecaster:
    vcfg = config["vanillalstm"]
    lookback = int(vcfg.get("input_sequence_length", config["windowing"]["lookback"]))
    prediction_horizon = int(vcfg.get("prediction_horizon", max(config["windowing"]["horizons"])))
    batch_size = int(vcfg.get("batch_size", 32))
    epochs = int(vcfg.get("epochs", 20))
    val_fraction = 0.1
    clip_norm = float(vcfg.get("gradient_clip", 1.0))
    patience = int(vcfg.get("patience", 10))

    starts = make_window_starts(
        len(train_matrix), lookback, prediction_horizon, 1, segments
    )
    if len(starts) < 2:
        raise ValueError(f"Not enough training rows for lookback={lookback} and horizon={prediction_horizon}")
    val_count = max(1, int(len(starts) * val_fraction))
    train_starts = starts[:-val_count]
    val_starts = starts[-val_count:]

    train_loader = DataLoader(
        VanillaWindowDataset(train_matrix, train_starts, lookback, prediction_horizon),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **data_loader_kwargs(config.get("data_loader")),
    )
    val_loader = DataLoader(
        VanillaWindowDataset(train_matrix, val_starts, lookback, prediction_horizon),
        batch_size=batch_size,
        shuffle=False,
        **data_loader_kwargs(config.get("data_loader")),
    )

    model_config = build_model_config(config, train_matrix.shape[1])
    model = VanillaLSTMForecaster(model_config).to(device_for(config))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(vcfg.get("learning_rate", 0.001)))
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    log_rows: list[dict[str, Any]] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(next(model.parameters()).device)
            y = y.to(next(model.parameters()).device)
            optimizer.zero_grad()
            pred = model(x)
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
                x = x.to(next(model.parameters()).device)
                y = y.to(next(model.parameters()).device)
        
                pred = autoregressive_rollout(
                    model=model,
                    x=x,
                    prediction_horizon=prediction_horizon,
                )
        
                val_loss += criterion(pred, y).item() * x.size(0)
        
        val_loss /= max(len(val_loader.dataset), 1)

        epoch_duration = time.perf_counter() - epoch_start
        log_rows.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=timestamp_utc(),
                epoch_duration_sec=epoch_duration,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        print(f"{chunk_id} epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={epoch_duration:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if bool(vcfg.get("early_stopping", True)) and no_improve >= patience:
                break

    total_time = time.perf_counter() - t_start
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
        },
        checkpoints / f"{chunk_id}_vanillalstm.pt",
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
    out, checkpoints = prepare_output_dirs(config, "VanillaLSTM")
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        checkpoints = out / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)

    for chunk in chunk_specs(config):
        print(f"Training VanillaLSTM for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        data = load_chunk(config, chunk)
        train = data.splits[data.train_split].model_input
        train_one_model(
            config, train, data.splits[data.train_split].segments,
            checkpoints, out, chunk.chunk_id,
        )


if __name__ == "__main__":
    main()
