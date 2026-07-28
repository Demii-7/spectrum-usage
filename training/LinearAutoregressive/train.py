"""Train the LinearAutoregressive baseline on POWDER spectrum data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.LinearAutoregressive import LinearAutoregressiveForecaster
from training.common.forecasting import forecast
from training.common.preprocessing import (
    fit_per_frequency_normalization,
    apply_per_frequency_normalization,
)
from training.common.windowing import build_window_loaders
from training.common.runtime import device_for


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

import yaml


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def load_powder_csv(path: Path, start_mhz: float, end_mhz: float) -> tuple[np.ndarray, list[float]]:
    df = pd.read_csv(path)
    freq_cols = [c for c in df.columns[1:] if start_mhz <= float(c) <= end_mhz]
    freq_cols.sort(key=float)
    arr = df[freq_cols].to_numpy(dtype=np.float32)
    return arr, [float(c) for c in freq_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LinearAutoregressive baseline.")
    parser.add_argument("--mode", choices=["1d", "2d", "4d"], required=True)
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    parser.add_argument("--freq-index", type=int, default=100, help="Frequency bin index for 1D mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    device = device_for(config)
    print(f"Device: {device}")

    lookback = int(config["model"]["input_sequence_length"])
    rollout_horizon = 1
    train_cfg = config["train"]
    model_cfg = config["model"]

    start_mhz = 600.0
    end_mhz = 800.0
    val_fraction = float(train_cfg.get("val_fraction", 0.1))

    train_csv = ROOT / "evaluation" / "powder" / "guesthouse-nuc1" / "20260618T0036Z" / "600_800" / "power_1mhz_avg_per_minute.csv"
    test_csv = ROOT / "evaluation" / "powder" / "guesthouse-nuc1" / "20260628T0436Z" / "600_800" / "power_1mhz_avg_per_minute.csv"

    if args.mode == "4d":
        train_map = ROOT / "evaluation" / "results" / "idw" / "powder_20260618T0036Z_humanities_guesthouse_600_800.npz"
        test_map = ROOT / "evaluation" / "results" / "idw" / "powder_20260628T0436Z_humanities_guesthouse_600_800.npz"

        train_raw = np.load(str(train_map))["map_db"].astype(np.float32)
        test_raw = np.load(str(test_map))["map_db"].astype(np.float32)
        frequencies = np.load(str(train_map))["freqs_mhz"].tolist()

        freq_mask = np.array([(start_mhz <= f <= end_mhz) for f in frequencies])
        train_raw = train_raw[:, :, :, freq_mask]
        test_raw = test_raw[:, :, :, freq_mask]
        frequencies = [f for f, m in zip(frequencies, freq_mask) if m]

        fit_end = int(len(train_raw) * (1.0 - val_fraction))
        mean = np.mean(train_raw[:fit_end], axis=(0, 1, 2))
        std = np.std(train_raw[:fit_end], axis=(0, 1, 2))
        std[std < 1e-8] = 1.0
        train_data = apply_per_frequency_normalization(train_raw, mean, std).astype(np.float32)
        test_data = apply_per_frequency_normalization(test_raw, mean, std).astype(np.float32)

        print(f"4D mode: train={train_data.shape}, test={test_data.shape}")

    else:
        train_raw, freqs = load_powder_csv(train_csv, start_mhz, end_mhz)
        test_raw, _ = load_powder_csv(test_csv, start_mhz, end_mhz)
        frequencies = freqs

        if args.mode == "1d":
            idx = min(args.freq_index, train_raw.shape[1] - 1)
            train_raw = train_raw[:, idx:idx+1]
            test_raw = test_raw[:, idx:idx+1]
            print(f"1D mode: freq idx={idx}, {frequencies[idx]:.1f} MHz")

        fit_end = int(len(train_raw) * (1.0 - val_fraction))
        mean, std = fit_per_frequency_normalization(train_raw[:fit_end])
        train_data = apply_per_frequency_normalization(train_raw, mean, std).astype(np.float32)
        test_data = apply_per_frequency_normalization(test_raw, mean, std).astype(np.float32)

        print(f"{args.mode.upper()} mode: train={train_data.shape}, test={test_data.shape}")

    input_size = int(np.prod(train_data.shape[1:]))
    model_cfg["input_size"] = input_size

    build_cfg = {"model": dict(model_cfg)}
    model = LinearAutoregressiveForecaster(build_cfg).to(device)

    has_params = any(True for _ in model.parameters())
    criterion = nn.MSELoss()
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))
        if has_params else None
    )

    batch_size = int(train_cfg["batch_size"])
    train_loader, val_loader = build_window_loaders(
        data=train_data,
        lookback=lookback,
        rollout_horizon=rollout_horizon,
        batch_size=batch_size,
        val_fraction=val_fraction,
        train_stride=int(train_cfg["train_stride"]),
        val_stride=int(train_cfg["val_stride"]),
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    patience = int(train_cfg.get("early_stopping_patience", 10))
    do_early_stop = bool(train_cfg.get("early_stopping", True))
    epochs = int(train_cfg["epochs"])

    model_name = f"linearar{args.mode}"
    print(f"Training {model_name}")
    print(f"  Data: train={train_data.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters())}")

    t0 = time.perf_counter()
    training_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            pred = forecast(model, x, 1, rollout_horizon, targets=y)
            data_loss = criterion(pred, y)
            loss = data_loss + model.ridge_penalty()
            if has_params:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
                optimizer.step()
            train_loss_sum += data_loss.item() * x.size(0)
            train_count += x.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = forecast(model, x, 1, rollout_horizon, targets=None)
                val_loss_sum += criterion(pred, y).item() * x.size(0)
                val_count += x.size(0)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        training_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train={train_loss:.6f}  val={val_loss:.6f}")

        if do_early_stop and epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    ckpt_dir = ROOT / "training" / "results" / "baselines"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}.pt"
    torch.save({
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "normalization": {"mean_dbm": mean, "std_dbm": std},
        "frequencies": frequencies,
        "config": config,
        "best_val_loss": best_val_loss,
        "log": training_log,
    }, ckpt_path)

    training_log_path = ckpt_dir / f"{model_name}_training_log.json"
    with open(training_log_path, "w") as f:
        json.dump({"config": config, "best_val_loss": best_val_loss, "log": training_log}, f, indent=2)

    print(f"  Best val loss: {best_val_loss:.6f}  ({elapsed:.1f}s)")
    print(f"  Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
