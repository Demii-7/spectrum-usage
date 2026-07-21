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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import SequenceDataset  # noqa: E402
from train import build_model  # noqa: E402
from training.common.config import load_config  # noqa: E402
from training.common.data_loader import data_loader_kwargs  # noqa: E402
from training.common.integrated import epoch_log_row, prepare_output_dirs, timestamp_utc  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.windowing import make_window_starts  # noqa: E402


MODEL_NAME = "autoformer_csa"


def device_for(config: dict[str, Any]) -> torch.device:
    requested = str(config["autoformer_csa"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_runner_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    acfg = config["autoformer_csa"]
    return {
        "windowing": {
            "seq_len": int(acfg.get("seq_len", config["windowing"]["lookback"])),
            "label_len": int(acfg.get("label_len", config["windowing"]["lookback"] // 2)),
            "pred_len": int(acfg.get("pred_len", max(config["windowing"]["horizons"]))),
        },
        "model": {
            "enc_in": n_bins,
            "dec_in": n_bins,
            "c_out": n_bins,
            **dict(acfg["model"]),
        },
        "architecture": dict(acfg.get("architecture", {"model_variant": "autoformer_csa"})),
        "training": {
            "learning_rate": float(acfg.get("learning_rate", 0.0005)),
        },
    }


def make_dataset(data_2d: np.ndarray, seq_len: int, label_len: int, pred_len: int, starts: np.ndarray) -> SequenceDataset:
    indices = [(0, int(start)) for start in starts]
    return SequenceDataset(data_2d[None, :, :].astype(np.float32), seq_len, label_len, pred_len, indices)


def run_forecast(model: nn.Module, seq_x: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
    dec_input = torch.zeros(seq_x.shape[0], label_len + pred_len, seq_x.shape[-1], device=seq_x.device)
    dec_input[:, :label_len, :] = seq_x[:, -label_len:, :]
    x_mark_enc = torch.zeros(seq_x.shape[0], seq_x.shape[1], 4, device=seq_x.device)
    x_mark_dec = torch.zeros(dec_input.shape[0], dec_input.shape[1], 4, device=seq_x.device)
    return model(seq_x, x_mark_enc, dec_input, x_mark_dec)


def train_one_model(config: dict[str, Any], train_matrix: np.ndarray, segments, checkpoints: Path, out: Path, chunk_id: str):
    acfg = config["autoformer_csa"]
    seq_len = int(acfg.get("seq_len", config["windowing"]["lookback"]))
    label_len = int(acfg.get("label_len", seq_len // 2))
    pred_len = int(acfg.get("pred_len", max(config["windowing"]["horizons"])))
    batch_size = int(acfg.get("batch_size", 8))
    epochs = int(acfg.get("epochs", 10))
    patience = int(acfg.get("patience", 6))
    clip = float(acfg.get("gradient_clip", 5.0))

    starts = make_window_starts(
        len(train_matrix), seq_len, pred_len,
        int(acfg.get("train_stride", 1)), segments,
    )
    if len(starts) < 2:
        raise ValueError(f"Not enough training rows for seq_len={seq_len} and pred_len={pred_len}")
    val_count = max(1, int(len(starts) * 0.1))
    train_starts = starts[:-val_count]
    val_starts = starts[-val_count:]

    loader_kwargs = data_loader_kwargs(config.get("data_loader"))
    train_loader = DataLoader(make_dataset(train_matrix, seq_len, label_len, pred_len, train_starts), batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(make_dataset(train_matrix, seq_len, label_len, pred_len, val_starts), batch_size=batch_size, shuffle=False, **loader_kwargs)

    runner_config = build_runner_config(config, train_matrix.shape[1])
    model, model_cfg = build_model(runner_config, device_for(config))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(acfg.get("learning_rate", 0.0005)))
    criterion = lambda pred, target: torch.sqrt(torch.mean((pred - target) ** 2) + 1e-12)

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
        for seq_x, seq_y in train_loader:
            seq_x = seq_x.to(next(model.parameters()).device)
            seq_y = seq_y.to(next(model.parameters()).device)
            optimizer.zero_grad()
            output = run_forecast(model, seq_x, model_cfg.label_len, model_cfg.pred_len)
            target = seq_y[:, -model_cfg.pred_len :, :]
            loss = criterion(output, target)
            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss.item() * seq_x.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_x, seq_y in val_loader:
                seq_x = seq_x.to(next(model.parameters()).device)
                seq_y = seq_y.to(next(model.parameters()).device)
                output = run_forecast(model, seq_x, model_cfg.label_len, model_cfg.pred_len)
                target = seq_y[:, -model_cfg.pred_len :, :]
                val_loss += criterion(output, target).item() * seq_x.size(0)
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
            if bool(acfg.get("early_stopping", True)) and no_improve >= patience:
                break

    total_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_training_log.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": runner_config,
            "common_config": config,
            "training_start_time": training_start_time,
            "training_end_time": timestamp_utc(),
            "training_duration_sec": total_time,
        },
        checkpoints / f"{chunk_id}_autoformer_csa.pt",
    )
    return model, model_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out, _ = prepare_output_dirs(config, "Autoformer-CSA")
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    for chunk in chunk_specs(config):
        print(f"Training Autoformer-CSA for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        data = load_chunk(config, chunk)
        train = data.splits[data.train_split].model_input
        train_one_model(
            config, train, data.splits[data.train_split].segments,
            out / "checkpoints", out, chunk.chunk_id,
        )


if __name__ == "__main__":
    main()
