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

from dataset import SpectrumMapDataset, normalize_splits  # noqa: E402
from model import DSwinLSTM_I  # noqa: E402
from training.common.config import load_config  # noqa: E402
from training.common.integrated import epoch_log_row, prepare_output_dirs, timestamp_utc  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402


MODEL_NAME = "dswinlstm_i"


def device_for(config: dict[str, Any]) -> torch.device:
    requested = str(config["dswinlstm_i"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def to_pseudo_map(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, None, :, None].astype(np.float32)


def build_runner_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    dcfg = config["dswinlstm_i"]
    return {
        "windowing": {
            "input_sequence_length": int(dcfg.get("input_sequence_length", config["windowing"]["lookback"])),
            "prediction_horizon": int(dcfg.get("prediction_horizon", max(config["windowing"]["horizons"]))),
            "train_stride": int(dcfg.get("train_stride", 1)),
            "val_stride": int(dcfg.get("val_stride", max(config["windowing"]["horizons"]))),
            "test_stride": int(dcfg.get("test_stride", max(config["windowing"]["horizons"]))),
        },
        "preprocessing": {
            "normalization": str(dcfg.get("normalization", "minmax")),
            "minmax_range": list(dcfg.get("minmax_range", [-1, 1])),
            "fit_on_train_only": True,
            "missing_rate": float(dcfg.get("missing_rate", 0.3)),
            "missing_strategy": str(dcfg.get("missing_strategy", "random")),
            "mask_targets": False,
        },
        "model": {
            "model_name": MODEL_NAME,
            "map_height": 1,
            "map_width": n_bins,
            "input_channels": 1,
            **dict(dcfg["model"]),
        },
        "training": {
            "teacher_forcing_ratio": float(dcfg.get("teacher_forcing_ratio", 1.0)),
        },
    }


def denormalize_map(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    if stats["method"] == "minmax":
        lo, hi = stats["range"]
        scaled = (values - lo) / (hi - lo + 1e-8)
        return (scaled * (stats["dmax"] - stats["dmin"]) + stats["dmin"]).astype(np.float32)
    if stats["method"] == "zscore":
        return (values * stats["std"] + stats["mean"]).astype(np.float32)
    return values.astype(np.float32)


def train_one_model(config: dict[str, Any], train_raw: np.ndarray, segments, checkpoints: Path, out: Path, chunk_id: str):
    dcfg = config["dswinlstm_i"]
    split_idx = max(1, int(len(train_raw) * 0.9))
    train_map = to_pseudo_map(train_raw[:split_idx])
    val_map = to_pseudo_map(train_raw[split_idx:])
    if len(val_map) == 0:
        val_map = train_map[-max(2, int(len(train_map) * 0.1)) :]
    train_norm, val_norm, _, stats = normalize_splits(train_map, val_map, val_map, build_runner_config(config, train_raw.shape[1]))
    runner_config = build_runner_config(config, train_raw.shape[1])
    runner_config["sequence_segments"] = segments

    train_ds = SpectrumMapDataset(train_norm, runner_config, split="train")
    val_ds = SpectrumMapDataset(val_norm, runner_config, split="val")
    batch_size = int(dcfg.get("batch_size", 2))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) > 0 else None

    model = DSwinLSTM_I(runner_config).to(device_for(config))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(dcfg.get("learning_rate", 0.0001)))
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    log_rows: list[dict[str, Any]] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()

    for epoch in range(1, int(dcfg.get("epochs", 6)) + 1):
        epoch_start_time = timestamp_utc()
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for x, mask, y in train_loader:
            x = x.permute(0, 1, 4, 2, 3).contiguous().to(next(model.parameters()).device)
            mask = mask.to(next(model.parameters()).device)
            y = y.permute(0, 1, 4, 2, 3).contiguous().to(next(model.parameters()).device)
            optimizer.zero_grad()
            pred = model(x, mask, y_teacher=y, teacher_forcing_ratio=float(dcfg.get("teacher_forcing_ratio", 1.0)))
            loss = criterion(pred, y)
            loss.backward()
            clip = float(dcfg.get("gradient_clip", 5.0))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, mask, y in val_loader:
                    x = x.permute(0, 1, 4, 2, 3).contiguous().to(next(model.parameters()).device)
                    mask = mask.to(next(model.parameters()).device)
                    y = y.permute(0, 1, 4, 2, 3).contiguous().to(next(model.parameters()).device)
                    pred = model(x, mask)
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
        print(f"{chunk_id} epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={epoch_duration:.1f}s")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if bool(dcfg.get("early_stopping", True)) and no_improve >= int(dcfg.get("patience", 6)):
                break

    total_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_training_log.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": runner_config,
            "normalization_stats": stats,
            "training_start_time": training_start_time,
            "training_end_time": timestamp_utc(),
            "training_duration_sec": total_time,
        },
        checkpoints / f"{chunk_id}_dswinlstm_i.pt",
    )
    return model, runner_config, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out, _ = prepare_output_dirs(config, "DSwinLSTM-I")
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    for chunk in chunk_specs(config):
        print(f"Training DSwinLSTM-I for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        data = load_chunk(config, chunk)
        train_raw = data.splits[data.train_split].raw_dbm
        train_one_model(
            config, train_raw, data.splits[data.train_split].segments,
            out / "checkpoints", out, chunk.chunk_id,
        )


if __name__ == "__main__":
    main()
