from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from stsprednet import STSPredNet  # noqa: E402
from training.common.config import load_config  # noqa: E402
from training.common.data_loader import data_loader_kwargs  # noqa: E402
from training.common.runtime import epoch_log_row, timestamp_utc  # noqa: E402
from training.common.results import prepare_output_dirs  # noqa: E402
from training.common.interpolated_map import (  # noqa: E402
    load_interpolated_map_npz,
    normalize_map_by_frequency,
)
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.windowing import filter_target_rows  # noqa: E402

MODEL_NAME = "stsprednet"


class STSPredNetDataset(Dataset):
    def __init__(self, data_3d: np.ndarray, target_indices: np.ndarray,
                 lc: int, lp: int, period_interval: int):
        self.data = torch.from_numpy(data_3d).float()
        self.target_indices = target_indices
        self.lc = lc
        self.lp = lp
        self.period_interval = period_interval

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, idx: int):
        target_idx = int(self.target_indices[idx])
        t = target_idx - 1

        closeness = self.data[t - self.lc + 1 : t + 1]
        period_list = [self.data[target_idx - p * self.period_interval]
                       for p in range(self.lp, 0, -1)]
        period = torch.stack(period_list, dim=0)
        target = self.data[target_idx]
        return closeness, period, target


def collate_stsprednet(batch):
    closeness, period, target = zip(*batch)
    return (
        torch.stack(closeness, dim=0),
        torch.stack(period, dim=0),
        torch.stack(target, dim=0),
    )


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    scfg = config["stsprednet"]
    return {
        "model": {
            "input_channels": scfg["model"]["input_channels"],
            "map_height": scfg["model"]["map_height"],
            "map_width": n_bins,
            "num_layers": scfg["model"]["num_layers"],
            "hidden_dim": scfg["model"]["hidden_dim"],
            "kernel_size": list(scfg["model"]["kernel_size"]),
            "output_activation": scfg["model"]["output_activation"],
            "fusion_weight_shape": scfg["model"]["fusion_weight_shape"],
        },
        "branches": {
            "use_closeness": True,
            "use_period": True,
            "use_trend": False,
            "share_branch_weights": False,
        },
    }


def build_map_model_config(config: dict[str, Any], n_freq: int, grid_h: int, grid_w: int) -> dict[str, Any]:
    scfg = config["stsprednet"]
    return {
        "model": {
            "input_channels": n_freq,
            "map_height": grid_h,
            "map_width": grid_w,
            "num_layers": scfg["model"]["num_layers"],
            "hidden_dim": scfg["model"]["hidden_dim"],
            "kernel_size": list(scfg["model"]["kernel_size"]),
            "output_activation": scfg["model"]["output_activation"],
            "fusion_weight_shape": scfg["model"]["fusion_weight_shape"],
        },
        "branches": {
            "use_closeness": True,
            "use_period": True,
            "use_trend": False,
            "share_branch_weights": False,
        },
    }


def train_one_model(config: dict[str, Any], full_x: np.ndarray,
                    segments, checkpoints: Path, out: Path, chunk_id: str) -> STSPredNet:
    scfg = config["stsprednet"]
    lc = int(scfg["lc"])
    lp = int(scfg["lp"])
    period_interval = int(scfg["period_interval"])
    batch_size = int(scfg["batch_size"])
    epochs = int(scfg["epochs"])
    lr = float(scfg["learning_rate"])
    weight_decay = float(scfg["weight_decay"])
    clip_norm = float(scfg["gradient_clip_norm"])
    patience = int(scfg["patience"])

    n_bins = full_x.shape[1]
    data_3d = full_x[:, None, None, :].astype(np.float32)

    period_min = lp * period_interval
    all_targets = np.arange(period_min, len(full_x))
    all_targets = filter_target_rows(all_targets, period_min, segments)
    if len(all_targets) < 100:
        raise ValueError(
            f"Not enough valid targets ({len(all_targets)}) "
            f"for period history {period_min}."
        )

    n_val = max(1, int(len(all_targets) * 0.1))
    train_targets = all_targets[:-n_val]
    val_targets = all_targets[-n_val:]

    train_ds = STSPredNetDataset(data_3d, train_targets, lc, lp, period_interval)
    val_ds = STSPredNetDataset(data_3d, val_targets, lc, lp, period_interval)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_stsprednet,
        **data_loader_kwargs(config.get("data_loader")),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_stsprednet,
        **data_loader_kwargs(config.get("data_loader")),
    )

    model_config = build_model_config(config, n_bins)
    device = device_for()
    model = STSPredNet(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    log_rows = []
    epoch_times: list[float] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        t_epoch = time.perf_counter()
        model.train()
        train_loss = 0.0
        for closeness, period, target in train_loader:
            closeness = closeness.to(device)
            period = period.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(closeness, period, None)
            loss = criterion(pred, target)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_loss += loss.item() * target.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for closeness, period, target in val_loader:
                closeness = closeness.to(device)
                period = period.to(device)
                target = target.to(device)
                pred = model(closeness, period, None)
                val_loss += criterion(pred, target).item() * target.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    total_time = time.perf_counter() - t_start
    print(f"{chunk_id} training done in {total_time:.1f}s ({total_time/60:.1f} min)")

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
        checkpoints / f"{chunk_id}_stsprednet.pt",
    )
    return model


def train_map_model(
    config: dict[str, Any],
    train_x: np.ndarray,
    checkpoints: Path,
    out: Path,
    chunk_id: str,
) -> STSPredNet:
    scfg = config["stsprednet"]
    lc = int(scfg["lc"])
    lp = int(scfg["lp"])
    period_interval = int(scfg["period_interval"])
    batch_size = int(scfg["batch_size"])
    epochs = int(scfg["epochs"])
    lr = float(scfg["learning_rate"])
    weight_decay = float(scfg["weight_decay"])
    clip_norm = float(scfg["gradient_clip_norm"])
    patience = int(scfg["patience"])

    period_min = lp * period_interval
    all_targets = np.arange(period_min, len(train_x), dtype=np.int64)
    if len(all_targets) < 100:
        raise ValueError(
            f"Not enough valid map targets ({len(all_targets)}) for period history {period_min}."
        )

    n_val = max(1, int(len(all_targets) * 0.1))
    train_targets = all_targets[:-n_val]
    val_targets = all_targets[-n_val:]

    train_ds = STSPredNetDataset(train_x, train_targets, lc, lp, period_interval)
    val_ds = STSPredNetDataset(train_x, val_targets, lc, lp, period_interval)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_stsprednet,
        **data_loader_kwargs(config.get("data_loader")),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_stsprednet,
        **data_loader_kwargs(config.get("data_loader")),
    )

    _, n_freq, grid_h, grid_w = train_x.shape
    model_config = build_map_model_config(config, n_freq, grid_h, grid_w)
    device = device_for()
    model = STSPredNet(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    log_rows = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        t_epoch = time.perf_counter()
        model.train()
        train_loss = 0.0
        for closeness, period, target in train_loader:
            closeness = closeness.to(device)
            period = period.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(closeness, period, None)
            loss = criterion(pred, target)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_loss += loss.item() * target.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for closeness, period, target in val_loader:
                closeness = closeness.to(device)
                period = period.to(device)
                target = target.to(device)
                pred = model(closeness, period, None)
                val_loss += criterion(pred, target).item() * target.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        duration = time.perf_counter() - t_epoch
        log_rows.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=timestamp_utc(),
                epoch_duration_sec=duration,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        print(f"{chunk_id} map epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={duration:.1f}s")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
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
        checkpoints / f"{chunk_id}_stsprednet.pt",
    )
    return model


def predict_recursive_map(
    model: STSPredNet,
    device: torch.device,
    full_x: np.ndarray,
    target_rows: np.ndarray,
    horizon: int,
    lc: int,
    lp: int,
    period_interval: int,
) -> np.ndarray:
    preds = []
    for target_row in target_rows:
        origin = target_row - horizon
        running = [full_x[i].copy() for i in range(origin - lc + 1, origin + 1)]
        for step in range(1, horizon + 1):
            current_target = origin + step
            close = np.stack(running[-lc:], axis=0).astype(np.float32)
            period = np.stack(
                [full_x[current_target - p * period_interval] for p in range(lp, 0, -1)],
                axis=0,
            ).astype(np.float32)
            close_t = torch.from_numpy(close).float().unsqueeze(0)
            period_t = torch.from_numpy(period).float().unsqueeze(0)
            with torch.no_grad():
                pred = model(close_t.to(device), period_t.to(device), None)
            pred_np = pred.cpu().numpy()[0].astype(np.float32)
            if step == horizon:
                preds.append(pred_np)
            else:
                running.append(pred_np)
    return np.stack(preds, axis=0).astype(np.float32)


def run_map_mode(config: dict[str, Any], out: Path, checkpoints: Path) -> None:
    data_cfg = config["data"]
    train_map_path = data_cfg.get("train_map_path")
    test_map_path = data_cfg.get("test_map_path") or train_map_path
    map_key = str(data_cfg.get("map_key", "map_db"))
    if not train_map_path:
        raise ValueError("Map mode requires data.train_map_path")

    train_raw, train_meta = load_interpolated_map_npz(train_map_path, map_key)
    test_raw, test_meta = load_interpolated_map_npz(test_map_path, map_key)
    train_x, test_x, norm_stats = normalize_map_by_frequency(
        train_raw,
        test_raw,
        enabled=bool(config.get("preprocessing", {}).get("normalize", True)),
    )

    chunk_id = str(data_cfg.get("chunk_id", "powder_map"))
    model = train_map_model(config, train_x, checkpoints, out, chunk_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = "stsprednet"
    if args.output_dir is not None:
        run_dir = args.output_dir
    else:
        exp_name = args.name or f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out, checkpoints = prepare_output_dirs(run_dir)

    if config["data"].get("train_map_path"):
        print("Interpolated-map mode enabled — training STS-PredNet on map data.")
        run_map_mode(config, out, checkpoints)
        return

    for chunk in chunk_specs(config):
        print(f"Training STS-PredNet for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        data = load_chunk(config, chunk)
        train = data.splits[data.train_split].model_input
        train_one_model(
            config, train, data.splits[data.train_split].segments,
            checkpoints, out, chunk.chunk_id,
        )


if __name__ == "__main__":
    main()
