from __future__ import annotations

import argparse
from pathlib import Path
import sys
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
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.metrics import absolute_and_squared_errors_dbm  # noqa: E402
from training.common.results import append_metric_rows, load_band_definitions, output_dir  # noqa: E402
from training.common.windowing import target_rows_for  # noqa: E402

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


def train_one_model(config: dict[str, Any], full_x: np.ndarray,
                    out: Path, chunk_id: str) -> STSPredNet:
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
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_stsprednet,
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

    for epoch in range(1, epochs + 1):
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

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"{chunk_id} epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_training_log.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "common_config": config,
        },
        out / "models" / f"{chunk_id}_stsprednet.pt",
    )
    return model


def predict_recursive(model: STSPredNet, device: torch.device,
                      full_x: np.ndarray, target_rows: np.ndarray,
                      horizon: int, lc: int, lp: int,
                      period_interval: int) -> np.ndarray:
    n_bins = full_x.shape[1]
    preds = []

    for target_row in target_rows:
        origin = target_row - horizon
        running = [full_x[i].copy() for i in range(origin - lc + 1, origin + 1)]

        for step in range(1, horizon + 1):
            current_target = origin + step

            close = np.stack(running[-lc:], axis=0)
            close = close[:, None, None, :].astype(np.float32)
            close_t = torch.from_numpy(close).float().unsqueeze(0)

            period_list = [full_x[current_target - p * period_interval]
                           for p in range(lp, 0, -1)]
            period = np.stack(period_list, axis=0)
            period = period[:, None, None, :].astype(np.float32)
            period_t = torch.from_numpy(period).float().unsqueeze(0)

            with torch.no_grad():
                pred = model(close_t.to(device), period_t.to(device), None)
            pred_np = pred.cpu().numpy()[0, 0, 0, :]

            if step == horizon:
                preds.append(pred_np)
            else:
                running.append(pred_np)

    return np.stack(preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands: pd.DataFrame, out: Path):
    scfg = config["stsprednet"]
    lc = int(scfg["lc"])
    lp = int(scfg["lp"])
    period_interval = int(scfg["period_interval"])
    min_history_base = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    test_splits = config["data"].get("test_splits", ["CC2_test"])

    data = load_chunk(config, chunk)
    train = data.splits["CC2_train"].model_input
    train_raw = data.splits["CC2_train"].raw_dbm

    full_x_all = np.vstack([train, data.splits["CC2_test"].model_input])
    model = train_one_model(config, full_x_all, out, chunk.chunk_id)
    device = next(model.parameters()).device

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    period_min_base = lp * period_interval
    for horizon in horizons:
        min_needed = max(period_min_base + horizon - 1, min_history_base, horizon + lc - 1)
        for split_name in test_splits:
            split = data.splits[split_name]
            full_x = np.vstack([train, split.model_input]).astype(np.float32)
            full_raw = np.vstack([train_raw, split.raw_dbm]).astype(np.float32)
            history_offset = len(train)

            target_rows = target_rows_for(
                len(split.raw_dbm), history_offset, horizon,
                lc, min_needed,
            )
            if len(target_rows) == 0:
                print(f"  No valid target rows for {chunk.chunk_id} {split_name} h={horizon}")
                continue

            pred = predict_recursive(
                model, device, full_x, target_rows, horizon,
                lc, lp, period_interval,
            )
            target = full_raw[target_rows]
            _, abs_err, sq_err = absolute_and_squared_errors_dbm(
                pred, target, data.normalization,
            )
            append_metric_rows(
                aggregate_rows, frequency_rows, band_rows,
                chunk_id=chunk.chunk_id,
                start_mhz=chunk.start_mhz,
                end_mhz=chunk.end_mhz,
                split_name=split_name,
                horizon=horizon,
                model=MODEL_NAME,
                target_rows=target_rows,
                history_offset=history_offset,
                freqs=data.shared_frequencies,
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
    out = args.output_dir or output_dir(config, "STS-PredNet")
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    bands = load_band_definitions(config)

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for chunk in chunk_specs(config):
        print(f"Training STS-PredNet for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        a, f, b = evaluate_chunk(config, chunk, bands, out)
        aggregate_rows.extend(a)
        frequency_rows.extend(f)
        band_rows.extend(b)

    pd.DataFrame(aggregate_rows).to_csv(out / "aggregate_metrics.csv", index=False)
    pd.DataFrame(frequency_rows).to_csv(out / "per_frequency_metrics.csv", index=False)
    pd.DataFrame(band_rows).to_csv(out / "per_band_metrics.csv", index=False)
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
