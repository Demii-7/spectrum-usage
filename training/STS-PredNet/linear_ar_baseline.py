"""Lag-matched linear AR controls for the STS-PredNet evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import resolve_branch_config, required_history
from train_integrated import to_sts_layout, validation_with_training_context
from training.common.config import load_config
from training.common.data import chunk_specs, load_chunk
from training.common.data_loader import data_loader_kwargs
from training.common.forecast_export import export_map_forecasts
from training.common.metrics import absolute_and_squared_errors_dbm
from training.common.results import (
    append_metric_rows,
    finalize_results,
    load_band_definitions,
    prepare_output_dirs,
)
from training.common.windowing import filter_target_rows


class LaggedARDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, lags: list[int]):
        self.data = torch.from_numpy(data).float()
        self.targets = np.asarray(targets, dtype=np.int64)
        self.lags = lags

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        target = int(self.targets[index])
        x = torch.stack([self.data[target - lag] for lag in self.lags])
        return x, self.data[target]


class LagMatchedLinearAR(nn.Module):
    """Independent linear AR at each frequency and spatial location."""

    def __init__(self, feature_shape: tuple[int, ...], n_lags: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_lags, *feature_shape))
        self.bias = nn.Parameter(torch.zeros(feature_shape))
        nn.init.xavier_uniform_(self.weight.reshape(n_lags, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.weight.unsqueeze(0)).sum(dim=1) + self.bias


def lag_sets(config: dict[str, Any]) -> dict[str, list[int]]:
    branches = resolve_branch_config(config)
    recent = list(range(branches["lc"], 0, -1))
    period = [
        index * branches["period_interval"]
        for index in range(branches["lp"], 0, -1)
    ]
    requested = config.get("stsprednet_linear_ar", {}).get(
        "variants", ["recent", "recent_daily"]
    )
    available = {"recent": recent, "recent_daily": recent + period}
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown STS linear AR variants: {unknown}")
    return {name: available[name] for name in requested}


def _targets(length: int, history: int, segments) -> np.ndarray:
    rows = np.arange(history, length, dtype=np.int64)
    return filter_target_rows(rows, history, segments)


def train_variant(
    config: dict[str, Any], data, chunk, variant: str, lags: list[int], checkpoints: Path,
) -> Path:
    cfg = config["stsprednet_linear_ar"]
    train_split = data.splits[data.train_split]
    validation_split = data.splits.get(data.validation_split)
    train_x = to_sts_layout(train_split.model_input)
    history = max(lags)
    train_targets = _targets(len(train_x), history, train_split.segments)
    if validation_split is None:
        n_val = max(1, int(len(train_targets) * float(cfg.get("val_fraction", 0.1))))
        val_x = train_x
        val_targets = train_targets[-n_val:]
        train_targets = train_targets[:-n_val]
    else:
        combined, val_targets = validation_with_training_context(
            train_split.model_input,
            validation_split.model_input,
            validation_split.segments,
            history,
        )
        val_x = to_sts_layout(combined)

    loader_cfg = data_loader_kwargs(config.get("data_loader"))
    batch_size = int(cfg.get("batch_size", 8))
    train_loader = DataLoader(
        LaggedARDataset(train_x, train_targets, lags),
        batch_size=batch_size,
        shuffle=True,
        **loader_cfg,
    )
    val_loader = DataLoader(
        LaggedARDataset(val_x, val_targets, lags),
        batch_size=batch_size,
        shuffle=False,
        **loader_cfg,
    )

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LagMatchedLinearAR(tuple(train_x.shape[1:]), len(lags)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg.get("learning_rate", 0.001))
    )
    alpha = float(cfg.get("ridge_alpha", 1.0))
    best_loss = float("inf")
    best_state = None
    patience = int(cfg.get("early_stopping_patience", 5))
    stale = 0

    for epoch in range(1, int(cfg.get("epochs", 20)) + 1):
        model.train()
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss = loss + alpha * model.weight.square().mean()
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for x, target in val_loader:
                x, target = x.to(device), target.to(device)
                total += nn.functional.mse_loss(model(x), target).item() * len(x)
                count += len(x)
        val_loss = total / max(count, 1)
        print(f"{chunk.chunk_id} linear-ar-{variant} epoch {epoch:03d} val_loss={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if bool(cfg.get("early_stopping", True)) and stale >= patience:
                break

    path = checkpoints / f"{chunk.chunk_id}_stsprednet_linear_ar_{variant}.pt"
    torch.save(
        {
            "model_state_dict": best_state or model.state_dict(),
            "feature_shape": tuple(train_x.shape[1:]),
            "lags": lags,
            "variant": variant,
            "normalization": data.normalization,
        },
        path,
    )
    return path


def predict_recursive(
    model: LagMatchedLinearAR,
    data: np.ndarray,
    target_rows: np.ndarray,
    horizon: int,
    lags: list[int],
    device: torch.device,
) -> np.ndarray:
    generated: list[np.ndarray] = []
    for step in range(1, horizon + 1):
        frames = []
        for lag in lags:
            if lag < step:
                frames.append(generated[step - lag - 1])
            else:
                frames.append(data[target_rows - horizon + step - lag])
        x = torch.from_numpy(np.stack(frames, axis=1)).float().to(device)
        with torch.no_grad():
            generated.append(model(x).cpu().numpy().astype(np.float32))
    return generated[-1]


def evaluate_variant(
    config: dict[str, Any], data, chunk, bands, variant: str, checkpoint: Path, out: Path,
):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lags = [int(value) for value in saved["lags"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LagMatchedLinearAR(tuple(saved["feature_shape"]), len(lags)).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    model_name = f"stsprednet_linear_ar_{variant}"
    aggregate_rows, frequency_rows, band_rows = [], [], []

    split = data.splits[data.test_split]
    split_x = to_sts_layout(split.model_input)
    raw_model = to_sts_layout(split.raw_dbm)
    predictions, targets, rows_by_horizon = {}, {}, {}
    for horizon in [int(value) for value in config["windowing"]["horizons"]]:
        history = max(lags) + horizon - 1
        rows = _targets(len(split_x), history, split.segments)
        if not len(rows):
            continue
        pred_norm = predict_recursive(model, split_x, rows, horizon, lags, device)
        target_raw = raw_model[rows]
        pred_last = np.moveaxis(pred_norm, 1, -1)
        target_last = np.moveaxis(target_raw, 1, -1)
        pred_dbm_last, abs_err, sq_err = absolute_and_squared_errors_dbm(
            pred_last, target_last, data.normalization,
        )
        append_metric_rows(
            aggregate_rows, frequency_rows, band_rows,
            chunk_id=chunk.chunk_id,
            start_mhz=chunk.start_mhz,
            end_mhz=chunk.end_mhz,
            split_name=data.test_split,
            horizon=horizon,
            model=model_name,
            target_rows=rows + int(split.row_start),
            history_offset=int(split.row_start),
            freqs=data.frequencies,
            abs_err=np.mean(abs_err, axis=(1, 2)),
            sq_err=np.mean(sq_err, axis=(1, 2)),
            bands=bands,
            feature_labels=data.feature_labels,
        )
        predictions[horizon] = np.moveaxis(pred_dbm_last, -1, 1)
        targets[horizon] = target_raw
        rows_by_horizon[horizon] = rows

    if predictions and split_x.ndim == 4:
        export_map_forecasts(
            out, chunk.chunk_id, model_name, predictions, targets, rows_by_horizon,
            {"model": model_name, "lags": lags, "checkpoint": str(checkpoint)},
        )
    return aggregate_rows, frequency_rows, band_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = args.output_dir or Path("runs") / (
        "stsprednet_linear_ar_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    out, checkpoints = prepare_output_dirs(run_dir)
    bands = load_band_definitions(config)
    all_aggregate, all_frequency, all_band = [], [], []
    for chunk in chunk_specs(config):
        data = load_chunk(
            config, chunk,
            val_fraction=float(config["stsprednet_linear_ar"].get("val_fraction", 0.1)),
        )
        for variant, lags in lag_sets(config).items():
            checkpoint = checkpoints / f"{chunk.chunk_id}_stsprednet_linear_ar_{variant}.pt"
            if args.phase in ("train", "all"):
                checkpoint = train_variant(config, data, chunk, variant, lags, checkpoints)
            if args.phase in ("evaluate", "all"):
                rows = evaluate_variant(config, data, chunk, bands, variant, checkpoint, out)
                all_aggregate.extend(rows[0])
                all_frequency.extend(rows[1])
                all_band.extend(rows[2])
    if args.phase in ("evaluate", "all"):
        finalize_results(
            out, "STS-PredNet lag-matched linear AR",
            all_aggregate, all_frequency, all_band,
        )


if __name__ == "__main__":
    main()
