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
from training.common.forecast_export import export_map_forecasts  # noqa: E402
from training.common.integrated import epoch_log_row, finalize_results, prepare_output_dirs, timestamp_utc  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.metrics import absolute_and_squared_errors_dbm  # noqa: E402
from training.common.results import append_metric_rows, load_band_definitions  # noqa: E402
from training.common.windowing import aligned_history_matrix, selected_horizon_index, target_rows_for  # noqa: E402


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


def train_one_model(config: dict[str, Any], train_matrix: np.ndarray, checkpoints: Path, out: Path, chunk_id: str) -> VanillaLSTMForecaster:
    vcfg = config["vanillalstm"]
    lookback = int(vcfg.get("input_sequence_length", config["windowing"]["lookback"]))
    prediction_horizon = int(vcfg.get("prediction_horizon", max(config["windowing"]["horizons"])))
    batch_size = int(vcfg.get("batch_size", 32))
    epochs = int(vcfg.get("epochs", 20))
    val_fraction = 0.1
    clip_norm = float(vcfg.get("gradient_clip", 1.0))
    patience = int(vcfg.get("patience", 10))

    starts = np.arange(0, len(train_matrix) - lookback - prediction_horizon + 1, dtype=np.int64)
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
    )
    val_loader = DataLoader(
        VanillaWindowDataset(train_matrix, val_starts, lookback, prediction_horizon),
        batch_size=batch_size,
        shuffle=False,
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
                pred = model(x)
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


def predict_for_targets(model: VanillaLSTMForecaster, full_x: np.ndarray, target_rows: np.ndarray, horizon: int, lookback: int, batch_size: int) -> np.ndarray:
    histories = aligned_history_matrix(full_x, target_rows, horizon, lookback)
    loader = DataLoader(torch.from_numpy(histories).float(), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            pred = model(x.to(next(model.parameters()).device))
            preds.append(pred[:, selected_horizon_index(horizon), :].cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands: pd.DataFrame, out: Path):
    vcfg = config["vanillalstm"]
    lookback = int(vcfg.get("input_sequence_length", config["windowing"]["lookback"]))
    min_history = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    batch_size = int(vcfg.get("batch_size", 32))
    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm
    model = train_one_model(config, train, out / "checkpoints", out, chunk.chunk_id)

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    export_payloads: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        for split_name in test_splits:
            split = data.splits[split_name]
            full_x = np.vstack([train, split.model_input]).astype(np.float32)
            full_raw = np.vstack([train_raw, split.raw_dbm]).astype(np.float32)
            history_offset = len(train)
            target_rows = target_rows_for(len(split.raw_dbm), history_offset, horizon, lookback, min_history)
            pred = predict_for_targets(model, full_x, target_rows, horizon, lookback, batch_size)
            target = full_raw[target_rows]
            local_target_rows = (target_rows - history_offset).astype(np.int64)
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

            payload = export_payloads.setdefault(
                split_name,
                {
                    "predictions_by_horizon": {},
                    "targets_by_horizon": {},
                    "target_rows_by_horizon": {},
                },
            )
            payload["predictions_by_horizon"][horizon] = pred.astype(np.float32)
            payload["targets_by_horizon"][horizon] = target.astype(np.float32)
            payload["target_rows_by_horizon"][horizon] = local_target_rows

    for split_name, payload in export_payloads.items():
        export_map_forecasts(
            out,
            chunk_id=f"{chunk.chunk_id}_{split_name}",
            model_name=MODEL_NAME,
            predictions_by_horizon=payload["predictions_by_horizon"],
            targets_by_horizon=payload["targets_by_horizon"],
            target_rows_by_horizon=payload["target_rows_by_horizon"],
            metadata={
                "model": "VanillaLSTM",
                "split_name": split_name,
                "train_split": data.train_split,
                "test_split": split_name,
                "chunk_id": chunk.chunk_id,
                "start_mhz": chunk.start_mhz,
                "end_mhz": chunk.end_mhz,
                "lookback": lookback,
                "batch_size": batch_size,
                "history_offset": len(train),
                "frequencies_mhz": np.asarray(data.frequencies, dtype=np.float32),
                "normalization": None if data.normalization is None else data.normalization.get("source_split"),
            },
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
    out, _ = prepare_output_dirs(config, "VanillaLSTM")
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    bands = load_band_definitions(config)

    total_start_time = timestamp_utc()
    total_start = time.perf_counter()
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for chunk in chunk_specs(config):
        print(f"Training VanillaLSTM for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        a, f, b = evaluate_chunk(config, chunk, bands, out)
        aggregate_rows.extend(a)
        frequency_rows.extend(f)
        band_rows.extend(b)

    total_run = time.perf_counter() - total_start
    finalize_results(
        out,
        "VanillaLSTM",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Training start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
