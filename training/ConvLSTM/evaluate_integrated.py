from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import ConvLSTMPredictor
from training.common.config import load_config
from training.common.forecast_export import export_map_forecasts
from training.common.integrated import finalize_results, prepare_output_dirs, timestamp_utc
from training.common.interpolated_map import (
    denormalize_map,
    load_interpolated_map_npz,
    normalize_map_by_frequency,
    prediction_start_row,
)
from training.common.data import (
    chunk_specs,
    clean_interpolated_map,
    load_chunk,
    model_matrix_to_convlstm_frames,
)
from training.common.metrics import absolute_and_squared_errors_dbm
from training.common.results import append_metric_rows, load_band_definitions
from training.common.windowing import aligned_history_matrix, selected_horizon_index, target_rows_for


MODEL_NAME = "convlstm"


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


def predict_map_for_targets(
    model: ConvLSTMPredictor,
    full_x: np.ndarray,
    target_rows: np.ndarray,
    horizon: int,
    lookback: int,
    batch_size: int,
) -> np.ndarray:
    origins = target_rows - horizon + 1
    histories = np.stack([full_x[origin - lookback : origin] for origin in origins], axis=0).astype(np.float32)
    loader = DataLoader(torch.from_numpy(histories).float(), batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    preds = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            pred = model(x.to(device))
            preds.append(pred[:, selected_horizon_index(horizon)].cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


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


def evaluate_map_mode(config: dict[str, Any], out: Path, checkpoint_path: Path) -> None:
    data_cfg = config["data"]
    ccfg = config["convlstm"]
    map_cfg = ccfg.get("interpolated_map", {})
    train_map_path = data_cfg.get("train_map_path") or map_cfg.get("map_path")
    test_map_path = data_cfg.get("test_map_path") or train_map_path
    map_key = str(data_cfg.get("map_key") or map_cfg.get("map_key", "map_db"))
    if not train_map_path:
        raise ValueError("Map mode requires data.train_map_path or convlstm.interpolated_map.map_path")

    train_raw, train_meta = load_map_for_path(train_map_path, map_key)
    test_raw, test_meta = load_map_for_path(test_map_path, map_key)
    train_x, test_x, norm_stats = normalize_map_by_frequency(
        train_raw,
        test_raw,
        enabled=bool(config.get("preprocessing", {}).get("normalize", True)),
    )

    T, F, H, W = train_x.shape
    lookback = int(ccfg.get("input_sequence_length", config["windowing"]["lookback"]))
    prediction_horizon = int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"])))
    batch_size = int(ccfg.get("batch_size", 32))

    model_config = build_map_model_config(config, F, H, W)
    device = device_for()
    model = ConvLSTMPredictor(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    start_idx = prediction_start_row(config, len(test_x))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    predictions_by_horizon: dict[int, np.ndarray] = {}
    targets_by_horizon: dict[int, np.ndarray] = {}
    target_rows_by_horizon: dict[int, np.ndarray] = {}
    for horizon in horizons:
        first_target = max(start_idx, lookback + horizon - 1)
        target_rows = np.arange(first_target, len(test_x), dtype=np.int64)
        if len(target_rows) == 0:
            continue
        pred_norm = predict_map_for_targets(model, test_x, target_rows, horizon, lookback, batch_size)
        pred = denormalize_map(pred_norm, norm_stats)
        target = test_raw[target_rows].astype(np.float32)
        predictions_by_horizon[horizon] = pred
        targets_by_horizon[horizon] = target
        target_rows_by_horizon[horizon] = target_rows

    export_map_forecasts(
        out,
        chunk_id=str(data_cfg.get("chunk_id", "powder_map")),
        model_name=MODEL_NAME,
        predictions_by_horizon=predictions_by_horizon,
        targets_by_horizon=targets_by_horizon,
        target_rows_by_horizon=target_rows_by_horizon,
        metadata={
            "model": "ConvLSTM",
            "train_map_path": str(train_meta["path"]),
            "test_map_path": str(test_meta["path"]),
            "map_key": map_key,
            "prediction_start_row": config.get("evaluation", {}).get("prediction_start_row"),
            "train_shape_tf_hw": list(train_raw.shape),
            "test_shape_tf_hw": list(test_raw.shape),
            "normalization": None if norm_stats is None else norm_stats["method"],
            "train_map_metadata": train_meta.get("metadata"),
            "test_map_metadata": test_meta.get("metadata"),
        },
    )


def evaluate_csv_chunk(config: dict[str, Any], chunk, bands, out: Path, checkpoint_path: Path):
    ccfg = config["convlstm"]
    lookback = int(ccfg.get("input_sequence_length", config["windowing"]["lookback"]))
    min_history = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    batch_size = int(ccfg.get("batch_size", 32))
    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm

    model_config = build_model_config(config, train.shape[1])
    device = device_for()
    model = ConvLSTMPredictor(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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
    parser = argparse.ArgumentParser(description="Evaluate a trained ConvLSTM checkpoint")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path (use {chunk_id} for per-chunk substitution in CSV mode)")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out, _ = prepare_output_dirs(config, "ConvLSTM")
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)

    if config["data"].get("train_map_path") or config["convlstm"].get("interpolated_map", {}).get("enabled", False):
        print("Interpolated-map mode enabled — evaluating on map data.")
        ckpt_path = args.checkpoint or out / "checkpoints" / "interpolated_map_convlstm.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            return
        evaluate_map_mode(config, out, ckpt_path)
        return

    bands = load_band_definitions(config)
    total_start_time = timestamp_utc()
    total_start = time.perf_counter()
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for chunk in chunk_specs(config):
        print(f"Evaluating ConvLSTM for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        if args.checkpoint:
            ckpt_path = Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
        else:
            ckpt_path = out / "checkpoints" / f"{chunk.chunk_id}_convlstm.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}, skipping {chunk.chunk_id}")
            continue
        a, f, b = evaluate_csv_chunk(config, chunk, bands, out, ckpt_path)
        aggregate_rows.extend(a)
        frequency_rows.extend(f)
        band_rows.extend(b)

    total_run = time.perf_counter() - total_start
    finalize_results(
        out,
        "ConvLSTM",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
