from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from dataset import SpectrumMapDataset, normalize_splits
from model import DSwinLSTM_I
from training.common.config import load_config
from training.common.runtime import timestamp_utc
from training.common.results import append_metric_rows, finalize_results, load_band_definitions, prepare_output_dirs
from training.common.data import chunk_specs, load_chunk
from training.common.windowing import target_rows_for


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


def predict_for_targets(model: DSwinLSTM_I, full_norm_map: np.ndarray, target_rows: np.ndarray, horizon: int, lookback: int, batch_size: int) -> np.ndarray:
    starts = target_rows - horizon - lookback + 1
    windows = np.stack([full_norm_map[start : start + lookback] for start in starts], axis=0).astype(np.float32)
    mask = np.ones_like(windows, dtype=np.float32)
    x = torch.from_numpy(windows).permute(0, 1, 4, 2, 3).contiguous()
    m = torch.from_numpy(mask)
    loader = DataLoader(list(zip(x, m)), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_mask in loader:
            batch_x = batch_x.to(next(model.parameters()).device)
            batch_mask = batch_mask.to(next(model.parameters()).device)
            pred = model(batch_x, batch_mask)
            preds.append(pred[:, horizon - 1, 0, 0, :].cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands, out: Path, checkpoint_path: Path):
    dcfg = config["dswinlstm_i"]
    lookback = int(dcfg.get("input_sequence_length", config["windowing"]["lookback"]))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    min_history = int(config["windowing"].get("min_history", 4320))
    batch_size = int(dcfg.get("batch_size", 2))
    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train_raw = data.splits[data.train_split].raw_dbm

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = ckpt["normalization_stats"]
    model = DSwinLSTM_I(ckpt["model_config"]).to(device_for(config))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    train_map = to_pseudo_map(train_raw)
    for horizon in horizons:
        for split_name in test_splits:
            split = data.splits[split_name]
            split_raw = split.raw_dbm
            split_map = to_pseudo_map(split_raw)
            full_train, _, full_test, stats = normalize_splits(train_map, split_map, split_map, build_runner_config(config, train_raw.shape[1]), full_data=np.concatenate([train_map, split_map], axis=0))
            full_norm = np.concatenate([full_train, full_test], axis=0)
            full_raw = np.vstack([train_raw, split_raw]).astype(np.float32)
            history_offset = len(train_raw)
            target_rows = target_rows_for(len(split_raw), history_offset, horizon, lookback, min_history)
            pred_norm = predict_for_targets(model, full_norm, target_rows, horizon, lookback, batch_size)
            pred_raw = denormalize_map(pred_norm, stats)
            target = full_raw[target_rows]
            abs_err = np.abs(pred_raw - target)
            sq_err = (pred_raw - target) ** 2
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
    parser = argparse.ArgumentParser(description="Evaluate a trained DSwinLSTM-I checkpoint")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path template (use {chunk_id} for per-chunk substitution)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = MODEL_NAME
    if args.output_dir is not None:
        run_dir = args.output_dir
    else:
        exp_name = args.name or f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out, checkpoints = prepare_output_dirs(run_dir)
    bands = load_band_definitions(config)

    total_start_time = timestamp_utc()
    total_start = time.perf_counter()
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for chunk in chunk_specs(config):
        print(f"Evaluating DSwinLSTM-I for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        if args.checkpoint:
            ckpt_path = Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
        else:
            ckpt_path = out / "checkpoints" / f"{chunk.chunk_id}_dswinlstm_i.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}, skipping {chunk.chunk_id}")
            continue
        a, f, b = evaluate_chunk(config, chunk, bands, out, ckpt_path)
        aggregate_rows.extend(a)
        frequency_rows.extend(f)
        band_rows.extend(b)

    total_run = time.perf_counter() - total_start
    finalize_results(
        out,
        "DSwinLSTM-I",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
