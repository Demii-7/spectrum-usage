from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import SequenceDataset
from train import build_model
from training.common.config import load_config
from training.common.runtime import timestamp_utc
from training.common.results import append_metric_rows, finalize_results, load_band_definitions, prepare_output_dirs
from training.common.data import chunk_specs, load_chunk
from training.common.metrics import absolute_and_squared_errors_dbm
from training.common.windowing import target_rows_for


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


def run_forecast(model: nn.Module, seq_x: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
    dec_input = torch.zeros(seq_x.shape[0], label_len + pred_len, seq_x.shape[-1], device=seq_x.device)
    dec_input[:, :label_len, :] = seq_x[:, -label_len:, :]
    x_mark_enc = torch.zeros(seq_x.shape[0], seq_x.shape[1], 4, device=seq_x.device)
    x_mark_dec = torch.zeros(dec_input.shape[0], dec_input.shape[1], 4, device=seq_x.device)
    return model(seq_x, x_mark_enc, dec_input, x_mark_dec)


def predict_for_targets(model: nn.Module, model_cfg, full_x: np.ndarray, target_rows: np.ndarray, horizon: int, seq_len: int, label_len: int, batch_size: int) -> np.ndarray:
    starts = target_rows - horizon - seq_len + 1
    x = np.stack([full_x[start : start + seq_len] for start in starts], axis=0).astype(np.float32)
    loader = DataLoader(torch.from_numpy(x).float(), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(next(model.parameters()).device)
            output = run_forecast(model, batch_x, label_len, model_cfg.pred_len)
            preds.append(output[:, horizon - 1, :].cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands, out: Path, checkpoint_path: Path):
    acfg = config["autoformer_csa"]
    seq_len = int(acfg.get("seq_len", config["windowing"]["lookback"]))
    label_len = int(acfg.get("label_len", seq_len // 2))
    min_history = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    batch_size = int(acfg.get("batch_size", 8))
    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm

    runner_config = build_runner_config(config, train.shape[1])
    model, model_cfg = build_model(runner_config, device_for(config))
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
            target_rows = target_rows_for(len(split.raw_dbm), history_offset, horizon, seq_len, min_history)
            pred = predict_for_targets(model, model_cfg, full_x, target_rows, horizon, seq_len, label_len, batch_size)
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
    parser = argparse.ArgumentParser(description="Evaluate a trained Autoformer-CSA checkpoint")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path template (use {chunk_id} for per-chunk substitution)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = "autoformer-csa"
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
        print(f"Evaluating Autoformer-CSA for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        if args.checkpoint:
            ckpt_path = Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
        else:
            ckpt_path = out / "checkpoints" / f"{chunk.chunk_id}_autoformer_csa.pt"
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
        "Autoformer-CSA",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
