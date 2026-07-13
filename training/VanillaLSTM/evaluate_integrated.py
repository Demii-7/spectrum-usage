from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import VanillaLSTMForecaster
from training.common.config import load_config
from training.common.forecast_export import export_map_forecasts
from training.common.integrated import finalize_results, prepare_output_dirs, timestamp_utc
from training.common.data import chunk_specs, load_chunk
from training.common.metrics import absolute_and_squared_errors_dbm
from training.common.results import append_metric_rows, load_band_definitions
from training.common.windowing import aligned_history_matrix, selected_horizon_index, target_rows_for


MODEL_NAME = "vanillalstm"


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


def autoregressive_predict_for_origins(
    model: VanillaLSTMForecaster,
    full_x: np.ndarray,
    origin_rows: np.ndarray,
    max_horizon: int,
    lookback: int,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    For each origin row s:
      - initialize with full_x[s : s + lookback]
      - predict step 1
      - append prediction
      - shift lookback by 1
      - repeat until max_horizon

    Returns:
      predictions_by_horizon[h] with shape (num_origins, n_bins)
    """

    device = next(model.parameters()).device
    n_bins = full_x.shape[1]

    current_windows = np.stack(
        [full_x[s : s + lookback] for s in origin_rows],
        axis=0,
    ).astype(np.float32)

    predictions_by_horizon: dict[int, list[np.ndarray]] = {
        h: [] for h in range(1, max_horizon + 1)
    }

    model.eval()

    with torch.no_grad():
        for start in range(0, len(current_windows), batch_size):
            window = torch.from_numpy(current_windows[start : start + batch_size]).float().to(device)

            rollout_preds = []

            for h in range(1, max_horizon + 1):
                output = model(window)

                # Use the model's horizon-1 output as the next autoregressive step
                next_pred = output[:, selected_horizon_index(1), :]

                rollout_preds.append(next_pred.cpu().numpy())

                # Shift lookback left and append prediction
                window = torch.cat(
                    [window[:, 1:, :], next_pred.unsqueeze(1)],
                    dim=1,
                )

            for h, pred_h in enumerate(rollout_preds, start=1):
                predictions_by_horizon[h].append(pred_h)

    return {
        h: np.concatenate(parts, axis=0).astype(np.float32)
        for h, parts in predictions_by_horizon.items()
    }


def evaluate_chunk(config: dict[str, Any], chunk, bands: pd.DataFrame, out: Path, checkpoint_path: Path):
    vcfg = config["vanillalstm"]
    lookback = int(vcfg.get("input_sequence_length", config["windowing"]["lookback"]))
    min_history = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    batch_size = int(vcfg.get("batch_size", 32))
    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm

    model_config = build_model_config(config, train.shape[1])
    model = VanillaLSTMForecaster(model_config).to(device_for(config))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    export_payloads: dict[str, dict[str, Any]] = {}
    max_horizon = max(horizons)

    for split_name in test_splits:
        split = data.splits[split_name]
    
        full_x = split.model_input.astype(np.float32)
        full_raw = split.raw_dbm.astype(np.float32)
    
        history_offset = 0
    
        max_origin = len(full_x) - lookback - max_horizon + 1
        if max_origin <= 0:
            print(f"  Not enough rows for split {split_name}; skipping")
            continue
    
        origin_rows = np.arange(max_origin, dtype=np.int64)
    
        all_preds = autoregressive_predict_for_origins(
            model=model,
            full_x=full_x,
            origin_rows=origin_rows,
            max_horizon=max_horizon,
            lookback=lookback,
            batch_size=batch_size,
        )
    
        for horizon in horizons:
            pred = all_preds[horizon]
    
            target_rows = origin_rows + lookback + horizon - 1
            target = full_raw[target_rows]
    
            local_target_rows = target_rows.astype(np.int64)
    
            _, abs_err, sq_err = absolute_and_squared_errors_dbm(
                pred,
                target,
                data.normalization,
            )
    
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
                    "history_offset": 0,
                    "frequencies_mhz": np.asarray(data.frequencies, dtype=np.float32),
                    "normalization": None if data.normalization is None else data.normalization.get("source_split"),
                    "mean_dbm": None if data.normalization is None else data.normalization.get("mean_dbm"),
                    "std_dbm": None if data.normalization is None else data.normalization.get("std_dbm"),
                    "evaluation_mode": "autoregressive_rollout",
                    "max_horizon": max_horizon,
                    "stored_horizons": horizons,
                },
            )

    return aggregate_rows, frequency_rows, band_rows

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained VanillaLSTM checkpoint")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path template (use {chunk_id} for per-chunk substitution)")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out, _ = prepare_output_dirs(config, "VanillaLSTM")
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
    bands = load_band_definitions(config)

    total_start_time = timestamp_utc()
    total_start = time.perf_counter()
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for chunk in chunk_specs(config):
        print(f"Evaluating VanillaLSTM for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        if args.checkpoint:
            ckpt_path = Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
        else:
            ckpt_path = out / "checkpoints" / f"{chunk.chunk_id}_vanillalstm.pt"
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
        "VanillaLSTM",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")

    from training.common.plot_forecasts import generate_all_plots
    generate_all_plots(
        results_dir=out,
        model_name=MODEL_NAME,
        bins=(30, 150),
        max_steps=500,
    )


if __name__ == "__main__":
    main()
