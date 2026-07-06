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


if __name__ == "__main__":
    main()
