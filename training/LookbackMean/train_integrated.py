"""LookbackMean integrated training — parameter-free, no-op training loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.config import load_config
from training.common.runtime import epoch_log_row, timestamp_utc
from training.common.results import prepare_output_dirs
from training.common.data import chunk_specs, load_chunk
from training.common.model_factory import SUPPORTED_MODELS, build_model


MODEL_NAME = None  # set per chunk from config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = str(config["training"]["model_name"]).lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Integrated training supports {sorted(SUPPORTED_MODELS)}, got {model_name!r}."
        )

    train_cfg = config[model_name]["train"]
    val_fraction = float(train_cfg.get("val_fraction", 0.1))

    if args.output_dir is not None:
        run_dir = args.output_dir
    else:
        exp_name = args.name or f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out, checkpoints = prepare_output_dirs(run_dir)

    for chunk in chunk_specs(config):
        print(f"Training {model_name} for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")

        data = load_chunk(config, chunk, val_fraction=val_fraction)
        train = data.splits[data.train_split].model_input

        model = build_model(model_name=model_name, config=config, train_data=train)

        # LookbackMean is parameter-free — skip the training loop.
        # Save a checkpoint with the model state (empty params) and data metadata
        # so the shared evaluation_integrated.py can load and evaluate it.
        import torch
        training_start_time = timestamp_utc()
        t_start = time.perf_counter()

        # "Train" = 1 epoch of nothing, just to produce a log row
        epoch_start_time = timestamp_utc()
        epoch_start = time.perf_counter()
        epoch_duration = time.perf_counter() - epoch_start
        total_time = time.perf_counter() - t_start

        log_row = epoch_log_row(
            epoch=1,
            train_loss=0.0,
            val_loss=0.0,
            epoch_start_time=epoch_start_time,
            epoch_end_time=timestamp_utc(),
            epoch_duration_sec=epoch_duration,
            learning_rate=0.0,
        )
        pd.DataFrame([log_row]).to_csv(out / f"{chunk.chunk_id}_training_log.csv", index=False)

        summary_lines = [
            f"Model: {model_name}",
            f"Chunk: {chunk.chunk_id}",
            "Epochs completed: 1 (parameter-free baseline, no training)",
            f"Total time: {total_time:.2f} seconds",
        ]
        (out / f"{chunk.chunk_id}_training_summary.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "normalization": data.normalization,
                "frequencies": data.frequencies,
                "training_results": {
                    "model_name": model_name,
                    "epochs_completed": 1,
                    "best_epoch": 1,
                    "best_val_loss": 0.0,
                    "train_loss_at_best_epoch": 0.0,
                    "best_train_loss": 0.0,
                    "final_train_loss": 0.0,
                    "final_val_loss": 0.0,
                    "average_train_loss": 0.0,
                    "average_val_loss": 0.0,
                    "best_epoch_loss_gap": 0.0,
                    "average_epoch_time_sec": epoch_duration,
                    "fastest_epoch_time_sec": epoch_duration,
                    "slowest_epoch_time_sec": epoch_duration,
                    "training_start_time": training_start_time,
                    "training_end_time": timestamp_utc(),
                    "training_duration_sec": total_time,
                },
            },
            checkpoints / f"{chunk.chunk_id}_{model_name}.pt",
        )
        print(f"  Saved checkpoint (parameter-free, no training needed)")


if __name__ == "__main__":
    main()
