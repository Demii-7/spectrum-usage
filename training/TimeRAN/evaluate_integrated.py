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

from momentfm import MOMENTPipeline
from training.common.config import load_config
from training.common.runtime import timestamp_utc
from training.common.results import append_metric_rows, finalize_results, load_band_definitions, prepare_output_dirs
from training.common.data import chunk_specs, load_chunk
from training.common.metrics import absolute_and_squared_errors_dbm
from training.common.windowing import target_rows_for


MODEL_NAME = "timeran"

VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: dict[str, Any], device: torch.device, t_in: int, t_out: int):
    tcfg = config["timeran"]
    variant = tcfg["checkpoint_size"]
    model_name = VARIANT_TO_MODEL.get(variant)
    if model_name is None:
        raise ValueError(f"Unknown checkpoint_size: {variant}")

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": t_out,
            "seq_len": t_in,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
    )
    model.init()
    model = model.to(device)
    return model


def predict_timeran(model: nn.Module, device: torch.device,
                    full_x: np.ndarray, target_rows: np.ndarray,
                    horizon: int, t_in: int, batch_size: int) -> np.ndarray:
    origins = target_rows - horizon
    inputs = np.stack(
        [full_x[o - t_in + 1 : o + 1] for o in origins],
        axis=0,
    ).astype(np.float32)
    inputs = inputs.transpose(0, 2, 1)

    loader = DataLoader(torch.from_numpy(inputs).float(), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            input_mask = torch.ones(x.shape[0], x.shape[-1], device=device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    model_output = model(x_enc=x, input_mask=input_mask)
            else:
                model_output = model(x_enc=x, input_mask=input_mask)
            preds.append(model_output.forecast[:, :, horizon - 1].cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands, out: Path, checkpoint_path: Path):
    tcfg = config["timeran"]
    batch_size = int(tcfg["batch_size"])
    lookback = int(config["windowing"]["lookback"])
    min_history = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm
    t_in = lookback
    max_horizon = max(horizons)

    device = device_for()
    model = build_model(config, device, t_in, max_horizon)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        min_needed = max(horizon + t_in - 1, min_history)
        for split_name in test_splits:
            split = data.splits[split_name]
            full_x = np.vstack([train, split.model_input]).astype(np.float32)
            full_raw = np.vstack([train_raw, split.raw_dbm]).astype(np.float32)
            history_offset = len(train)

            target_rows = target_rows_for(
                len(split.raw_dbm), history_offset, horizon,
                t_in, min_needed,
            )
            if len(target_rows) == 0:
                continue

            pred = predict_timeran(
                model, device, full_x, target_rows,
                horizon, t_in, batch_size,
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
                freqs=data.frequencies,
                abs_err=abs_err,
                sq_err=sq_err,
                bands=bands,
            )

    return aggregate_rows, frequency_rows, band_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TimeRAN checkpoint")
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
        print(f"Evaluating TimeRAN for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        if args.checkpoint:
            ckpt_path = Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
        else:
            ckpt_path = out / "checkpoints" / f"{chunk.chunk_id}_timeran.pt"
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
        "TimeRAN",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
