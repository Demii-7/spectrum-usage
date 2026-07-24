from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import SpectrumFrameDataset, _colormap, _make_frames, _normalize, _pad_w  # noqa: E402
from model import SwinSTB3D  # noqa: E402
from utils import invert_colormap  # noqa: E402
from config_support import resolve_deepspred_config  # noqa: E402
from training.common.config import load_config  # noqa: E402
from training.common.data_loader import data_loader_kwargs  # noqa: E402
from training.common.runtime import epoch_log_row, timestamp_utc  # noqa: E402
from training.common.results import prepare_output_dirs  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402


MODEL_NAME = "deepspred"


def device_for(config: dict[str, Any]) -> torch.device:
    requested = str(resolve_deepspred_config(config, 1)["train"]["device"])
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_runner_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    return resolve_deepspred_config(config, n_bins)


def frames_from_raw(raw: np.ndarray, config: dict[str, Any], vmin: float, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    runner = build_runner_config(config, raw.shape[1])
    minutes_per_frame = runner["frames"]["minutes_per_frame"]
    normalized = _normalize(raw, vmin, vmax)
    rgb = _colormap(normalized, runner["preprocessing"]["colormap"])
    frames_orig = _make_frames(rgb, minutes_per_frame)
    frames_pad = _pad_w(frames_orig, runner["frames"]["w_pad"])
    return frames_pad, frames_orig


def _frame_segments(segments, start: int, end: int, minutes_per_frame: int):
    if not segments:
        return ()
    result = []
    for segment in segments:
        clipped_start = max(segment.start, start)
        clipped_end = min(segment.end, end)
        frame_start = (clipped_start - start + minutes_per_frame - 1) // minutes_per_frame
        frame_end = (clipped_end - start) // minutes_per_frame
        if frame_start < frame_end:
            result.append(type(segment)(frame_start, frame_end, segment.label))
    return tuple(result)


def build_frame_dataset(
    frames_pad: np.ndarray,
    frames_orig: np.ndarray,
    input_frames: int,
    output_frames: int,
    stride: int,
    segments=(),
) -> SpectrumFrameDataset | None:
    total = input_frames + output_frames
    if len(frames_pad) < total:
        return None
    indices = list(range(0, len(frames_pad) - total + 1, stride))
    if segments:
        indices = [
            index for index in indices
            if any(
                index >= segment.start
                and index + total <= segment.end
                for segment in segments
            )
        ]
    if not indices:
        return None
    return SpectrumFrameDataset(frames_pad, frames_orig, input_frames, output_frames, indices, ["CC2"] * len(indices))


def train_one_model(config: dict[str, Any], train_raw: np.ndarray, segments, checkpoints: Path, out: Path, chunk_id: str, normalization=None, frequencies=None):
    runner = build_runner_config(config, train_raw.shape[1])
    dcfg = runner["train"]
    input_frames = runner["windowing"]["input_frames"]
    output_frames = runner["windowing"]["output_frames"]
    stride = runner["windowing"]["stride"]
    minutes_per_frame = runner["frames"]["minutes_per_frame"]
    split_idx = max(1, int(len(train_raw) * 0.9))
    train_part = train_raw[:split_idx]
    val_part = train_raw[split_idx:]
    if len(val_part) == 0:
        val_part = train_part[-runner["frames"]["minutes_per_frame"] * 2 :]
    vmin = float(train_part.min())
    vmax = float(train_part.max())

    train_pad, train_orig = frames_from_raw(train_part, config, vmin, vmax)
    val_pad, val_orig = frames_from_raw(val_part, config, vmin, vmax)
    train_segments = _frame_segments(segments, 0, split_idx, minutes_per_frame)
    val_segments = _frame_segments(segments, split_idx, len(train_raw), minutes_per_frame)
    train_ds = build_frame_dataset(
        train_pad, train_orig, input_frames, output_frames, stride, train_segments
    )
    val_ds = build_frame_dataset(
        val_pad, val_orig, input_frames, output_frames, stride, val_segments
    )
    if train_ds is None:
        raise ValueError("Not enough frames for DeepSPred training.")

    loader_kwargs = data_loader_kwargs(config.get("data_loader"))
    train_loader = DataLoader(train_ds, batch_size=int(dcfg.get("batch_size", 2)), shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(dcfg.get("batch_size", 2)), shuffle=False, **loader_kwargs) if val_ds is not None else None
    model = SwinSTB3D(runner).to(device_for(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(dcfg.get("learning_rate", 0.001)), weight_decay=float(dcfg.get("weight_decay", 0.05)))
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    log_rows: list[dict[str, Any]] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()

    for epoch in range(1, int(dcfg.get("epochs", 6)) + 1):
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
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        val_loss = train_loss
        if val_loader is not None:
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
        print(f"{chunk_id} epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={epoch_duration:.1f}s")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(dcfg.get("early_stopping_epochs", 4)):
                break

    total_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_training_log.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "model_config": runner,
            "normalization_stats": {"vmin": vmin, "vmax": vmax},
            "normalization": normalization,
            "frequencies": frequencies,
            "representation": {
                "native": "direct_rgb_frames",
                "evaluation_adaptation": "grouped_dbm_via_inverse_colormap",
            },
            "minmax": {"vmin": vmin, "vmax": vmax},
            "colormap": runner["preprocessing"]["colormap"],
            "resolved_config": runner,
            "common_config": config,
            "training_start_time": training_start_time,
            "training_end_time": timestamp_utc(),
            "training_duration_sec": total_time,
        },
        checkpoints / f"{chunk_id}_deepspred.pt",
    )
    return model, runner, {"vmin": vmin, "vmax": vmax}


def train_chunk(config: dict[str, Any], chunk, data, out: Path, checkpoints: Path):
    """Train one loaded shared-pipeline chunk and return the trained artifacts."""
    split = data.splits[data.train_split]
    return train_one_model(
        config, split.raw_dbm, split.segments, checkpoints, out, chunk.chunk_id,
        normalization=data.normalization, frequencies=data.frequencies,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = "deepspred"
    if args.output_dir is not None:
        run_dir = args.output_dir
    else:
        exp_name = args.name or f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out, checkpoints = prepare_output_dirs(run_dir)

    for chunk in chunk_specs(config):
        print(f"Training DeepSPred for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        data = load_chunk(config, chunk)
        train_chunk(config, chunk, data, out, checkpoints)


if __name__ == "__main__":
    main()
