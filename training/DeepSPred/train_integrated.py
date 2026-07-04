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
from training.common.config import load_config  # noqa: E402
from training.common.integrated import epoch_log_row, finalize_results, prepare_output_dirs, timestamp_utc  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.results import append_metric_rows, load_band_definitions  # noqa: E402


MODEL_NAME = "deepspred"


def device_for(config: dict[str, Any]) -> torch.device:
    requested = str(config["deepspred"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_runner_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    dcfg = config["deepspred"]
    return {
        "preprocessing": {
            "colormap": str(dcfg.get("colormap", "jet")),
            "normalization": str(dcfg.get("normalization", "minmax")),
        },
        "frames": {
            "minutes_per_frame": int(dcfg.get("minutes_per_frame", 60)),
            "w_pad": 256,
            "w_orig": n_bins,
        },
        "windowing": {
            "input_frames": int(dcfg.get("input_frames", 1)),
            "output_frames": int(dcfg.get("output_frames", 1)),
            "stride": int(dcfg.get("stride", 1)),
        },
        "model": dict(dcfg["model"]),
    }


def frames_from_raw(raw: np.ndarray, config: dict[str, Any], vmin: float, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    runner = build_runner_config(config, raw.shape[1])
    minutes_per_frame = runner["frames"]["minutes_per_frame"]
    normalized = _normalize(raw, vmin, vmax)
    rgb = _colormap(normalized, runner["preprocessing"]["colormap"])
    frames_orig = _make_frames(rgb, minutes_per_frame)
    frames_pad = _pad_w(frames_orig, runner["frames"]["w_pad"])
    return frames_pad, frames_orig


def build_frame_dataset(frames_pad: np.ndarray, frames_orig: np.ndarray, input_frames: int, output_frames: int, stride: int) -> SpectrumFrameDataset | None:
    total = input_frames + output_frames
    if len(frames_pad) < total:
        return None
    indices = list(range(0, len(frames_pad) - total + 1, stride))
    if not indices:
        return None
    return SpectrumFrameDataset(frames_pad, frames_orig, input_frames, output_frames, indices, ["CC2"] * len(indices))


def train_one_model(config: dict[str, Any], train_raw: np.ndarray, checkpoints: Path, out: Path, chunk_id: str):
    dcfg = config["deepspred"]
    runner = build_runner_config(config, train_raw.shape[1])
    input_frames = runner["windowing"]["input_frames"]
    output_frames = runner["windowing"]["output_frames"]
    stride = runner["windowing"]["stride"]
    split_idx = max(1, int(len(train_raw) * 0.9))
    train_part = train_raw[:split_idx]
    val_part = train_raw[split_idx:]
    if len(val_part) == 0:
        val_part = train_part[-runner["frames"]["minutes_per_frame"] * 2 :]
    vmin = float(train_part.min())
    vmax = float(train_part.max())

    train_pad, train_orig = frames_from_raw(train_part, config, vmin, vmax)
    val_pad, val_orig = frames_from_raw(val_part, config, vmin, vmax)
    train_ds = build_frame_dataset(train_pad, train_orig, input_frames, output_frames, stride)
    val_ds = build_frame_dataset(val_pad, val_orig, input_frames, output_frames, stride)
    if train_ds is None:
        raise ValueError("Not enough frames for DeepSPred training.")

    train_loader = DataLoader(train_ds, batch_size=int(dcfg.get("batch_size", 2)), shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(dcfg.get("batch_size", 2)), shuffle=False) if val_ds is not None else None
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
            "model_config": runner,
            "normalization_stats": {"vmin": vmin, "vmax": vmax},
            "training_start_time": training_start_time,
            "training_end_time": timestamp_utc(),
            "training_duration_sec": total_time,
        },
        checkpoints / f"{chunk_id}_deepspred.pt",
    )
    return model, runner, {"vmin": vmin, "vmax": vmax}


def predict_frame_windows(model: SwinSTB3D, frame_pad: np.ndarray, starts: list[int], batch_size: int) -> np.ndarray:
    x = np.stack([frame_pad[start : start + model.input_frames] for start in starts], axis=0).astype(np.float32)
    x = x.transpose(0, 1, 4, 2, 3)
    loader = DataLoader(torch.from_numpy(x).float(), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(next(model.parameters()).device)
            preds.append(model(batch_x).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands: pd.DataFrame, out: Path):
    dcfg = config["deepspred"]
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train_raw = data.splits[data.train_split].raw_dbm
    runner = build_runner_config(config, train_raw.shape[1])
    frame_height = runner["frames"]["minutes_per_frame"]
    input_frames = runner["windowing"]["input_frames"]
    output_frames = runner["windowing"]["output_frames"]
    max_horizon = max(horizons)
    if max_horizon > frame_height * output_frames:
        raise ValueError("DeepSPred configuration does not cover requested horizons.")

    model, _, stats = train_one_model(config, train_raw, out / "checkpoints", out, chunk.chunk_id)
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for split_name in test_splits:
        split = data.splits[split_name]
        split_raw = split.raw_dbm
        full_raw = np.vstack([train_raw, split_raw]).astype(np.float32)
        frame_pad, _ = frames_from_raw(full_raw, config, stats["vmin"], stats["vmax"])
        train_frame_count = len(train_raw) // frame_height
        total_frames = len(full_raw) // frame_height
        max_start = total_frames - input_frames - output_frames
        starts = list(range(max(train_frame_count - input_frames, 0), max_start + 1))
        if not starts:
            continue
        pred_rgb = predict_frame_windows(model, frame_pad, starts, int(dcfg.get("batch_size", 2)))
        pred_scalar = invert_colormap(pred_rgb.transpose(0, 1, 3, 4, 2), cmap_name=str(dcfg.get("colormap", "jet")))
        pred_dbm_frames = pred_scalar * (stats["vmax"] - stats["vmin"]) + stats["vmin"]

        for horizon in horizons:
            frame_offset = (horizon - 1) // frame_height
            minute_offset = (horizon - 1) % frame_height
            target_rows = np.array(
                [
                    (start + input_frames + frame_offset) * frame_height + minute_offset
                    for start in starts
                ],
                dtype=np.int64,
            )
            pred = pred_dbm_frames[:, frame_offset, minute_offset, :]
            valid = target_rows < len(full_raw)
            pred = pred[valid]
            target_rows = target_rows[valid]
            target = full_raw[target_rows]
            abs_err = np.abs(pred - target)
            sq_err = (pred - target) ** 2
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
                history_offset=len(train_raw),
                freqs=data.frequencies,
                abs_err=abs_err,
                sq_err=sq_err,
                bands=bands,
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
    out, _ = prepare_output_dirs(config, "DeepSPred")
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
        print(f"Training DeepSPred for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        a, f, b = evaluate_chunk(config, chunk, bands, out)
        aggregate_rows.extend(a)
        frequency_rows.extend(f)
        band_rows.extend(b)

    total_run = time.perf_counter() - total_start
    finalize_results(
        out,
        "DeepSPred",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Training start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
