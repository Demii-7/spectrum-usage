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

from dataset import SpectrumFrameDataset, _colormap, _make_frames, _normalize, _pad_w
from model import SwinSTB3D
from utils import invert_colormap
from training.common.config import load_config
from training.common.runtime import timestamp_utc
from training.common.results import append_metric_rows, finalize_results, load_band_definitions, prepare_output_dirs
from training.common.data import chunk_specs, load_chunk
from training.common.metrics import absolute_and_squared_errors_dbm


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


def evaluate_chunk(config: dict[str, Any], chunk, bands, out: Path, checkpoint_path: Path):
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

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = ckpt["normalization_stats"]
    model = SwinSTB3D(ckpt["model_config"]).to(device_for(config))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepSPred checkpoint")
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
        print(f"Evaluating DeepSPred for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        if args.checkpoint:
            ckpt_path = Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
        else:
            ckpt_path = out / "checkpoints" / f"{chunk.chunk_id}_deepspred.pt"
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
        "DeepSPred",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
