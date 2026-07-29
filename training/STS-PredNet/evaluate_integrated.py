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

from stsprednet import STSPredNet
from dataset import required_history, resolve_branch_config
from training.common.config import load_config
from training.common.forecast_export import export_map_forecasts
from training.common.runtime import timestamp_utc
from training.common.results import finalize_results, prepare_output_dirs
from training.common.data import chunk_specs, load_chunk
from training.common.metrics import absolute_and_squared_errors_dbm
from training.common.results import append_metric_rows, load_band_definitions
from training.common.windowing import filter_target_rows
from train_integrated import to_sts_layout


MODEL_NAME = "stsprednet"


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rollout_required_history(branches, horizon: int) -> int:
    """History needed at the first recursive step for an exact-horizon target."""
    return required_history(branches) + int(horizon) - 1


def validation_fraction(config: dict[str, Any]) -> float:
    section = config["stsprednet"]
    train = section.get("train") or section.get("training") or section
    return float(train.get("val_fraction", 0.1))


def build_model_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    scfg = config["stsprednet"]
    return {
        "model": {
            "input_channels": scfg["model"]["input_channels"],
            "map_height": scfg["model"]["map_height"],
            "map_width": n_bins,
            "num_layers": scfg["model"]["num_layers"],
            "hidden_dim": scfg["model"]["hidden_dim"],
            "kernel_size": list(scfg["model"]["kernel_size"]),
            "output_activation": scfg["model"]["output_activation"],
            "fusion_weight_shape": scfg["model"]["fusion_weight_shape"],
        },
        "branches": resolve_branch_config(config),
    }


def build_map_model_config(config: dict[str, Any], n_freq: int, grid_h: int, grid_w: int) -> dict[str, Any]:
    scfg = config["stsprednet"]
    return {
        "model": {
            "input_channels": n_freq,
            "map_height": grid_h,
            "map_width": grid_w,
            "num_layers": scfg["model"]["num_layers"],
            "hidden_dim": scfg["model"]["hidden_dim"],
            "kernel_size": list(scfg["model"]["kernel_size"]),
            "output_activation": scfg["model"]["output_activation"],
            "fusion_weight_shape": scfg["model"]["fusion_weight_shape"],
        },
        "branches": resolve_branch_config(config),
    }


def predict_recursive(model: STSPredNet, device: torch.device,
                      full_x: np.ndarray, target_rows: np.ndarray,
                      horizon: int, lc: int | None = None, lp: int | None = None,
                      period_interval: int | None = None, branches=None) -> np.ndarray:
    branches = branches or {
        "use_closeness": True, "use_period": True, "use_trend": False,
        "lc": lc, "lp": lp, "lq": 1, "period_interval": period_interval,
        "trend_interval": 1, "prediction_offset": 1,
    }
    branches = resolve_branch_config(branches)
    preds = []

    for target_row in target_rows:
        origin = target_row - horizon
        generated = {}

        for step in range(1, horizon + 1):
            current_target = origin + step

            def frame(index):
                return generated[index] if index in generated else full_x[index]

            inputs = []
            specs = (
                ("closeness", range(current_target - branches["lc"], current_target)),
                ("period", [current_target - p * branches["period_interval"] for p in range(branches["lp"], 0, -1)]),
                ("trend", [current_target - q * branches["trend_interval"] for q in range(branches["lq"], 0, -1)]),
            )
            for name, indices in specs:
                if not branches[f"use_{name}"]:
                    inputs.append(None)
                    continue
                seq = np.stack([frame(i) for i in indices])[:, None, None, :].astype(np.float32)
                inputs.append(torch.from_numpy(seq).unsqueeze(0).to(device))

            with torch.no_grad():
                pred = model(*inputs)
            pred_np = pred.cpu().numpy()[0, 0, 0, :]
            generated[current_target] = pred_np

            if step == horizon:
                preds.append(pred_np)

    return np.stack(preds, axis=0).astype(np.float32)


def predict_recursive_map(
    model: STSPredNet,
    device: torch.device,
    full_x: np.ndarray,
    target_rows: np.ndarray,
    horizon: int,
    lc: int | None = None,
    lp: int | None = None,
    period_interval: int | None = None,
    branches=None,
) -> np.ndarray:
    branches = resolve_branch_config(branches or {
        "use_closeness": True, "use_period": True, "use_trend": False,
        "lc": lc, "lp": lp, "lq": 1, "period_interval": period_interval,
        "trend_interval": 1, "prediction_offset": 1,
    })
    preds = []
    for target_row in target_rows:
        origin = target_row - horizon
        generated = {}
        for step in range(1, horizon + 1):
            current_target = origin + step
            def frame(index):
                return generated[index] if index in generated else full_x[index]
            specs = (
                ("closeness", range(current_target - branches["lc"], current_target)),
                ("period", [current_target - p * branches["period_interval"] for p in range(branches["lp"], 0, -1)]),
                ("trend", [current_target - q * branches["trend_interval"] for q in range(branches["lq"], 0, -1)]),
            )
            inputs = []
            for name, indices in specs:
                seq = None if not branches[f"use_{name}"] else torch.from_numpy(
                    np.stack([frame(i) for i in indices]).astype(np.float32)).unsqueeze(0).to(device)
                inputs.append(seq)
            with torch.no_grad():
                pred = model(*inputs)
            pred_np = pred.cpu().numpy()[0].astype(np.float32)
            generated[current_target] = pred_np
            if step == horizon:
                preds.append(pred_np)
    return np.stack(preds, axis=0).astype(np.float32)


def evaluate_map_mode(config: dict[str, Any], out: Path, checkpoint_path: Path) -> None:
    data_cfg = config["data"]
    train_map_path = data_cfg.get("train_map_path")
    test_map_path = data_cfg.get("test_map_path") or train_map_path
    map_key = str(data_cfg.get("map_key", "map_db"))
    if not train_map_path:
        raise ValueError("Map mode requires data.train_map_path")

    train_raw, train_meta = load_interpolated_map_npz(train_map_path, map_key)
    test_raw, test_meta = load_interpolated_map_npz(test_map_path, map_key)
    train_x, test_x, norm_stats = normalize_map_by_frequency(
        train_raw,
        test_raw,
        enabled=bool(config.get("preprocessing", {}).get("normalize", True)),
    )

    chunk_id = str(data_cfg.get("chunk_id", "powder_map"))

    scfg = config["stsprednet"]
    branches = resolve_branch_config(config, len(test_x))
    _, n_freq, grid_h, grid_w = train_x.shape
    model_config = build_map_model_config(config, n_freq, grid_h, grid_w)
    device = device_for()
    model = STSPredNet(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    horizons = [int(h) for h in config["windowing"]["horizons"]]
    start_idx = prediction_start_row(config, len(test_x))

    chunk_cfg = data_cfg.get("chunks", [{}])[0]
    start_mhz = float(chunk_cfg.get("start_mhz", 0))
    end_mhz = float(chunk_cfg.get("end_mhz", 0))
    freqs = list(range(n_freq))
    bands = load_band_definitions(config)

    total_start = time.perf_counter()
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    predictions_by_horizon: dict[int, np.ndarray] = {}
    targets_by_horizon: dict[int, np.ndarray] = {}
    target_rows_by_horizon: dict[int, np.ndarray] = {}
    for horizon in horizons:
        min_needed = required_history(branches, horizon)
        first_target = max(start_idx, min_needed)
        target_rows = np.arange(first_target, len(test_x), dtype=np.int64)
        if len(target_rows) == 0:
            continue
        pred_norm = predict_recursive_map(model, device, test_x, target_rows, horizon, branches=branches)
        pred = denormalize_map(pred_norm, norm_stats)
        target = test_raw[target_rows].astype(np.float32)
        predictions_by_horizon[horizon] = pred
        targets_by_horizon[horizon] = target
        target_rows_by_horizon[horizon] = target_rows

        _, abs_err, sq_err = absolute_and_squared_errors_dbm(pred, target, normalization=None)
        abs_err = np.mean(abs_err, axis=(2, 3))
        sq_err = np.mean(sq_err, axis=(2, 3))
        append_metric_rows(
            aggregate_rows, frequency_rows, band_rows,
            chunk_id=chunk_id,
            start_mhz=start_mhz,
            end_mhz=end_mhz,
            split_name="test",
            horizon=horizon,
            model=MODEL_NAME,
            target_rows=target_rows,
            history_offset=0,
            freqs=freqs,
            abs_err=abs_err,
            sq_err=sq_err,
            bands=bands,
        )

    export_map_forecasts(
        out,
        chunk_id=chunk_id,
        model_name=MODEL_NAME,
        predictions_by_horizon=predictions_by_horizon,
        targets_by_horizon=targets_by_horizon,
        target_rows_by_horizon=target_rows_by_horizon,
        metadata={
            "model": "STS-PredNet",
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

    total_run = time.perf_counter() - total_start
    finalize_results(
        out,
        "STS-PredNet",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {timestamp_utc()}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


def evaluate_csv_chunk(config: dict[str, Any], chunk, bands, out: Path, checkpoint_path: Path):
    scfg = config["stsprednet"]
    branches = resolve_branch_config(config)
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    data = load_chunk(config, chunk, val_fraction=validation_fraction(config))
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm

    model_config = build_model_config(config, train.shape[1])
    device = device_for()
    model = STSPredNet(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        branch_history = required_history(branches, horizon)
        min_needed = branch_history
        for split_name in test_splits:
            split = data.splits[split_name]
            split_x = split.model_input.astype(np.float32)
            split_raw = split.raw_dbm.astype(np.float32)
            target_rows = np.arange(
                min_needed,
                len(split_raw),
                dtype=np.int64,
            )
            target_rows = filter_target_rows(
                target_rows,
                history=branch_history,
                segments=split.segments,
            )
            if len(target_rows) == 0:
                print(f"  No valid target rows for {chunk.chunk_id} {split_name} h={horizon}")
                continue

            pred = predict_recursive(
                model, device, split_x, target_rows, horizon,
                branches=branches,
            )
            target = split_raw[target_rows]
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
                target_rows=target_rows + int(split.row_start),
                history_offset=int(split.row_start),
                freqs=data.frequencies,
                abs_err=abs_err,
                sq_err=sq_err,
                bands=bands,
            )

    return aggregate_rows, frequency_rows, band_rows


def evaluate_chunk(config: dict[str, Any], chunk, bands, out: Path, checkpoint_path: Path):
    """Evaluate common-pipeline vector or map splits with STS branch sampling."""
    branches = resolve_branch_config(config)
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    data = load_chunk(config, chunk, val_fraction=validation_fraction(config))
    train = data.splits[data.train_split].model_input

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("model_config")
    if model_config is None:
        model_config = (
            build_map_model_config(config, train.shape[-1], train.shape[1], train.shape[2])
            if train.ndim == 4
            else build_model_config(config, train.shape[1])
        )
    device = device_for()
    model = STSPredNet(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    test_splits = config["data"].get("test_splits", [data.test_split])

    for split_name in test_splits:
        split = data.splits[split_name]
        split_x = to_sts_layout(split.model_input)
        raw_model_layout = to_sts_layout(split.raw_dbm)
        predictions_by_horizon: dict[int, np.ndarray] = {}
        targets_by_horizon: dict[int, np.ndarray] = {}
        target_rows_by_horizon: dict[int, np.ndarray] = {}

        for horizon in horizons:
            history = rollout_required_history(branches, horizon)
            target_rows = np.arange(history, len(split_x), dtype=np.int64)
            target_rows = filter_target_rows(target_rows, history, split.segments)
            if len(target_rows) == 0:
                print(f"  No valid target rows for {chunk.chunk_id} {split_name} h={horizon}")
                continue

            pred_norm = predict_recursive_map(
                model, device, split_x, target_rows, horizon, branches=branches,
            )
            target_raw = raw_model_layout[target_rows]
            pred_frequency_last = np.moveaxis(pred_norm, 1, -1)
            target_frequency_last = np.moveaxis(target_raw, 1, -1)
            pred_dbm_frequency_last, abs_err, sq_err = absolute_and_squared_errors_dbm(
                pred_frequency_last, target_frequency_last, data.normalization,
            )
            abs_err = np.mean(abs_err, axis=(1, 2))
            sq_err = np.mean(sq_err, axis=(1, 2))
            global_rows = target_rows + int(split.row_start)
            append_metric_rows(
                aggregate_rows, frequency_rows, band_rows,
                chunk_id=chunk.chunk_id,
                start_mhz=chunk.start_mhz,
                end_mhz=chunk.end_mhz,
                split_name=split_name,
                horizon=horizon,
                model=MODEL_NAME,
                target_rows=global_rows,
                history_offset=int(split.row_start),
                freqs=data.frequencies,
                abs_err=abs_err,
                sq_err=sq_err,
                bands=bands,
                feature_labels=data.feature_labels,
            )
            predictions_by_horizon[horizon] = np.moveaxis(
                pred_dbm_frequency_last, -1, 1,
            )
            targets_by_horizon[horizon] = target_raw
            target_rows_by_horizon[horizon] = target_rows

        if predictions_by_horizon and split_x.ndim == 4:
            export_map_forecasts(
                out,
                chunk_id=chunk.chunk_id,
                model_name=MODEL_NAME,
                predictions_by_horizon=predictions_by_horizon,
                targets_by_horizon=targets_by_horizon,
                target_rows_by_horizon=target_rows_by_horizon,
                metadata={
                    "model": "STS-PredNet",
                    "split": split_name,
                    "checkpoint": str(checkpoint_path),
                    "branches": branches,
                    "normalization": data.normalization,
                },
            )

    return aggregate_rows, frequency_rows, band_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained STS-PredNet checkpoint")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path (use {chunk_id} for per-chunk substitution in CSV mode)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = "stsprednet"
    if args.output_dir is not None:
        run_dir = args.output_dir
    else:
        exp_name = args.name or f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out, checkpoints = prepare_output_dirs(run_dir)

    if config["data"].get("train_map_path"):
        print("Interpolated-map mode enabled — evaluating STS-PredNet on map data.")
        ckpt_path = args.checkpoint or out / "checkpoints" / f'{config["data"].get("chunk_id", "powder_map")}_stsprednet.pt'
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
        print(f"Evaluating STS-PredNet for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        if args.checkpoint:
            ckpt_path = Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
        else:
            ckpt_path = out / "checkpoints" / f"{chunk.chunk_id}_stsprednet.pt"
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
        "STS-PredNet",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
