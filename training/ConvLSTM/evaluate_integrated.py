from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import ConvLSTMPredictor
from training.common.config import load_config
from training.common.forecast_export import export_map_forecasts
from training.common.integrated import finalize_results, prepare_output_dirs, timestamp_utc
from training.common.interpolated_map import (
    denormalize_map,
    load_interpolated_map_npz,
    prediction_start_row,
)
from training.common.data import (
    chunk_specs,
    clean_interpolated_map,
    load_chunk,
)
from training.common.metrics import absolute_and_squared_errors_dbm
from training.common.plot_forecasts import generate_all_plots
from training.common.results import append_metric_rows, load_band_definitions
from training.common.windowing import selected_horizon_index


MODEL_NAME = "convlstm"


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_map_for_path(path: str | Path, map_key: str) -> tuple[np.ndarray, dict[str, Any]]:
    data, metadata = load_interpolated_map_npz(path, map_key)
    print(f"[load_map_for_path] Loaded map: shape {data.shape}")
    data = clean_interpolated_map(data, train_ratio=0.8, fit_on_train_only=True)
    return data, metadata


def build_model_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    ccfg = config["convlstm"]
    reference_site = str(config["data"].get("reference_site", "CC2"))
    return {
        "data": {"n_nodes": 1, "n_bins_per_node": n_bins, "node_names": [reference_site]},
        "windowing": {
            "input_sequence_length": int(ccfg.get("input_sequence_length", config["windowing"]["lookback"])),
            "prediction_horizon": int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"]))),
        },
        "model": ccfg["model"],
    }


def build_map_model_config(config: dict[str, Any], n_freq: int, grid_h: int, grid_w: int) -> dict[str, Any]:
    ccfg = config["convlstm"]
    model_cfg = dict(ccfg["model"])
    model_cfg["input_channels"] = n_freq
    return {
        "data": {
            "n_nodes": 1,
            "n_bins_per_node": 1,
            "node_names": ["map"],
            "grid_height": grid_h,
            "grid_width": grid_w,
            "n_freq_bins": n_freq,
        },
        "windowing": {
            "input_sequence_length": int(ccfg.get("input_sequence_length", config["windowing"]["lookback"])),
            "prediction_horizon": int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"]))),
        },
        "model": model_cfg,
    }


def autoregressive_predict_for_origins_convlstm(
    model: ConvLSTMPredictor,
    full_x: np.ndarray,
    origin_rows: np.ndarray,
    max_horizon: int,
    lookback: int,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Autoregressive rollout for ConvLSTM CSV mode.

    For each origin row s, initialise with ``full_x[s : s + lookback]`` and
    iteratively predict one step ahead, appending the prediction and sliding
    the lookback window forward.

    Returns:
        predictions_by_horizon[h] with shape ``(num_origins, n_bins)``.
    """
    device = next(model.parameters()).device

    current_windows = np.stack(
        [full_x[s : s + lookback] for s in origin_rows],
        axis=0,
    ).astype(np.float32)
    current_windows = current_windows[:, :, None, None, :]

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

                next_pred = output[:, selected_horizon_index(1), :, :, :]

                rollout_preds.append(next_pred.cpu().numpy())

                window = torch.cat(
                    [window[:, 1:, :, :, :], next_pred.unsqueeze(1)],
                    dim=1,
                )

            for h, pred_h in enumerate(rollout_preds, start=1):
                predictions_by_horizon[h].append(pred_h)

    return {
        h: np.concatenate(parts, axis=0)[:, 0, 0, :].astype(np.float32)
        for h, parts in predictions_by_horizon.items()
    }


def autoregressive_predict_for_origins_map(
    model: ConvLSTMPredictor,
    full_x: np.ndarray,
    origin_rows: np.ndarray,
    max_horizon: int,
    horizons: list[int],
    lookback: int,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Autoregressive rollout for ConvLSTM map mode using predict_step().

    For each origin, initializes a 60-step window from ground truth, then
    iteratively calls predict_step() to generate one future map at a time.
    The window is updated each step (drop oldest, append prediction) so the
    encoder always re-encodes the latest 60 maps.

    Only horizons in ``horizons`` are saved.

    Returns:
        predictions_by_horizon[h] with shape ``(num_origins, F, H, W)``.
    """
    device = next(model.parameters()).device

    current_windows = np.stack(
        [full_x[s : s + lookback] for s in origin_rows],
        axis=0,
    ).astype(np.float32)

    predictions_by_horizon: dict[int, list[np.ndarray]] = {
        h: [] for h in horizons
    }

    model.eval()

    with torch.no_grad():
        for start in range(0, len(current_windows), batch_size):
            window = torch.from_numpy(current_windows[start : start + batch_size]).float().to(device)

            for h in range(1, max_horizon + 1):
                next_pred = model.predict_step(window)

                if h in horizons:
                    predictions_by_horizon[h].append(next_pred.cpu().numpy())

                window = torch.cat(
                    [window[:, 1:, :, :, :], next_pred.unsqueeze(1)],
                    dim=1,
                )

    return {
        h: np.concatenate(parts, axis=0).astype(np.float32)
        for h, parts in predictions_by_horizon.items()
    }


def evaluate_map_mode(config: dict[str, Any], out: Path, checkpoint_path: Path) -> None:
    data_cfg = config["data"]
    ccfg = config["convlstm"]
    map_cfg = ccfg.get("interpolated_map", {})
    train_map_path = data_cfg.get("train_map_path") or map_cfg.get("map_path")
    test_map_path = data_cfg.get("test_map_path") or train_map_path
    map_key = str(data_cfg.get("map_key") or map_cfg.get("map_key", "map_db"))
    if not train_map_path:
        raise ValueError("Map mode requires data.train_map_path or convlstm.interpolated_map.map_path")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm_stats = ckpt.get("normalization_stats")
    model_config = ckpt.get("model_config")

    test_raw, test_meta = load_map_for_path(test_map_path, map_key)

    if norm_stats and norm_stats.get("method") == "zscore_per_frequency":
        test_x = ((test_raw - norm_stats["mean"]) / norm_stats["std"]).astype(np.float32)
    else:
        test_x = test_raw.astype(np.float32)

    _npz_full = np.load(test_map_path, allow_pickle=True)
    site_data_db = _npz_full["site_data_db"].astype(np.float32)
    raw_site_data_db = _npz_full["raw_site_data_db"].astype(np.float32)
    lon_grid = _npz_full["lon_grid"].astype(np.float64)
    lat_grid = _npz_full["lat_grid"].astype(np.float64)
    site_lons = _npz_full["site_lons"].astype(np.float64)
    site_lats = _npz_full["site_lats"].astype(np.float64)
    site_names = list(_npz_full["site_names"])
    del _npz_full

    site_grid_positions = []
    for slon, slat in zip(site_lons, site_lats):
        dist_sq = (lon_grid - slon) ** 2 + (lat_grid - slat) ** 2
        site_grid_positions.append(tuple(np.unravel_index(np.argmin(dist_sq), dist_sq.shape)))

    F, H, W = test_x.shape[1], test_x.shape[2], test_x.shape[3]
    lookback = int(ccfg.get("input_sequence_length", config["windowing"]["lookback"]))
    prediction_horizon = int(ccfg.get("prediction_horizon", max(config["windowing"]["horizons"])))
    batch_size = int(ccfg.get("batch_size", 32))

    device = device_for()
    model = ConvLSTMPredictor(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    start_idx = prediction_start_row(config, len(test_x))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    max_horizon = max(horizons)

    chunk_cfg = data_cfg.get("chunks", [{}])[0]
    raw_chunk_id = data_cfg.get("chunk_id")
    if raw_chunk_id is None:
        raw_chunk_id = chunk_cfg.get("id", "map")
    chunk_id = str(raw_chunk_id)
    start_mhz = float(chunk_cfg.get("start_mhz", 0))
    end_mhz = float(chunk_cfg.get("end_mhz", 0))
    freqs = list(range(F))
    bands = load_band_definitions(config)

    total_start = time.perf_counter()
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    predictions_by_horizon: dict[int, np.ndarray] = {}
    targets_by_horizon: dict[int, np.ndarray] = {}
    target_rows_by_horizon: dict[int, np.ndarray] = {}

    first_target = max(start_idx, lookback + max_horizon - 1)
    origin_rows = np.arange(first_target - lookback - max_horizon + 1, len(test_x) - lookback - max_horizon + 1, dtype=np.int64)
    if len(origin_rows) > 0:
        all_preds = autoregressive_predict_for_origins_map(
            model, test_x, origin_rows, max_horizon, horizons, lookback, batch_size,
        )
        for horizon in horizons:
            pred_norm = all_preds[horizon]
            pred = denormalize_map(pred_norm, norm_stats)
            target_rows = origin_rows + lookback + horizon - 1
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

            # Approach 2: per-site evaluation using interpolated site_data_db from npz
            for site_idx, site_name in enumerate(site_names):
                h_pos, w_pos = site_grid_positions[site_idx]
                pred_site = pred[:, :, h_pos, w_pos]
                target_site = site_data_db[site_idx, target_rows]
                _, abs_err_site, sq_err_site = absolute_and_squared_errors_dbm(
                    pred_site, target_site, normalization=None)
                append_metric_rows(
                    aggregate_rows, frequency_rows, band_rows,
                    chunk_id=chunk_id,
                    start_mhz=start_mhz,
                    end_mhz=end_mhz,
                    split_name=f"test_site_npz_{site_name}",
                    horizon=horizon,
                    model=MODEL_NAME,
                    target_rows=target_rows,
                    history_offset=0,
                    freqs=freqs,
                    abs_err=abs_err_site,
                    sq_err=sq_err_site,
                    bands=bands,
                )

            # Approach 3: per-site evaluation using raw CSV data
            for site_idx, site_name in enumerate(site_names):
                h_pos, w_pos = site_grid_positions[site_idx]
                pred_site = pred[:, :, h_pos, w_pos]
                target_raw = raw_site_data_db[site_idx, target_rows]
                _, abs_err_raw, sq_err_raw = absolute_and_squared_errors_dbm(
                    pred_site, target_raw, normalization=None)
                append_metric_rows(
                    aggregate_rows, frequency_rows, band_rows,
                    chunk_id=chunk_id,
                    start_mhz=start_mhz,
                    end_mhz=end_mhz,
                    split_name=f"test_site_raw_{site_name}",
                    horizon=horizon,
                    model=MODEL_NAME,
                    target_rows=target_rows,
                    history_offset=0,
                    freqs=freqs,
                    abs_err=abs_err_raw,
                    sq_err=sq_err_raw,
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
            "model": "ConvLSTM",
            "train_map_path": str(train_map_path),
            "test_map_path": str(test_meta["path"]),
            "map_key": map_key,
            "chunk_id": chunk_id,
            "prediction_start_row": config.get("evaluation", {}).get("prediction_start_row"),
            "test_shape_tf_hw": list(test_raw.shape),
            "normalization": None if norm_stats is None else norm_stats["method"],
            "mean_dbm": None if norm_stats is None else np.squeeze(norm_stats["mean"]),
            "std_dbm": None if norm_stats is None else np.squeeze(norm_stats["std"]),
            "frequencies_mhz": list(range(F)),
            "train_map_metadata": (ckpt.get("train_map_metadata") or {}).get("metadata"),
            "test_map_metadata": test_meta.get("metadata"),
        },
    )

    total_run = time.perf_counter() - total_start
    finalize_results(
        out,
        "ConvLSTM",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {timestamp_utc()}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")

    generate_all_plots(
        results_dir=out,
        model_name=MODEL_NAME,
        bins=(30, 150),
        max_steps=500,
    )


def evaluate_csv_chunk(config: dict[str, Any], chunk, bands, out: Path, checkpoint_path: Path):
    ccfg = config["convlstm"]
    lookback = int(ccfg.get("input_sequence_length", config["windowing"]["lookback"]))
    min_history = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    max_horizon = max(horizons)
    batch_size = int(ccfg.get("batch_size", 32))
    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm

    model_config = build_model_config(config, train.shape[1])
    device = device_for()
    model = ConvLSTMPredictor(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    export_payloads: dict[str, dict[str, Any]] = {}

    for split_name in test_splits:
        split = data.splits[split_name]

        full_x = np.vstack([train, split.model_input]).astype(np.float32)
        full_raw = np.vstack([train_raw, split.raw_dbm]).astype(np.float32)
        n_test = len(split.raw_dbm)
        history_offset = len(train)

        # All origins whose lookback window fits within full_x.
        # Targets will be filtered per-horizon below.
        origin_min = max(0, max(history_offset, min_history) - lookback)
        origin_max = len(full_x) - lookback
        origin_rows = np.arange(origin_min, origin_max, dtype=np.int64)

        if len(origin_rows) == 0:
            print(f"  Not enough rows for split {split_name}; skipping")
            continue

        all_preds = autoregressive_predict_for_origins_convlstm(
            model=model,
            full_x=full_x,
            origin_rows=origin_rows,
            max_horizon=max_horizon,
            lookback=lookback,
            batch_size=batch_size,
        )

        for horizon in horizons:
            target_rows = origin_rows + lookback + horizon - 1
            in_test = (target_rows >= history_offset) & (target_rows < history_offset + n_test)

            if not in_test.any():
                print(f"  No valid targets for h={horizon} in split {split_name}; skipping")
                continue

            pred = all_preds[horizon][in_test]
            target = full_raw[target_rows[in_test]]
            local_target_rows = target_rows[in_test].astype(np.int64)

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
                target_rows=local_target_rows,
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
                "model": "ConvLSTM",
                "split_name": split_name,
                "train_split": data.train_split,
                "test_split": split_name,
                "chunk_id": chunk.chunk_id,
                "start_mhz": chunk.start_mhz,
                "end_mhz": chunk.end_mhz,
                "lookback": lookback,
                "batch_size": batch_size,
                "history_offset": history_offset,
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
    parser = argparse.ArgumentParser(description="Evaluate a trained ConvLSTM checkpoint")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path (use {chunk_id} for per-chunk substitution in CSV mode)")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out, _ = prepare_output_dirs(config, "ConvLSTM")
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)

    if config["data"].get("train_map_path") or config["convlstm"].get("interpolated_map", {}).get("enabled", False):
        print("Interpolated-map mode enabled — evaluating on map data.")
        ckpt_path = args.checkpoint or out / "checkpoints" / "interpolated_map_convlstm.pt"
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
        print(f"Evaluating ConvLSTM for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        if args.checkpoint:
            ckpt_path = Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
        else:
            ckpt_path = out / "checkpoints" / f"{chunk.chunk_id}_convlstm.pt"
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
        "ConvLSTM",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")

    generate_all_plots(
        results_dir=out,
        model_name=MODEL_NAME,
        bins=(30, 150),
        max_steps=500,
    )


if __name__ == "__main__":
    main()
