"""Evaluate Autoformer-CSA checkpoints with the shared result pipeline."""

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

from model import AutoformerCSAForecaster, DotConfig  # noqa: E402
from training.common.config import load_config  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.forecasting import forecast  # noqa: E402
from training.common.forecast_export import export_map_forecasts  # noqa: E402
from training.common.metrics import absolute_and_squared_errors_dbm  # noqa: E402
from training.common.results import (  # noqa: E402
    append_metric_rows,
    finalize_results,
    load_band_definitions,
)
from training.common.runtime import device_for, timestamp_utc  # noqa: E402
from training.common.windowing import make_window_batch_array, make_window_starts  # noqa: E402


MODEL_NAME = "autoformer_csa"


def _map_streams(data: np.ndarray) -> tuple[np.ndarray, int, int]:
    if data.ndim != 4:
        raise ValueError(f"Expected map data shaped (T,H,W,F), got {data.shape}")
    time, height, width, features = data.shape
    streams = np.transpose(data, (1, 2, 0, 3)).reshape(height * width, time, features)
    return streams, height, width


def _make_windows(
    data: np.ndarray,
    starts: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, int | None, int | None]:
    if data.ndim == 2:
        from training.common.windowing import make_window_batch_array

        return make_window_batch_array(full_x=data, start_rows=starts, lookback=lookback), None, None
    streams, height, width = _map_streams(data)
    windows = np.stack(
        [streams[grid, start : start + lookback] for grid in range(streams.shape[0]) for start in starts],
        axis=0,
    ).astype(np.float32, copy=False)
    return windows, height, width


def _map_model_layout(
    stream_values: np.ndarray,
    height: int,
    width: int,
    n_origins: int,
) -> np.ndarray:
    features = stream_values.shape[-1]
    values = stream_values.reshape(height * width, n_origins, features)
    return values.transpose(1, 2, 0).reshape(n_origins, features, height, width)


def _model_config(config: dict[str, Any], n_features: int, checkpoint_config: dict[str, Any] | None = None) -> DotConfig:
    autoformer = config[MODEL_NAME]
    model = dict(autoformer["model"])
    if checkpoint_config is not None:
        model = dict(checkpoint_config["model"])
        windowing = checkpoint_config["windowing"]
        lookback = int(windowing["seq_len"])
        label_len = int(windowing["label_len"])
        prediction_horizon = int(windowing["pred_len"])
    else:
        lookback = int(autoformer.get("seq_len", config["windowing"]["lookback"]))
        label_len = int(autoformer.get("label_len", lookback // 2))
        prediction_horizon = int(autoformer.get("pred_len", max(config["windowing"]["horizons"])))
    if prediction_horizon != max(int(h) for h in config["windowing"]["horizons"]):
        raise ValueError(
            "The checkpoint prediction horizon must equal max(windowing.horizons)"
        )
    return DotConfig(
        seq_len=lookback,
        label_len=label_len,
        pred_len=prediction_horizon,
        enc_in=n_features,
        dec_in=n_features,
        c_out=n_features,
        d_model=int(model["d_model"]),
        e_layers=int(model.get("encoder_layers", model.get("e_layers", 2))),
        d_layers=int(model.get("decoder_layers", model.get("d_layers", 1))),
        n_heads=int(model["n_heads"]),
        moving_avg=int(model["moving_avg"]),
        dropout=float(model.get("dropout", 0.1)),
        factor=int(model.get("factor", 3)),
        output_attention=bool(model.get("output_attention", False)),
        csam_kernel_size=int(model.get("csam_kernel_size", 7)),
        d_ff=int(model.get("d_ff", 4 * int(model["d_model"]))),
    )


def _load_model(
    config: dict[str, Any],
    n_features: int,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[AutoformerCSAForecaster, DotConfig]:
    if str(checkpoint.get("model_name", MODEL_NAME)).lower() != MODEL_NAME:
        raise ValueError("Checkpoint model_name is not autoformer_csa")
    model_config = _model_config(config, n_features, checkpoint.get("model_config"))
    model = AutoformerCSAForecaster(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, model_config


def _validate_checkpoint_metadata(checkpoint: dict[str, Any], data) -> None:
    checkpoint_frequencies = checkpoint.get("frequencies")
    if checkpoint_frequencies is not None and not np.allclose(
        np.asarray(checkpoint_frequencies, dtype=np.float64),
        np.asarray(data.frequencies, dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("Checkpoint frequencies do not match evaluation data")
    checkpoint_normalization = checkpoint.get("normalization")
    if checkpoint_normalization is None and data.normalization is None:
        return
    if checkpoint_normalization is None or data.normalization is None:
        raise ValueError("Checkpoint and evaluation normalization do not agree")
    for key in ("mean_dbm", "std_dbm"):
        if not np.allclose(
            np.asarray(checkpoint_normalization[key], dtype=np.float32),
            np.asarray(data.normalization[key], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(f"Checkpoint normalization field {key!r} does not match evaluation data")


def evaluate_chunk(
    config: dict[str, Any],
    chunk,
    bands,
    output_dir: Path,
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    autoformer = config[MODEL_NAME]
    val_fraction = float(autoformer.get("val_fraction", 0.1))
    data = load_chunk(config, chunk, val_fraction=val_fraction)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_checkpoint_metadata(checkpoint, data)
    device = device_for(config)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train_split = data.splits[data.train_split]
    model, model_config = _load_model(config, train_split.model_input.shape[-1], checkpoint, device)
    test_stride = int(autoformer.get("test_stride", 1))
    horizons = sorted({int(h) for h in config["windowing"]["horizons"]})
    prediction_horizon = model_config.pred_len
    batch_size = int(autoformer.get("batch_size", 32))

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for split_name in test_splits:
        if split_name not in data.splits:
            raise KeyError(f"Configured test split {split_name!r} is not available")
        split = data.splits[split_name]
        is_map = split.model_input.ndim == 4
        starts = make_window_starts(
            split.model_input.shape[0],
            model_config.seq_len,
            prediction_horizon,
            test_stride,
            split.segments,
        )
        windows, map_height, map_width = _make_windows(
            split.model_input,
            starts,
            model_config.seq_len,
        )
        if len(windows) == 0:
            print(f"  No valid windows for split {split_name}; skipping")
            continue

        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
                prediction = forecast(
                    model=model,
                    x=batch,
                    prediction_horizon=prediction_horizon,
                    rollout_horizon=prediction_horizon,
                )
                predictions.append(prediction.cpu().numpy().astype(np.float32))
        all_predictions = np.concatenate(predictions, axis=0)
        if is_map:
            assert map_height is not None and map_width is not None
            grid_count = map_height * map_width
            if len(all_predictions) != grid_count * len(starts):
                raise RuntimeError("Map forecast count does not match grid and origin dimensions")

        predictions_by_horizon: dict[int, np.ndarray] = {}
        targets_by_horizon: dict[int, np.ndarray] = {}
        target_rows_by_horizon: dict[int, np.ndarray] = {}
        for horizon in horizons:
            target_rows = starts + model_config.seq_len + horizon - 1
            target_raw = split.raw_dbm[target_rows]
            if is_map:
                assert map_height is not None and map_width is not None
                prediction_normalized = _map_model_layout(
                    all_predictions[:, horizon - 1],
                    map_height,
                    map_width,
                    len(starts),
                )
                target_model = np.transpose(target_raw, (0, 3, 1, 2)).astype(np.float32, copy=False)
                prediction_frequency_last = np.transpose(prediction_normalized, (0, 2, 3, 1))
                target_frequency_last = np.transpose(target_model, (0, 2, 3, 1))
                prediction_dbm_frequency_last, abs_map, squared_map = absolute_and_squared_errors_dbm(
                    prediction_frequency_last,
                    target_frequency_last,
                    data.normalization,
                )
                prediction_dbm = np.transpose(
                    prediction_dbm_frequency_last,
                    (0, 3, 1, 2),
                ).astype(np.float32, copy=False)
                abs_error = np.mean(np.transpose(abs_map, (0, 3, 1, 2)), axis=(2, 3))
                squared_error = np.mean(np.transpose(squared_map, (0, 3, 1, 2)), axis=(2, 3))
                target_export = target_model
            else:
                prediction_normalized = all_predictions[:, horizon - 1]
                prediction_dbm, abs_error, squared_error = absolute_and_squared_errors_dbm(
                    prediction_normalized,
                    target_raw,
                    data.normalization,
                )
                target_export = target_raw.astype(np.float32, copy=False)
            global_target_rows = target_rows + int(split.row_start)
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
                target_rows=global_target_rows,
                history_offset=int(split.row_start),
                freqs=data.frequencies,
                abs_err=abs_error,
                sq_err=squared_error,
                bands=bands,
                feature_labels=data.feature_labels,
            )
            predictions_by_horizon[horizon] = prediction_dbm
            targets_by_horizon[horizon] = target_export
            target_rows_by_horizon[horizon] = target_rows.astype(np.int64, copy=False)

        export_map_forecasts(
            output_dir,
            chunk_id=f"{chunk.chunk_id}_{split_name}",
            model_name=MODEL_NAME,
            predictions_by_horizon=predictions_by_horizon,
            targets_by_horizon=targets_by_horizon,
            target_rows_by_horizon=target_rows_by_horizon,
            metadata={
                "model": MODEL_NAME,
                "split_name": split_name,
                "train_split": data.train_split,
                "test_split": split_name,
                "chunk_id": chunk.chunk_id,
                "start_mhz": chunk.start_mhz,
                "end_mhz": chunk.end_mhz,
                "lookback": model_config.seq_len,
                "label_len": model_config.label_len,
                "prediction_horizon": prediction_horizon,
                "batch_size": batch_size,
                "test_stride": test_stride,
                "history_offset": int(split.row_start),
                "frequencies_mhz": np.asarray(data.frequencies, dtype=np.float32),
                "mean_dbm": None if data.normalization is None else data.normalization["mean_dbm"],
                "std_dbm": None if data.normalization is None else data.normalization["std_dbm"],
                "evaluation_mode": "direct_multi_step",
                "max_horizon": prediction_horizon,
                "stored_horizons": horizons,
                "forecast_layout": "N,F,H,W" if is_map else "N,F",
                "map_height": map_height,
                "map_width": map_width,
                "checkpoint_path": str(checkpoint_path),
                "data_files": {key: str(value) for key, value in data.files.items()},
            },
        )
    return aggregate_rows, frequency_rows, band_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Autoformer-CSA")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None, help="Existing training run name")
    parser.add_argument("--output-dir", type=Path, default=None, help="Existing training run directory")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path or {chunk_id} template")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is None and args.name is None:
        raise ValueError("Evaluation requires --name or --output-dir for an existing run")
    run_dir = args.output_dir or Path("runs") / args.name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    output_dir = run_dir
    bands = load_band_definitions(config)
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    started_at = timestamp_utc()
    started_counter = time.perf_counter()

    for chunk in chunk_specs(config):
        checkpoint_path = (
            Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
            if args.checkpoint
            else output_dir / "checkpoints" / f"{chunk.chunk_id}_{MODEL_NAME}.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        print(
            f"Evaluating Autoformer-CSA for {chunk.chunk_id} "
            f"({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)"
        )
        aggregate, frequency, band = evaluate_chunk(
            config,
            chunk,
            bands,
            output_dir,
            checkpoint_path,
        )
        aggregate_rows.extend(aggregate)
        frequency_rows.extend(frequency)
        band_rows.extend(band)

    duration = time.perf_counter() - started_counter
    finalize_results(
        output_dir,
        MODEL_NAME,
        aggregate_rows,
        frequency_rows,
        band_rows,
        [
            f"Evaluation start time: {started_at}",
            f"Evaluation end time: {timestamp_utc()}",
            f"Total evaluation time seconds: {duration:.2f}",
        ],
    )
    if not args.skip_plots:
        from training.common.plot_forecasts import generate_all_plots

        generate_all_plots(
            results_dir=output_dir,
            model_name=MODEL_NAME,
            bins=(30, 150),
            max_steps=500,
            horizons=config["windowing"]["horizons"],
        )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {output_dir / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
