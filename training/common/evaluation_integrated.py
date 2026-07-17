"""
Model-agnostic evaluation for the integrated spectrum-prediction pipeline.

This module evaluates trained spectrum-forecasting models across configured
frequency chunks and test splits using the same data-loading, preprocessing,
windowing, model-construction, and forecasting policies used during training.

Supported model families include sequence models operating on frequency vectors
and spatiotemporal models operating on spectrum maps. Loader outputs are
converted into the model-specific layout before inference, while raw targets
remain available for denormalization, metric calculation, and forecast export.

Forecasting behavior is determined by the model's configured prediction horizon:

1. One-step models are rolled forward autoregressively for the complete
   evaluation horizon.
2. Direct multi-step models are called once and must return the complete
   configured rollout horizon.

For every requested horizon, this module:

- creates chronological lookback windows from the final test split;
- performs batched inference through the shared forecasting function;
- selects predictions and aligned raw targets;
- converts normalized outputs back to dBm when normalization is enabled;
- calculates aggregate, per-frequency, and frequency-band errors;
- exports predictions, targets, target-row indices, and metadata;
- combines chunk-level results into final evaluation result files; and
- optionally generates forecast and error plots.

Expected loader layouts:

    CSV:
        model_input: (T, F)
        raw_dbm:     (T, F)

    Map:
        model_input: (T, H, W, F)
        raw_dbm:     (T, H, W, F)

Expected model layouts:

    VanillaLSTM:
        input:  (B, T, F)
        output: (B, prediction_horizon, F)

    ConvLSTM:
        input:  (B, T, F, H, W)
        output: (B, prediction_horizon, F, H, W)

Export layouts:

    VanillaLSTM:
        (N, F)

    ConvLSTM:
        (N, F, H, W)
"""

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

# Ensure project package imports work when this file is run directly.

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from training.common.config import load_config
from training.common.data import ChunkSpec, chunk_specs, load_chunk
from training.common.forecasting import forecast
from training.common.forecast_export import export_map_forecasts
from training.common.runtime import device_for, timestamp_utc
from training.common.metrics import absolute_and_squared_errors_dbm
from training.common.results import (
    finalize_results,
    prepare_output_dirs,
    append_metric_rows,
    load_band_definitions,
)
from training.common.model_factory import (
    SUPPORTED_MODELS,
    build_model,
    load_checkpoint_into_model,
    checkpoint_path_for_chunk,
)

from training.common.windowing import (
    make_window_batch_array,
    make_window_starts,
    raw_targets_to_model_layout,
    to_model_layout,
)



# ===========================================================================
# Shared configuration helpers
# ===========================================================================

def model_sections(
    config: dict[str, Any],
    model_name: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """
    Return model, training, and optional evaluation settings for a model.
    """

    if model_name not in config:
        raise KeyError(
            f"Configuration has no section for model "
            f"{model_name!r}."
        )

    model_section = config[model_name]

    if "model" not in model_section:
        raise KeyError(
            f"Configuration section {model_name!r} "
            "is missing its 'model' subsection."
        )

    if "train" not in model_section:
        raise KeyError(
            f"Configuration section {model_name!r} "
            "is missing its 'train' subsection."
        )

    model_cfg = model_section["model"]
    train_cfg = model_section["train"]
    evaluation_cfg = model_section.get(
        "evaluation",
        {},
    )

    return model_cfg, train_cfg, evaluation_cfg


def evaluation_parameters(
    config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """
    Resolve evaluation settings while preserving training defaults.
    """

    model_cfg, train_cfg, evaluation_cfg = model_sections(
        config,
        model_name,
    )

    horizons = [
        int(value)
        for value in config["windowing"]["horizons"]
    ]

    if not horizons:
        raise ValueError(
            "windowing.horizons must contain at least one horizon."
        )

    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(
            "All configured evaluation horizons must be positive."
        )

    horizons = sorted(set(horizons))
    max_horizon = max(horizons)

    lookback = int(
        model_cfg["input_sequence_length"]
    )

    prediction_horizon = int(
        model_cfg["prediction_horizon"]
    )

    if lookback <= 0:
        raise ValueError(
            f"{model_name}.model.input_sequence_length "
            "must be positive."
        )

    if prediction_horizon <= 0:
        raise ValueError(
            f"{model_name}.model.prediction_horizon "
            "must be positive."
        )

    # This is the exact behavior enforced by the confirmed training script:
    # the model must either be one-step or directly output the full rollout.
    if prediction_horizon not in {
        1,
        max_horizon,
    }:
        raise ValueError(
            f"{model_name}.model.prediction_horizon must be "
            f"either 1 or max(windowing.horizons)={max_horizon}. "
            f"Got {prediction_horizon}."
        )

    batch_size = int(
        evaluation_cfg.get(
            "batch_size",
            train_cfg.get("batch_size", 32),
        )
    )

    test_stride = int(
        evaluation_cfg.get(
            "test_stride",
            train_cfg.get("test_stride", 1),
        )
    )

    val_fraction = float(
        train_cfg.get("val_fraction", 0.1)
    )

    if batch_size <= 0:
        raise ValueError(
            "Evaluation batch size must be positive."
        )

    if test_stride <= 0:
        raise ValueError(
            "Evaluation test stride must be positive."
        )

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"{model_name}.train.val_fraction must be "
            f"between 0 and 1, got {val_fraction}."
        )


    return {
        "lookback": lookback,
        "prediction_horizon": prediction_horizon,
        "horizons": horizons,
        "max_horizon": max_horizon,
        "batch_size": batch_size,
        "test_stride": test_stride,
        "val_fraction": val_fraction,
    }



# ===========================================================================
# Metric adaptation
# ===========================================================================


def calculate_errors_and_export_arrays(
    *,
    model_name: str,
    prediction_normalized: np.ndarray,
    target_raw_model_layout: np.ndarray,
    normalization: dict[str, Any] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Denormalize forecasts and prepare metric arrays.

    Returns:
        prediction_dbm_export
        target_dbm_export
        absolute_error_for_append_metric_rows
        squared_error_for_append_metric_rows

    `training.common.metrics.denormalize()` expects frequency to be the last
    axis because its mean/std vectors are shaped (F,). ConvLSTM forecasts use
    frequency as axis 1, so map forecasts are temporarily converted to
    frequency-last layout for denormalization and error calculation.
    """

    if model_name == "vanillalstm":
        (
            prediction_dbm,
            absolute_error,
            squared_error,
        ) = absolute_and_squared_errors_dbm(
            prediction_normalized,
            target_raw_model_layout,
            normalization,
        )

        return (
            prediction_dbm.astype(
                np.float32,
                copy=False,
            ),
            target_raw_model_layout.astype(
                np.float32,
                copy=False,
            ),
            absolute_error.astype(
                np.float32,
                copy=False,
            ),
            squared_error.astype(
                np.float32,
                copy=False,
            ),
        )

    if model_name == "convlstm":
        if prediction_normalized.ndim != 4:
            raise ValueError(
                "ConvLSTM predictions must be shaped "
                "(samples, frequency, height, width), "
                f"got {prediction_normalized.shape}."
            )

        if target_raw_model_layout.ndim != 4:
            raise ValueError(
                "ConvLSTM targets must be shaped "
                "(samples, frequency, height, width), "
                f"got {target_raw_model_layout.shape}."
            )

        # Convert model/export layout:
        #     (N, F, H, W)
        #
        # Into normalization layout:
        #     (N, H, W, F)
        prediction_frequency_last = np.transpose(
            prediction_normalized,
            (0, 2, 3, 1),
        )

        target_frequency_last = np.transpose(
            target_raw_model_layout,
            (0, 2, 3, 1),
        )

        (
            prediction_dbm_frequency_last,
            absolute_error_frequency_last,
            squared_error_frequency_last,
        ) = absolute_and_squared_errors_dbm(
            prediction_frequency_last,
            target_frequency_last,
            normalization,
        )

        # Convert back to common map export layout:
        #     (N, F, H, W)
        prediction_dbm = np.transpose(
            prediction_dbm_frequency_last,
            (0, 3, 1, 2),
        ).astype(
            np.float32,
            copy=False,
        )

        absolute_error_map = np.transpose(
            absolute_error_frequency_last,
            (0, 3, 1, 2),
        )

        squared_error_map = np.transpose(
            squared_error_frequency_last,
            (0, 3, 1, 2),
        )

        # append_metric_rows expects:
        #     (N, F)
        #
        # Average each frequency's error across the spatial grid.
        absolute_error_for_metrics = np.mean(
            absolute_error_map,
            axis=(2, 3),
        ).astype(
            np.float32,
            copy=False,
        )

        squared_error_for_metrics = np.mean(
            squared_error_map,
            axis=(2, 3),
        ).astype(
            np.float32,
            copy=False,
        )

        return (
            prediction_dbm,
            target_raw_model_layout.astype(
                np.float32,
                copy=False,
            ),
            absolute_error_for_metrics,
            squared_error_for_metrics,
        )

    raise ValueError(
        f"Unsupported metric adapter: {model_name!r}."
    )


# ===========================================================================
# Chunk evaluation
# ===========================================================================


def evaluate_chunk(
    *,
    config: dict[str, Any],
    model_name: str,
    chunk: ChunkSpec,
    bands: pd.DataFrame,
    output_directory: Path,
    checkpoint_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Evaluate one model checkpoint for one configured frequency chunk.
    """

    parameters = evaluation_parameters(
        config,
        model_name,
    )

    lookback = parameters["lookback"]
    prediction_horizon = parameters[
        "prediction_horizon"
    ]
    horizons = parameters["horizons"]
    max_horizon = parameters["max_horizon"]
    batch_size = parameters["batch_size"]
    test_stride = parameters["test_stride"]
    val_fraction = parameters["val_fraction"]

    # Use the exact same loader and preprocessing path as training.
    data = load_chunk(
        config,
        chunk,
        val_fraction=val_fraction,
    )

    train_split = data.splits[
        data.train_split
    ]

    train_data = train_split.model_input

    model = build_model(
        model_name=model_name,
        config=config,
        train_data=train_data,
    )

    device = device_for(config)

    model, checkpoint = load_checkpoint_into_model(
        checkpoint_path=checkpoint_path,
        model=model,
        model_name=model_name,
        data_normalization=data.normalization,
        data_frequencies=data.frequencies,
        device=device,
    )

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    test_splits = config["data"].get(
        "test_splits",
        [data.test_split],
    )

    for split_name in test_splits:
        if split_name not in data.splits:
            raise KeyError(
                f"Configured test split {split_name!r} "
                "was not returned by the data loader."
            )

        split = data.splits[split_name]

        full_x = to_model_layout(
            split.model_input
        )

        minimum_rows = lookback + max_horizon

        if len(full_x) < minimum_rows:
            print(
                f"  Not enough rows for split {split_name}; "
                f"need at least {minimum_rows}, got {len(full_x)}. "
                "Skipping."
            )
            continue
        
        start_rows = make_window_starts(
            n_timesteps=len(full_x),
            lookback=lookback,
            rollout_horizon=max_horizon,
            stride=test_stride,
        )

        windows = make_window_batch_array(
            full_x=full_x,
            start_rows=start_rows,
            lookback=lookback,
        )
        
        prediction_parts: dict[int, list[np.ndarray]] = {
            horizon: []
            for horizon in horizons
        }
        
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch_start in range(
                0,
                len(windows),
                batch_size,
            ):
                window = torch.from_numpy(
                    windows[
                        batch_start :
                        batch_start + batch_size
                    ]
                ).float().to(device)
        
                batch_forecast = forecast(
                    model=model,
                    x=window,
                    prediction_horizon=prediction_horizon,
                    rollout_horizon=max_horizon,
                    targets=None,
                )
        
                if batch_forecast.shape[1] != max_horizon:
                    raise RuntimeError(
                        "Evaluation forecast returned the wrong horizon. "
                        f"Expected {max_horizon}, "
                        f"got {batch_forecast.shape[1]}."
                    )
        
                for horizon in horizons:
                    prediction_parts[horizon].append(
                        batch_forecast[
                            :,
                            horizon - 1,
                            ...
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
        
        predictions = {
            horizon: np.concatenate(
                parts,
                axis=0,
            ).astype(np.float32)
            for horizon, parts in prediction_parts.items()
        }

        predictions_by_horizon: dict[int, np.ndarray] = {}
        targets_by_horizon: dict[int, np.ndarray] = {}
        target_rows_by_horizon: dict[int, np.ndarray] = {}

        for horizon in horizons:
            local_target_rows_all = (
                start_rows
                + lookback
                + horizon
                - 1
            )

            local_target_rows = (
                local_target_rows_all
            )
            
            prediction_normalized = (
                predictions[horizon]
            )

            target_raw = raw_targets_to_model_layout(
                split.raw_dbm,
                local_target_rows,
            )

            (
                prediction_dbm,
                target_dbm_export,
                absolute_error,
                squared_error,
            ) = calculate_errors_and_export_arrays(
                model_name=model_name,
                prediction_normalized=prediction_normalized,
                target_raw_model_layout=target_raw,
                normalization=data.normalization,
            )

            if (
                prediction_dbm.shape
                != target_dbm_export.shape
            ):
                raise RuntimeError(
                    "Denormalized prediction and target "
                    "shapes do not match:\n"
                    f"prediction={prediction_dbm.shape}\n"
                    f"target={target_dbm_export.shape}"
                )

            # append_metric_rows uses global target rows, then subtracts
            # history_offset to store split-local target start/end values.
            global_target_rows = (
                local_target_rows
                + int(split.row_start)
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
                model=model_name,
                target_rows=global_target_rows,
                history_offset=int(split.row_start),
                freqs=data.frequencies,
                abs_err=absolute_error,
                sq_err=squared_error,
                bands=bands,
            )

            predictions_by_horizon[
                horizon
            ] = prediction_dbm.astype(
                np.float32,
                copy=False,
            )

            targets_by_horizon[
                horizon
            ] = target_dbm_export.astype(
                np.float32,
                copy=False,
            )

            # Forecast files use rows local to the selected test split.
            target_rows_by_horizon[
                horizon
            ] = local_target_rows.astype(
                np.int64,
                copy=False,
            )

        if not predictions_by_horizon:
            print(
                f"  No forecasts were generated for "
                f"split {split_name}; skipping export."
            )
            continue

        export_map_forecasts(
            output_directory,
            chunk_id=(
                f"{chunk.chunk_id}_{split_name}"
            ),
            model_name=model_name,
            predictions_by_horizon=(
                predictions_by_horizon
            ),
            targets_by_horizon=(
                targets_by_horizon
            ),
            target_rows_by_horizon=(
                target_rows_by_horizon
            ),
            metadata={
                "model": model_name,
                "split_name": split_name,
                "train_split": data.train_split,
                "test_split": split_name,
                "chunk_id": chunk.chunk_id,
                "start_mhz": chunk.start_mhz,
                "end_mhz": chunk.end_mhz,
                "lookback": lookback,
                "prediction_horizon": prediction_horizon,
                "batch_size": batch_size,
                "test_stride": test_stride,
                "history_offset": int(
                    split.row_start
                ),
                "frequencies_mhz": np.asarray(
                    data.frequencies,
                    dtype=np.float32,
                ),
                "normalization": (
                    None
                    if data.normalization is None
                    else data.normalization.get(
                        "source_split"
                    )
                ),
                "mean_dbm": (
                    None
                    if data.normalization is None
                    else data.normalization.get(
                        "mean_dbm"
                    )
                ),
                "std_dbm": (
                    None
                    if data.normalization is None
                    else data.normalization.get(
                        "std_dbm"
                    )
                ),
                "evaluation_mode": (
                    "external_autoregressive_rollout"
                    if prediction_horizon == 1
                    else "direct_multi_step"
                ),
                "max_horizon": max_horizon,
                "stored_horizons": horizons,
                "forecast_layout": (
                    "N,F"
                    if model_name == "vanillalstm"
                    else "N,F,H,W"
                ),
                "checkpoint_path": str(
                    checkpoint_path
                ),
                "checkpoint_training_results": (
                    checkpoint.get(
                        "training_results"
                    )
                ),
                "data_files": {
                    key: str(value)
                    for key, value
                    in data.files.items()
                },
            },
        )

    return (
        aggregate_rows,
        frequency_rows,
        band_rows,
    )


# ===========================================================================
# Command-line and top-level execution
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model trained by the integrated "
            "spectrum-prediction training pipeline."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the shared YAML configuration file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint path override. Use "
            "'{chunk_id}' in the path for per-chunk substitution."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional evaluation output directory override. "
            "This does not change the default checkpoint location."
        ),
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Calculate metrics and exports without generating plots.",
    )

    return parser.parse_args()



def main() -> None:
    args = parse_args()

    config = load_config(
        args.config
    )

    model_name = str(
        config["training"]["model_name"]
    ).lower()

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Error! Integrated evaluation currently supports "
            f"{sorted(SUPPORTED_MODELS)}, got {model_name!r}."
        )

    # This is the standard model output location used by training.
    standard_output_directory, standard_checkpoint_directory = (
        prepare_output_dirs(
            config,
            model_name,
        )
    )

    output_directory = (
        standard_output_directory
        if args.output_dir is None
        else args.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    bands = load_band_definitions(
        config
    )

    evaluation_start_time = timestamp_utc()
    evaluation_start_counter = time.perf_counter()

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    evaluated_chunks = 0
    skipped_chunks = 0

    for chunk in chunk_specs(config):
        print(
            f"Evaluating {model_name} for "
            f"{chunk.chunk_id} "
            f"({chunk.start_mhz:g}-"
            f"{chunk.end_mhz:g} MHz)"
        )

        checkpoint_path = checkpoint_path_for_chunk(
            checkpoint_override=args.checkpoint,
            default_checkpoint_directory=(
                standard_checkpoint_directory
            ),
            chunk_id=chunk.chunk_id,
            model_name=model_name,
        )

        if not checkpoint_path.exists():
            print(
                f"  Checkpoint not found: "
                f"{checkpoint_path}; skipping."
            )

            skipped_chunks += 1
            continue

        (
            chunk_aggregate_rows,
            chunk_frequency_rows,
            chunk_band_rows,
        ) = evaluate_chunk(
            config=config,
            model_name=model_name,
            chunk=chunk,
            bands=bands,
            output_directory=output_directory,
            checkpoint_path=checkpoint_path,
        )

        aggregate_rows.extend(
            chunk_aggregate_rows
        )

        frequency_rows.extend(
            chunk_frequency_rows
        )

        band_rows.extend(
            chunk_band_rows
        )

        evaluated_chunks += 1

    evaluation_duration = (
        time.perf_counter()
        - evaluation_start_counter
    )

    finalize_results(
        output_directory,
        model_name,
        aggregate_rows,
        frequency_rows,
        band_rows,
        [
            (
                "Evaluation start time: "
                f"{evaluation_start_time}"
            ),
            (
                "Evaluation end time: "
                f"{timestamp_utc()}"
            ),
            (
                "Total evaluation time seconds: "
                f"{evaluation_duration:.2f}"
            ),
            (
                "Chunks evaluated: "
                f"{evaluated_chunks}"
            ),
            (
                "Chunks skipped: "
                f"{skipped_chunks}"
            ),
        ],
    )

    print(
        f"Wrote {len(aggregate_rows)} aggregate "
        f"metric rows to "
        f"{output_directory / 'aggregate_metrics.csv'}"
    )

    if not args.skip_plots:
        from training.common.plot_forecasts import (
            generate_all_plots,
        )

        generate_all_plots(
            results_dir=output_directory,
            model_name=model_name,
            bins=(30, 150),
            max_steps=500,
        )


if __name__ == "__main__":
    main()
