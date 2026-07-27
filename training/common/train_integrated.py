""" Training Script"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


#--- Ensure Python Can properly find custom modules ---
# Locate Root folder 3 directories up from current location
ROOT = Path(__file__).resolve().parents[2] 

# Check if root folder begins file path so that modules can be found
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
# Locate the current folder
SCRIPT_DIR = Path(__file__).resolve().parent

# Check if current folder begins file path so that modules in the same fodler as the script can be found
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))



#--- Import custom utilities--- 
# Loads global yaml configuration file
from training.common.config import load_config, model_names, unique_run_dir

# Helper for runtime functions: GPU/CPU Compute Check, metric logging, and clean formatting of UTC dates.
from training.common.runtime import device_for, epoch_log_row, timestamp_utc

# Helper utilities for  directory creation
from training.common.results import prepare_output_dirs

# Helper utilities that identify tracking specs and load raw .npz or .csv multi-frequency grid arrays into memory.
from training.common.data import chunk_specs, load_chunk

#Helper for Model forecasting
from training.common.forecasting import forecast
from training.common.validation_diagnostics import (
    max_abs_prediction_by_horizon,
    prediction_diagnostic_log,
    prediction_guard_status,
    prediction_guard_threshold,
)

# Helper utilities for model construction
from training.common.model_factory import (
    SUPPORTED_MODELS,
    build_model,
)
from models.ConvLSTM_FM import pretrain_backbone
from training.common.specialized_models import (
    SPECIALIZED_MODELS,
    train_specialized_chunk,
)

# Helper for dataloading
from training.common.windowing import (
    build_window_loaders,
)
from training.common.data_loader import seed_everything

def train_model(
    model_name: str,
    model: nn.Module,
    train_data: np.ndarray,
    config: dict[str, Any],
    segments=(),
    val_data: np.ndarray | None = None,
    val_segments=(),
):
    """ Integrated Training, Validation, and Logging """
    print(f"[DEBUG] train_model entry: model_name={model_name}, train_data.shape={train_data.shape}")
    
    # Load Shared settings
    model_cfg = config[model_name]["model"]
    train_cfg = config[model_name]["train"]
    convlstm_guard_threshold = (
        prediction_guard_threshold(train_cfg)
        if model_name in ("convlstm", "residualconvlstm")
        else None
    )


    lookback = int(model_cfg["input_sequence_length"])

    # Keep model output at one timestep
    prediction_horizon = int(model_cfg["prediction_horizon"])

    # Validation can still roll out farther
    rollout_horizon = max(config["windowing"]["horizons"])

    if ( prediction_horizon != 1 and prediction_horizon != rollout_horizon ):
        raise ValueError(
            " Error! For direct multi-step prediction, prediction_horizon "
            f"must equal rollout_horizon. Got "
            f"prediction_horizon={prediction_horizon} and "
            f"rollout_horizon={rollout_horizon}."
        )
    
    # Load model training specs
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 20))
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    train_stride = int( train_cfg.get("train_stride", 1))
    val_stride = int( train_cfg.get("val_stride", 1))

    # Build data loaders for training and validation
    print(f"[DEBUG] building window loaders (train_data.shape={train_data.shape}, batch_size={batch_size}, lookback={lookback}) ...")
    train_loader, val_loader = build_window_loaders(
        data=train_data,
        lookback=lookback,
        rollout_horizon=rollout_horizon,
        batch_size=batch_size,
        val_fraction=val_fraction,
        train_stride=train_stride,
        val_stride=val_stride,
        segments=segments,
        data_loader_config=config.get("data_loader"),
        val_data=val_data,
        val_segments=val_segments,
    )
    print(f"[DEBUG] window loaders built: train={len(train_loader.dataset)} windows, val={len(val_loader.dataset) if val_loader else 0} windows")

    #--- Build training components ----
    
    # Initialise compute
    device = device_for(config)
    print(f"[DEBUG] device_for returned: {device} (cuda_available={torch.cuda.is_available()})")

    # Move the model to GPU or CPU.
    model = model.to(device)

    pretraining_metadata = None
    if model_name == "convlstmfm":
        pretrain_epochs = int(model_cfg.get("pretrain_epochs", 0))
        if pretrain_epochs > 0:
            pretraining_metadata = pretrain_backbone(
                model,
                train_loader,
                epochs=pretrain_epochs,
                mask_ratio=float(model_cfg.get("pretrain_mask_ratio", 0.2)),
                learning_rate=float(model_cfg.get("pretrain_learning_rate", 1e-3)),
                mask_mode=str(model_cfg.get("pretrain_mask_mode", "tokens")),
            )
            if bool(model_cfg.get("freeze_backbone_after_pretrain", False)):
                model.freeze_backbone()

    # Mean squared error between predictions and targets.
    criterion = nn.MSELoss()

    # Shared default optimizer settings.
    learning_rate = float(
        train_cfg.get("learning_rate", 0.001)
    )

    weight_decay = float(
        train_cfg.get("weight_decay", 0.0)
    )

    optimizer_name = str(
        train_cfg.get("optimizer", "adam")
    ).lower()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = None
    if not trainable_parameters:
        learning_rate = 0.0
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        momentum = float(
            train_cfg.get("momentum", 0.0)
        )
        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer "
            f"{optimizer_name!r}. "
            f"Supported: adam, adamw, sgd."
        )

    # Maximum allowed gradient size.
    clip_norm = float(
        train_cfg.get("gradient_clip_norm", 1.0)
    )
    
    # Number of epochs allowed without validation improvement.
    patience = int(
        train_cfg.get("early_stopping_patience", 10)
    )
    
    early_stopping = bool(
        train_cfg.get("early_stopping", True)
    )

    # Initialize training tracking containers

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    log_rows: list[dict[str, Any]] = []

    training_start_time = timestamp_utc()
    training_start_counter = time.perf_counter()
    
    print(f"[DEBUG] entering epoch loop: epochs={epochs}, device={device}")
    #--- Training---
    # Epoch loop
    for epoch in range(1, epochs + 1):
        if epoch == 1 or epoch % 5 == 0:
            print(f"[DEBUG] epoch {epoch}/{epochs} starting")
        epoch_start_time = timestamp_utc()
        epoch_start_counter = time.perf_counter()

        # Training phase
        model.train()

        _t_train = time.perf_counter()

        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if epoch == 1 and batch_idx == 0:
                print(f"[DEBUG] first batch: x.shape={x.shape}, y.shape={y.shape}, device={device}")
            # Move this batch to the same device as the model.
            x = x.to(device)
            y = y.to(device)

            # Clear gradients from the previous batch.
            if optimizer is not None:
                optimizer.zero_grad()

            # Most models use the normal forward call.
            if epoch == 1 and batch_idx == 0:
                print(f"[DEBUG] epoch1/batch0: calling forecast() ...")
                import time as _time_bt
                _bt0 = _time_bt.perf_counter()
            pred = forecast(
                model=model,
                x=x,
                prediction_horizon=prediction_horizon,
                rollout_horizon=rollout_horizon,
                targets=y,
            )
            if epoch == 1 and batch_idx == 0:
                print(f"[DEBUG] epoch1/batch0: forecast() done ({_time_bt.perf_counter() - _bt0:.1f}s), pred.shape={pred.shape}")
            
            # Confirm that model output matches the target.
            if pred.shape != y.shape:
                raise RuntimeError(
                    "Error! Training shape mismatch:\n"
                    f"Prediction shape: {tuple(pred.shape)}\n"
                    f"Target shape:     {tuple(y.shape)}"
                )

            # Measure prediction error.
            loss = criterion(pred, y)
            if epoch == 1 and batch_idx == 0:
                print(f"[DEBUG] epoch1/batch0: loss={loss.item():.6f}, calling backward() ...")
                _bt1 = _time_bt.perf_counter()

            # Calculate gradients.
            if optimizer is not None:
                loss.backward()
                if epoch == 1 and batch_idx == 0:
                    print(f"[DEBUG] epoch1/batch0: backward() done ({_time_bt.perf_counter() - _bt1:.1f}s)")

                # Limit very large gradients.
                if clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        clip_norm,
                    )

                # Update model parameters.
                optimizer.step()
                if epoch == 1 and batch_idx == 0:
                    print(f"[DEBUG] epoch1/batch0: optimizer.step() done")

            # Track total training loss.
            batch_samples = x.size(0)

            train_loss_sum += (
                loss.item() * batch_samples
            )

            train_sample_count += batch_samples

        train_loss = (
            train_loss_sum
            / max(train_sample_count, 1)
        )
        _t_train = time.perf_counter() - _t_train

        #--- Validation phase ---
        model.eval()

        val_loss_sum = 0.0
        val_sample_count = 0

        val_teacher_sum = 0.0
        val_teacher_count = 0

        # Per-horizon accumulators (autoregressive)
        horizons_cfg = sorted(config["windowing"]["horizons"])
        horizon_step_map = {h: h - 1 for h in horizons_cfg}
        val_horizon_sums = {h: 0.0 for h in horizons_cfg}
        val_horizon_counts = {h: 0 for h in horizons_cfg}
        max_abs_by_horizon = [0.0] * rollout_horizon

        _t_val_start = time.perf_counter()
        _t_tf_acc = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                # --- Autoregressive (current default) ---
                pred_ar = forecast(
                    model=model,
                    x=x,
                    prediction_horizon=prediction_horizon,
                    rollout_horizon=rollout_horizon,
                    targets=None,
                )

                if pred_ar.shape != y.shape:
                    raise RuntimeError(
                        "Validation shape mismatch:\n"
                        f"Prediction shape: {tuple(pred_ar.shape)}\n"
                        f"Target shape:     {tuple(y.shape)}"
                    )

                loss_ar = criterion(pred_ar, y)
                if convlstm_guard_threshold is not None:
                    batch_maxima = max_abs_prediction_by_horizon(pred_ar)
                    max_abs_by_horizon = [
                        max(current, batch)
                        if np.isfinite(current) and np.isfinite(batch)
                        else float("nan")
                        for current, batch in zip(max_abs_by_horizon, batch_maxima)
                    ]
                batch_samples = x.size(0)
                val_loss_sum += loss_ar.item() * batch_samples
                val_sample_count += batch_samples

                # Per-horizon breakdown
                step_mse = ((pred_ar - y) ** 2).mean(
                    dim=tuple(range(2, pred_ar.ndim))
                )
                for h in horizons_cfg:
                    idx = horizon_step_map[h]
                    val_horizon_sums[h] += step_mse[:, idx].sum().item()
                    val_horizon_counts[h] += batch_samples

                # --- Teacher-forced (diagnostic) ---
                if prediction_horizon == 1:
                    _t_tf = time.perf_counter()
                    pred_tf = forecast(
                        model=model,
                        x=x,
                        prediction_horizon=1,
                        rollout_horizon=rollout_horizon,
                        targets=y,
                    )
                    loss_tf = criterion(pred_tf, y)
                    val_teacher_sum += loss_tf.item() * batch_samples
                    val_teacher_count += batch_samples
                    _t_tf_acc += time.perf_counter() - _t_tf

        _t_val_total = time.perf_counter() - _t_val_start
        _t_val_ar = _t_val_total - _t_tf_acc
        _t_val_tf = _t_tf_acc

        val_loss = val_loss_sum / max(val_sample_count, 1)
        val_teacher_loss = (
            val_teacher_sum / max(val_teacher_count, 1)
            if val_teacher_count > 0
            else None
        )

        val_horizon_losses = {
            h: val_horizon_sums[h] / max(val_horizon_counts[h], 1)
            for h in horizons_cfg
        }

        horizon_str = "  ".join(
            f"t+{h}={val_horizon_losses[h]:.6f}"
            for h in horizons_cfg
        )
        teacher_str = (
            f"  teacher_val={val_teacher_loss:.6f}"
            if val_teacher_loss is not None
            else ""
        )
        print(
            f"  val_ar={val_loss:.6f}{teacher_str}  "
            f"horizons: {horizon_str}"
        )

        #--- Log epoch results ---
        epoch_duration = (
            time.perf_counter()
            - epoch_start_counter
        )

        log_row = epoch_log_row(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            epoch_start_time=epoch_start_time,
            epoch_end_time=timestamp_utc(),
            epoch_duration_sec=epoch_duration,
            learning_rate=(
                float(optimizer.param_groups[0]["lr"])
                if optimizer is not None
                else 0.0
            ),
        )
        if val_teacher_loss is not None:
            log_row["val_teacher_loss"] = val_teacher_loss
        for h in horizons_cfg:
            log_row[f"val_loss_t{h}"] = val_horizon_losses[h]
        selection_eligible = True
        guard_status = "not_applicable"
        if convlstm_guard_threshold is not None:
            selection_eligible, guard_status = prediction_guard_status(
                max_abs_by_horizon,
                convlstm_guard_threshold,
            )
            log_row.update(prediction_diagnostic_log(max_abs_by_horizon))
            log_row["validation_selection_eligible"] = selection_eligible
            log_row["validation_guard_status"] = guard_status
        log_rows.append(log_row)

        diagnostic_str = (
            f"max_abs_pred={max(max_abs_by_horizon):.6g} "
            f"guard={guard_status} "
            if convlstm_guard_threshold is not None
            else ""
        )
        print(
            f"{model_name} "
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"{diagnostic_str}"
            f"time={epoch_duration:.1f}s"
        )

        _t_overhead = epoch_duration - _t_train - _t_val_total
        print(
            f"=== EPOCH PROFILE ===\n"
            f"  Training fwd+bwd+opt:    {_t_train:>8.1f} s  ({_t_train/epoch_duration*100:>5.1f}%)\n"
            f"  Validation AR:           {_t_val_ar:>8.1f} s  ({_t_val_ar/epoch_duration*100:>5.1f}%)\n"
            f"  Validation TF:           {_t_val_tf:>8.1f} s  ({_t_val_tf/epoch_duration*100:>5.1f}%)\n"
            f"  Epoch overhead:          {_t_overhead:>8.1f} s  ({_t_overhead/epoch_duration*100:>5.1f}%)\n"
            f"  Total:                   {epoch_duration:>8.1f} s\n"
            f"====================="
        )

        # Best model and early stopping
        if selection_eligible and np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            best_state = {
                name: value.detach().cpu().clone()
                for name, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        elif selection_eligible:
            epochs_without_improvement += 1

            if early_stopping and epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch was {best_epoch}."
                )

                break
    
    # Restore and return the best model
    total_training_time = (
        time.perf_counter()
        - training_start_counter
    )
    
    if best_state is None:
        if convlstm_guard_threshold is not None:
            raise RuntimeError(
                "Error! ConvLSTM training finished without an eligible validation "
                "epoch; all predictions were non-finite or exceeded "
                f"validation_prediction_magnitude_threshold="
                f"{convlstm_guard_threshold:g}"
            )
        raise RuntimeError(
            "Error! Training finished without saving a model state"
        )
    
    model.load_state_dict(best_state)
    
    if not log_rows:
        raise RuntimeError("Error! No epoch results were recorded")

    # Convert the epoch log into a DataFrame for summary calculations.
    log_frame = pd.DataFrame(log_rows)

    # Find the row belonging to the best validation epoch.
    best_epoch_row = log_frame.loc[
        log_frame["epoch"] == best_epoch
    ].iloc[0]

    training_end_time = timestamp_utc()

    training_results = {
        "model_name": model_name,
        "pretraining": pretraining_metadata,
        "epochs_completed": len(log_rows),

        # Best validation result and corresponding training loss.
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "train_loss_at_best_epoch": float(
            best_epoch_row["train_loss"]
        ),

        # Best training loss observed during any epoch.
        "best_train_loss": float(
            log_frame["train_loss"].min()
        ),

        # Losses from the final completed epoch.
        "final_train_loss": float(
            log_frame.iloc[-1]["train_loss"]
        ),
        "final_val_loss": float(
            log_frame.iloc[-1]["val_loss"]
        ),

        # Average losses across all completed epochs.
        "average_train_loss": float(
            log_frame["train_loss"].mean()
        ),
        "average_val_loss": float(
            log_frame["val_loss"].mean()
        ),

        # Difference between validation and training loss
        # at the best validation epoch.
        "best_epoch_loss_gap": float(
            best_epoch_row["val_loss"]
            - best_epoch_row["train_loss"]
        ),

        # Timing information.
        "average_epoch_time_sec": float(
            log_frame["epoch_duration_sec"].mean()
        ),
        "fastest_epoch_time_sec": float(
            log_frame["epoch_duration_sec"].min()
        ),
        "slowest_epoch_time_sec": float(
            log_frame["epoch_duration_sec"].max()
        ),
        "training_start_time": training_start_time,
        "training_end_time": training_end_time,
        "training_duration_sec": total_training_time,

        # Complete per-epoch records.
        "log_frame": log_frame,
    }
    
    return model, training_results


def parse_args() -> argparse.Namespace:
    """ Set up Parsers for command line inputs """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)    # config path to yaml file
    parser.add_argument("--name", type=str, default=None)       # experiment name (default: <model>_<timestamp>)
    parser.add_argument("--output-dir", type=Path, default=None) # override run directory
    return parser.parse_args()

def train_one_model(config: dict[str, Any], model_name: str, run_dir: Path) -> None:
    print(f"[DEBUG] train_one_model entry: model={model_name}, run_dir={run_dir}")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Error! Integrated training supports "
            f"{sorted(SUPPORTED_MODELS)}, got {model_name!r}."
        )
    if model_name in SPECIALIZED_MODELS:
        out, checkpoints = prepare_output_dirs(run_dir)
        for chunk in chunk_specs(config):
            print(
                f"Training {model_name} for {chunk.chunk_id} "
                f"({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)"
            )
            data = load_chunk(config, chunk)
            train_specialized_chunk(
                model_name, config, chunk, data, out, checkpoints
            )
        return

    train_cfg = config[model_name]["train"]
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    out, checkpoints = prepare_output_dirs(run_dir)
    print(f"[DEBUG] output dirs ready: out={out}, checkpoints={checkpoints}")
        
    # Iterate through each chunk segments
    for chunk in chunk_specs(config):
        print(
            f"Training {model_name} for {chunk.chunk_id} "
            f"({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)"
        )
        
        print(f"[DEBUG] calling load_chunk for {chunk.chunk_id} ...")
        data = load_chunk(config, chunk, val_fraction = val_fraction)
        print(f"[DEBUG] load_chunk done: train_split={data.train_split}, test_split={data.test_split}")
        train = data.splits[data.train_split].model_input
        validation = data.splits.get(data.validation_split)
        print(f"[DEBUG] train data shape: {train.shape}, dtype: {train.dtype}")
        
        # Build model based on model specific needs
        seed_everything(int(train_cfg.get("seed", 42)))
        print(f"[DEBUG] calling build_model for {model_name} ...")
        model = build_model(
            model_name=model_name,
            config=config,
            train_data=train,
        )
        print(f"[DEBUG] build_model done: {type(model).__name__}")
        # Parameter-free baselines (for example LookbackMean) still use the
        # shared runtime device during forecasting and need no model parameter
        # lookup here.
        device = device_for(config)
        print(f"[DEBUG] model parameters on device: {device}")
        
        # Train model
        print(f"[DEBUG] calling train_model ...")
        model, training_results = train_model(
            model_name=model_name,
            model=model,
            train_data=train,
            config=config,
            segments=data.splits[data.train_split].segments,
            val_data=None if validation is None else validation.model_input,
            val_segments=() if validation is None else validation.segments,
        )
        
        # Directly stream the pre-compiled log_frame to file  to avoid wasting CPU cycles reconstructing the table
        training_results["log_frame"].to_csv(
            out / f"{chunk.chunk_id}_training_log.csv",
            index=False,
        )
        
        summary_lines = [
            f"Model: {model_name}",
            f"Chunk: {chunk.chunk_id}",
            f"Epochs completed: {training_results['epochs_completed']}",
            "",
            f"Best epoch: {training_results['best_epoch']}",
            f"Best validation loss: {training_results['best_val_loss']:.6f}",
            (
                "Training loss at best epoch: "
                f"{training_results['train_loss_at_best_epoch']:.6f}"
            ),
            (
                "Loss gap at best epoch: "
                f"{training_results['best_epoch_loss_gap']:.6f}"
            ),
            "",
            f"Best training loss: {training_results['best_train_loss']:.6f}",
            f"Final training loss: {training_results['final_train_loss']:.6f}",
            f"Final validation loss: {training_results['final_val_loss']:.6f}",
            (
                "Average training loss: "
                f"{training_results['average_train_loss']:.6f}"
            ),
            (
                "Average validation loss: "
                f"{training_results['average_val_loss']:.6f}"
            ),
            "",
            (
                "Average epoch time: "
                f"{training_results['average_epoch_time_sec']:.2f} seconds"
            ),
            (
                "Fastest epoch time: "
                f"{training_results['fastest_epoch_time_sec']:.2f} seconds"
            ),
            (
                "Slowest epoch time: "
                f"{training_results['slowest_epoch_time_sec']:.2f} seconds"
            ),
            (
                "Total training time: "
                f"{training_results['training_duration_sec']:.2f} seconds"
            ),
            "",
            f"Training started: {training_results['training_start_time']}",
            f"Training ended: {training_results['training_end_time']}",
        ]

        (
            out / f"{chunk.chunk_id}_training_summary.txt"
        ).write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        # Save training log and checkpoints
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "normalization": data.normalization,
                "frequencies": data.frequencies,
                "training_results": {
                    key: value
                    for key, value in training_results.items()
                    if key != "log_frame"
                },
                "pretraining": training_results.get("pretraining"),
            },
            checkpoints/ f"{chunk.chunk_id}_{model_name}.pt",
        )


def main() -> None:
    print(f"[DEBUG] main() started")
    args = parse_args()
    print(f"[DEBUG] config={args.config}, name={args.name}, output_dir={args.output_dir}")
    config = load_config(args.config)
    models = model_names(config)
    print(f"[DEBUG] loaded config, models={models}")
    unsupported = [model for model in models if model not in SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(
            f"Integrated training does not support: {', '.join(unsupported)}"
        )

    default_name = (
        models[0] if len(models) == 1 else "comparison"
    ) + f"_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    requested_dir = args.output_dir or Path("runs") / (args.name or default_name)
    run_dir = unique_run_dir(requested_dir)
    print(f"[DEBUG] run_dir={run_dir}")

    config_source = args.config or Path(__file__).with_name("config.yaml")
    shutil.copy2(config_source, run_dir / "config.yaml")
    for model_name in models:
        model_run_dir = run_dir if len(models) == 1 else run_dir / model_name
        print(f"[DEBUG] calling train_one_model for {model_name} ...")
        train_one_model(config, model_name, model_run_dir)

    print(f"Training run written to {run_dir}")


if __name__ == "__main__":
    main()
