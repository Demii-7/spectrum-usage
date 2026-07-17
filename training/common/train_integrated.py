""" Training Script"""

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
from training.common.config import load_config

# Helper for runtime functions: GPU/CPU Compute Check, metric logging, and clean formatting of UTC dates.
from training.common.runtime import device_for, epoch_log_row, timestamp_utc

# Helper utilities for  directory creation
from training.common.results import prepare_output_dirs

# Helper utilities that identify tracking specs and load raw .npz or .csv multi-frequency grid arrays into memory.
from training.common.data import chunk_specs, load_chunk

#Helper for Model forecasting
from training.common.forecasting import forecast

# Helper utilities for model construction
from training.common.model_factory import (
    SUPPORTED_MODELS,
    build_model,
)

# Helper for dataloading
from training.common.windowing import (
    build_window_loaders,
)

def train_model( model_name: str, model: nn.Module, train_data: np.ndarray, config: dict[str, Any],):
    """ Integrated Training, Validation, and Logging """
    
    # Shared settings
    model_cfg = config[model_name]["model"]
    train_cfg = config[model_name]["train"]


    seed = int(train_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    train_loader, val_loader = build_window_loaders(
        data=train_data,
        lookback=lookback,
        rollout_horizon=rollout_horizon,
        batch_size=batch_size,
        val_fraction=val_fraction,
        train_stride=train_stride,
        val_stride=val_stride,
    )

    #--- Build training components ----
    
    # Initialise compute
    device = device_for(config)

    # Move the model to GPU or CPU.
    model = model.to(device)

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

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        momentum = float(
            train_cfg.get("momentum", 0.0)
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
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
    
    #--- Training---
    # Epoch loop
    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        epoch_start_counter = time.perf_counter()

        # Training phase
        model.train()

        train_loss_sum = 0.0
        train_sample_count = 0

        for x, y in train_loader:
            # Move this batch to the same device as the model.
            x = x.to(device)
            y = y.to(device)

            # Clear gradients from the previous batch.
            optimizer.zero_grad()

            # Most models use the normal forward call.
            pred = forecast(
                model=model,
                x=x,
                prediction_horizon=prediction_horizon,
                rollout_horizon=rollout_horizon,
                targets=y,
            )
            
            # Confirm that model output matches the target.
            if pred.shape != y.shape:
                raise RuntimeError(
                    "Error! Training shape mismatch:\n"
                    f"Prediction shape: {tuple(pred.shape)}\n"
                    f"Target shape:     {tuple(y.shape)}"
                )

            # Measure prediction error.
            loss = criterion(pred, y)

            # Calculate gradients.
            loss.backward()

            # Limit very large gradients.
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_norm,
                )

            # Update model parameters.
            optimizer.step()

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

        #--- Validation phase ---
        model.eval()

        val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                pred = forecast(
                    model=model,
                    x=x,
                    prediction_horizon=prediction_horizon,
                    rollout_horizon=rollout_horizon,
                    targets= None,
                )

                if pred.shape != y.shape:
                    raise RuntimeError(
                        "Validation shape mismatch:\n"
                        f"Prediction shape: {tuple(pred.shape)}\n"
                        f"Target shape:     {tuple(y.shape)}"
                    )

                loss = criterion(pred, y)

                batch_samples = x.size(0)

                val_loss_sum += (
                    loss.item() * batch_samples
                )

                val_sample_count += batch_samples

        val_loss = (
            val_loss_sum
            / max(val_sample_count, 1)
        )

        #--- Log epoch results ---
        epoch_duration = (
            time.perf_counter()
            - epoch_start_counter
        )

        log_rows.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=timestamp_utc(),
                epoch_duration_sec=epoch_duration,
                learning_rate=float(
                    optimizer.param_groups[0]["lr"]
                ),
            )
        )

        print(
            f"{model_name} "
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"time={epoch_duration:.1f}s"
        )

        # Best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            best_state = {
                name: value.detach().cpu().clone()
                for name, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Load current model being executed and enforce lowercase to be consistent with config
    model_name = str(config["training"]["model_name"]).lower()

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Error! Integrated training supports "
            f"{sorted(SUPPORTED_MODELS)}, got {model_name!r}."
        )

    train_cfg = config[model_name]["train"]

    val_fraction = float(
        train_cfg.get("val_fraction", 0.1)
    )
    
    # Create output directories
    out, checkpoints = prepare_output_dirs(
        config,
        model_name,
    )
    
    # Ensures Output directory exists
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        checkpoints = out / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        
    # Iterate through each chunk segments
    for chunk in chunk_specs(config):
        print(
            f"Training {model_name} for {chunk.chunk_id} "
            f"({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)"
        )
        
        data = load_chunk(config, chunk, val_fraction = val_fraction)
        train = data.splits[data.train_split].model_input
        
        # Build model based on model specific needs
        model = build_model(
            model_name=model_name,
            config=config,
            train_data=train,
        )
        
        # Train model
        model, training_results = train_model(
            model_name=model_name,
            model=model,
            train_data=train,
            config=config,
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
            },
            checkpoints/ f"{chunk.chunk_id}_{model_name}.pt",
        )


if __name__ == "__main__":
    main()