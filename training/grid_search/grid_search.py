"""
Hyperparameter search for the integrated spectrum-prediction pipeline.

Supports VanillaLSTM (CSV) and ConvLSTM (map) with staged searches.

Three stages per model:
  0. Smoke test   — diverse configs, few epochs, validate pipeline
  1. Architecture  — search model structure (candidates/generated grid)
  2. Optimization  — search training params on top architectures

All trials share loaded data, normalization, and seed for reproducibility.
One failed trial never stops the search.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from training.common.config import load_config, resolve_path
from training.common.data import chunk_specs, load_chunk
from training.common.forecasting import forecast
from training.common.model_factory import SUPPORTED_MODELS, build_model
from training.common.runtime import device_for
from training.common.train_integrated import train_model
from training.common.windowing import build_window_loaders

VANILLA_SMOKE_GRID: list[dict[str, Any]] = [
    {"hidden_size": 8, "num_layers": 1, "dropout": 0.0, "learning_rate": 0.001, "batch_size": 32, "weight_decay": 0.0},
    {"hidden_size": 16, "num_layers": 2, "dropout": 0.1, "learning_rate": 0.003, "batch_size": 32, "weight_decay": 1e-4},
    {"hidden_size": 32, "num_layers": 2, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64, "weight_decay": 1e-4},
    {"hidden_size": 32, "num_layers": 3, "dropout": 0.1, "learning_rate": 0.003, "batch_size": 64, "weight_decay": 0.0},
]

CONVLSTM_SMOKE_GRID: list[dict[str, Any]] = [
    {"hidden_channels": [16], "kernel_size": [[1, 3]], "num_encoder_layers": 1, "decoder_hidden_channels": 16, "decoder_lstm_hidden": 64, "dropout": 0.0, "use_batch_norm": False, "learning_rate": 0.0002, "batch_size": 16, "weight_decay": 0.0},
    {"hidden_channels": [16, 32], "kernel_size": [[1, 3], [1, 3]], "num_encoder_layers": 2, "decoder_hidden_channels": 16, "decoder_lstm_hidden": 64, "dropout": 0.1, "use_batch_norm": True, "learning_rate": 0.0002, "batch_size": 16, "weight_decay": 0.001},
    {"hidden_channels": [32, 64], "kernel_size": [[1, 3], [1, 1]], "num_encoder_layers": 2, "decoder_hidden_channels": 32, "decoder_lstm_hidden": 128, "dropout": 0.2, "use_batch_norm": True, "learning_rate": 0.0001, "batch_size": 8, "weight_decay": 0.001},
    {"hidden_channels": [16, 32], "kernel_size": [[1, 5], [1, 3]], "num_encoder_layers": 2, "decoder_hidden_channels": 32, "decoder_lstm_hidden": 128, "dropout": 0.1, "use_batch_norm": False, "learning_rate": 0.0005, "batch_size": 8, "weight_decay": 0.0},
]

HORIZONS = sorted([1, 5, 15, 60])
HORIZON_STEP_MAP = {h: h - 1 for h in HORIZONS}

RESULTS_COLUMNS = [
    "search_stage", "trial_id", "status",
    "model_name", "chunk_id", "seed",
    # VanillaLSTM-specific
    "hidden_size", "num_layers", "dropout",
    "output_strategy", "bidirectional",
    # ConvLSTM-specific
    "hidden_channels", "kernel_size", "num_encoder_layers",
    "decoder_hidden_channels", "decoder_kernel_size", "decoder_lstm_hidden",
    "use_batch_norm", "cell_activation",
    "fc_hidden_channels", "fc_kernel_size", "fc_intermediate_activation",
    "use_channel_projection", "channel_projection_dim",
    # Shared
    "learning_rate", "batch_size", "weight_decay",
    "optimizer", "gradient_clip_norm",
    "epochs_requested", "epochs_completed", "best_epoch",
    "train_loss_at_best", "teacher_val_loss_at_best", "val_ar_loss",
    "rmse_t1_db", "rmse_t5_db", "rmse_t15_db", "rmse_t60_db",
    "mean_horizon_rmse_db",
    "mae_t1_db", "mae_t5_db", "mae_t15_db", "mae_t60_db",
    "mean_horizon_mae_db",
    "parameter_count", "peak_gpu_memory_mb",
    "training_seconds", "validation_seconds", "total_seconds",
    "checkpoint_path", "config_path", "error_message",
]


def _generate_vanilla_architecture_grid() -> list[dict[str, Any]]:
    combinations: list[dict[str, Any]] = []
    for hidden_size in [8, 16, 32]:
        for num_layers in [1, 2, 3]:
            valid_dropouts = [0.0] if num_layers == 1 else [0.0, 0.1, 0.2]
            for dropout in valid_dropouts:
                combinations.append({
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                })
    return combinations


def _inject_params(
    config: dict[str, Any],
    params: dict[str, Any],
) -> None:
    model_cfg = config["vanillalstm"]["model"]
    train_cfg = config["vanillalstm"]["train"]

    model_cfg["hidden_size"] = int(params["hidden_size"])
    model_cfg["num_layers"] = int(params["num_layers"])
    model_cfg["dropout"] = float(params["dropout"]) if params["num_layers"] > 1 else 0.0
    model_cfg["output_strategy"] = str(params.get("output_strategy", "final_hidden"))
    model_cfg["bidirectional"] = bool(params.get("bidirectional", False))
    model_cfg["prediction_horizon"] = int(params.get("prediction_horizon", 1))

    train_cfg["learning_rate"] = float(params["learning_rate"])
    train_cfg["batch_size"] = int(params["batch_size"])
    train_cfg["weight_decay"] = float(params["weight_decay"])
    train_cfg["optimizer"] = str(params.get("optimizer", "adam"))
    train_cfg["gradient_clip_norm"] = float(params.get("gradient_clip_norm", 1.0))
    train_cfg["early_stopping"] = bool(params.get("early_stopping", True))
    train_cfg["early_stopping_patience"] = int(params.get("early_stopping_patience", 3))
    train_cfg["seed"] = int(params.get("seed", 42))
    train_cfg["epochs"] = int(params.get("epochs", 15))
    train_cfg["val_stride"] = int(params.get("val_stride", 5))


def _inject_convlstm_params(
    config: dict[str, Any],
    params: dict[str, Any],
) -> None:
    mc = config["convlstm"]["model"]
    tc = config["convlstm"]["train"]

    mc["hidden_channels"] = [int(v) for v in params["hidden_channels"]]
    mc["kernel_size"] = [[int(v) for v in k] for k in params["kernel_size"]]
    mc["num_encoder_layers"] = int(params.get("num_encoder_layers", 2))
    mc["decoder_hidden_channels"] = int(params.get("decoder_hidden_channels", 16))
    mc["decoder_kernel_size"] = params.get("decoder_kernel_size", [1, 1])
    mc["decoder_lstm_hidden"] = int(params.get("decoder_lstm_hidden", 64))
    mc["dropout"] = float(params.get("dropout", 0.1))
    mc["use_batch_norm"] = bool(params.get("use_batch_norm", True))
    mc["cell_activation"] = str(params.get("cell_activation", "relu"))
    mc["fc_hidden_channels"] = int(params.get("fc_hidden_channels", 0))
    mc["fc_kernel_size"] = params.get("fc_kernel_size", [1, 3])
    mc["fc_intermediate_activation"] = str(params.get("fc_intermediate_activation", "relu"))
    mc["use_channel_projection"] = bool(params.get("use_channel_projection", False))
    mc["channel_projection_dim"] = int(params.get("channel_projection_dim", 16))
    mc["prediction_horizon"] = int(params.get("prediction_horizon", 1))
    mc["input_sequence_length"] = int(config["windowing"]["lookback"])

    tc["learning_rate"] = float(params["learning_rate"])
    tc["batch_size"] = int(params["batch_size"])
    tc["weight_decay"] = float(params["weight_decay"])
    tc["optimizer"] = str(params.get("optimizer", "adam"))
    tc["gradient_clip_norm"] = float(params.get("gradient_clip_norm", 5.0))
    tc["early_stopping"] = bool(params.get("early_stopping", True))
    tc["early_stopping_patience"] = int(params.get("early_stopping_patience", 3))
    tc["seed"] = int(params.get("seed", 42))
    tc["epochs"] = int(params.get("epochs", 10))
    tc["val_stride"] = int(params.get("val_stride", 10))


def _validate_no_nan(
    train_data: np.ndarray,
    val_fraction: float,
) -> None:
    split_idx = int(len(train_data) * (1.0 - val_fraction))
    train_part = train_data[:split_idx]
    val_part = train_data[split_idx:]

    if not np.isfinite(train_part).all():
        n_bad = int(np.sum(~np.isfinite(train_part)))
        raise ValueError(
            f"Training portion contains {n_bad} non-finite value(s) "
            f"(nan={np.isnan(train_part).sum()}, "
            f"inf={np.isinf(train_part).sum()})."
        )

    if not np.isfinite(val_part).all():
        n_bad = int(np.sum(~np.isfinite(val_part)))
        raise ValueError(
            f"Validation portion contains {n_bad} non-finite value(s) "
            f"(nan={np.isnan(val_part).sum()}, "
            f"inf={np.isinf(val_part).sum()})."
        )


def _post_hoc_validation(
    model: nn.Module,
    train_data: np.ndarray,
    config: dict[str, Any],
    normalization: dict[str, Any] | None,
    val_stride: int,
    device: torch.device,
    model_name: str,
) -> dict[str, float]:
    model_cfg = config[model_name]["model"]
    train_cfg = config[model_name]["train"]
    lookback = int(model_cfg["input_sequence_length"])
    rollout_horizon = max(HORIZONS)
    batch_size = int(train_cfg["batch_size"])
    val_fraction = float(train_cfg["val_fraction"])

    _, val_loader = build_window_loaders(
        data=train_data,
        lookback=lookback,
        rollout_horizon=rollout_horizon,
        batch_size=batch_size,
        val_fraction=val_fraction,
        train_stride=1,
        val_stride=val_stride,
    )

    if normalization is not None:
        std_dbm = np.asarray(normalization["std_dbm"], dtype=np.float64)
    else:
        std_dbm = np.ones(1, dtype=np.float64)

    n_freqs = len(std_dbm)
    mse_f_sums: dict[int, np.ndarray] = {h: np.zeros(n_freqs, dtype=np.float64) for h in HORIZONS}
    mae_f_sums: dict[int, np.ndarray] = {h: np.zeros(n_freqs, dtype=np.float64) for h in HORIZONS}
    counts: dict[int, int] = {h: 0 for h in HORIZONS}

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = forecast(
                model=model,
                x=x,
                prediction_horizon=1,
                rollout_horizon=rollout_horizon,
                targets=None,
            )

            if pred.ndim == 3:
                sq_err_f = (pred - y) ** 2
                abs_err_f = torch.abs(pred - y)
            elif pred.ndim == 5:
                sq_err_f = ((pred - y) ** 2).mean(dim=(3, 4))
                abs_err_f = torch.abs(pred - y).mean(dim=(3, 4))
            else:
                raise ValueError(f"Unexpected prediction shape: {pred.shape}")

            batch_samples = x.size(0)
            for h in HORIZONS:
                idx = HORIZON_STEP_MAP[h]
                mse_f_sums[h] += sq_err_f[:, idx].sum(dim=0).cpu().numpy()
                mae_f_sums[h] += abs_err_f[:, idx].sum(dim=0).cpu().numpy()
                counts[h] += batch_samples

    metrics: dict[str, float] = {}
    for h in HORIZONS:
        c = max(counts[h], 1)
        mse_f = mse_f_sums[h] / c
        mae_f = mae_f_sums[h] / c
        rmse_db = float(np.sqrt(np.mean(mse_f * std_dbm ** 2)))
        mae_db = float(np.mean(mae_f * std_dbm))
        metrics[f"rmse_t{h}_db"] = rmse_db
        metrics[f"mae_t{h}_db"] = mae_db

    hor_rmse = [metrics[f"rmse_t{h}_db"] for h in HORIZONS]
    hor_mae = [metrics[f"mae_t{h}_db"] for h in HORIZONS]
    metrics["mean_horizon_rmse_db"] = float(np.mean(hor_rmse))
    metrics["mean_horizon_mae_db"] = float(np.mean(hor_mae))

    return metrics


def _run_trial(
    trial_id: str,
    search_stage: str,
    base_config: dict[str, Any],
    model_name: str,
    train_data: np.ndarray,
    normalization: dict[str, Any] | None,
    params: dict[str, Any],
    epochs: int,
    val_stride: int,
    output_base: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)

    full_params = dict(params)
    full_params.setdefault("epochs", epochs)
    full_params.setdefault("val_stride", val_stride)
    full_params.setdefault("seed", 42)

    if model_name == "vanillalstm":
        full_params.setdefault("output_strategy", "final_hidden")
        full_params.setdefault("bidirectional", False)
        full_params.setdefault("prediction_horizon", 1)
        full_params.setdefault("optimizer", "adam")
        full_params.setdefault("gradient_clip_norm", 1.0)
        full_params.setdefault("early_stopping", True)
        full_params.setdefault("early_stopping_patience", 3)
        _inject_params(config, full_params)
    elif model_name == "convlstm":
        full_params.setdefault("prediction_horizon", 1)
        full_params.setdefault("optimizer", "adam")
        full_params.setdefault("gradient_clip_norm", 5.0)
        full_params.setdefault("early_stopping", True)
        full_params.setdefault("early_stopping_patience", 3)
        full_params.setdefault("cell_activation", "relu")
        full_params.setdefault("decoder_kernel_size", [1, 1])
        full_params.setdefault("fc_hidden_channels", 0)
        full_params.setdefault("fc_kernel_size", [1, 3])
        full_params.setdefault("fc_intermediate_activation", "relu")
        full_params.setdefault("use_channel_projection", False)
        full_params.setdefault("channel_projection_dim", 16)
        _inject_convlstm_params(config, full_params)
    else:
        raise ValueError(f"Unsupported model for search: {model_name}")

    config_path = output_base / "configs" / f"{trial_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    _validate_no_nan(train_data, float(config[model_name]["train"]["val_fraction"]))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(
        model_name=model_name,
        config=config,
        train_data=train_data,
    )

    param_count = sum(p.numel() for p in model.parameters())

    try:
        t0 = time.perf_counter()
        model, results = train_model(
            model_name=model_name,
            model=model,
            train_data=train_data,
            config=config,
        )
        total_time = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    ) if torch.cuda.is_available() else 0

    device = device_for(config)
    val_t0 = time.perf_counter()
    val_metrics = _post_hoc_validation(
        model=model,
        train_data=train_data,
        config=config,
        normalization=normalization,
        val_stride=val_stride,
        device=device,
        model_name=model_name,
    )
    val_time = time.perf_counter() - val_t0

    log_frame: pd.DataFrame = results["log_frame"]
    best_epoch = int(results["best_epoch"])
    best_row = log_frame[log_frame["epoch"] == best_epoch].iloc[0]

    epochs_completed = int(results["epochs_completed"])
    train_loss = float(results["train_loss_at_best_epoch"])
    val_ar = float(results["best_val_loss"])
    teacher_val = float(best_row.get("val_teacher_loss", float("nan")))

    checkpoint_dir = output_base / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{trial_id}_best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    epoch_log_dir = output_base / "epoch_logs"
    epoch_log_dir.mkdir(parents=True, exist_ok=True)
    log_frame.to_csv(epoch_log_dir / f"{trial_id}_epochs.csv", index=False)

    row: dict[str, Any] = {
        "search_stage": search_stage,
        "trial_id": trial_id,
        "status": "completed",
        "model_name": model_name,
        "chunk_id": str(config["data"]["chunks"][0]["id"]),
        "seed": 42,
        "learning_rate": float(full_params["learning_rate"]),
        "batch_size": int(full_params["batch_size"]),
        "weight_decay": float(full_params["weight_decay"]),
        "optimizer": str(full_params.get("optimizer", "adam")),
        "gradient_clip_norm": float(full_params.get("gradient_clip_norm", 1.0)),
        "epochs_requested": epochs,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "train_loss_at_best": train_loss,
        "teacher_val_loss_at_best": teacher_val,
        "val_ar_loss": val_ar,
        "parameter_count": param_count,
        "peak_gpu_memory_mb": round(peak_memory, 1),
        "training_seconds": round(max(total_time - val_time, 0.0), 2),
        "validation_seconds": round(val_time, 2),
        "total_seconds": round(total_time, 2),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "error_message": "",
    }

    if model_name == "vanillalstm":
        row["hidden_size"] = int(full_params["hidden_size"])
        row["num_layers"] = int(full_params["num_layers"])
        row["dropout"] = float(full_params["dropout"]) if full_params["num_layers"] > 1 else 0.0
        row["output_strategy"] = "final_hidden"
        row["bidirectional"] = False
    elif model_name == "convlstm":
        row["hidden_channels"] = json.dumps(full_params["hidden_channels"])
        row["kernel_size"] = json.dumps(full_params["kernel_size"])
        row["num_encoder_layers"] = int(full_params.get("num_encoder_layers", 2))
        row["decoder_hidden_channels"] = int(full_params.get("decoder_hidden_channels", 16))
        row["decoder_kernel_size"] = json.dumps(full_params.get("decoder_kernel_size", [1, 1]))
        row["decoder_lstm_hidden"] = int(full_params.get("decoder_lstm_hidden", 64))
        row["dropout"] = float(full_params.get("dropout", 0.1))
        row["use_batch_norm"] = bool(full_params.get("use_batch_norm", True))
        row["cell_activation"] = str(full_params.get("cell_activation", "relu"))
        row["fc_hidden_channels"] = int(full_params.get("fc_hidden_channels", 0))
        row["fc_kernel_size"] = json.dumps(full_params.get("fc_kernel_size", [1, 3]))
        row["fc_intermediate_activation"] = str(full_params.get("fc_intermediate_activation", "relu"))
        row["use_channel_projection"] = bool(full_params.get("use_channel_projection", False))
        row["channel_projection_dim"] = int(full_params.get("channel_projection_dim", 16))

    for h in HORIZONS:
        row[f"rmse_t{h}_db"] = val_metrics[f"rmse_t{h}_db"]
        row[f"mae_t{h}_db"] = val_metrics[f"mae_t{h}_db"]

    row["mean_horizon_rmse_db"] = val_metrics["mean_horizon_rmse_db"]
    row["mean_horizon_mae_db"] = val_metrics["mean_horizon_mae_db"]

    for col in RESULTS_COLUMNS:
        if col not in row:
            row[col] = ""

    return row


def _failed_trial_row(
    trial_id: str,
    search_stage: str,
    model_name: str,
    params: dict[str, Any],
    error_message: str,
    epochs_requested: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {col: "" for col in RESULTS_COLUMNS}
    row["search_stage"] = search_stage
    row["trial_id"] = trial_id
    row["status"] = "failed"
    row["model_name"] = model_name
    row["seed"] = 42
    row["hidden_size"] = params.get("hidden_size", "")
    row["num_layers"] = params.get("num_layers", "")
    row["dropout"] = params.get("dropout", "")
    row["hidden_channels"] = json.dumps(params.get("hidden_channels", ""))
    row["learning_rate"] = params.get("learning_rate", "")
    row["batch_size"] = params.get("batch_size", "")
    row["weight_decay"] = params.get("weight_decay", "")
    row["epochs_requested"] = epochs_requested
    row["error_message"] = str(error_message)[:1000]
    return row


def _build_smoke_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "vanillalstm":
        return VANILLA_SMOKE_GRID
    elif model_name == "convlstm":
        return CONVLSTM_SMOKE_GRID
    raise ValueError(f"No smoke grid for {model_name}")


def _build_architecture_combos(
    hp_cfg: dict[str, Any],
    model_name: str,
) -> list[dict[str, Any]]:
    if model_name == "vanillalstm":
        return _generate_vanilla_architecture_grid()
    elif model_name == "convlstm":
        return list(hp_cfg.get("architecture_search", {}).get("candidates", []))
    raise ValueError(f"No architecture grid for {model_name}")


def _build_optimization_combos(
    hp_cfg: dict[str, Any],
    model_name: str,
    top_archs: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str]]:
    combos: list[tuple[dict[str, Any], str]] = []

    if model_name == "vanillalstm":
        opt_cfg = hp_cfg.get("optimization_search", {})
        lrs: list[float] = [float(v) for v in opt_cfg.get("learning_rate", [0.001])]
        bss: list[int] = [int(v) for v in opt_cfg.get("batch_size", [32])]
        wds: list[float] = [float(v) for v in opt_cfg.get("weight_decay", [0.0])]
        for ta in top_archs:
            for lr in lrs:
                for bs in bss:
                    for wd in wds:
                        combos.append(({
                            "hidden_size": ta["hidden_size"],
                            "num_layers": ta["num_layers"],
                            "dropout": ta["dropout"],
                            "learning_rate": lr,
                            "batch_size": bs,
                            "weight_decay": wd,
                        }, f"hs={ta['hidden_size']} nl={ta['num_layers']} lr={lr} bs={bs} wd={wd}"))
    elif model_name == "convlstm":
        opt_candidates = hp_cfg.get("optimization_search", {}).get("candidates", [])
        for ta in top_archs:
            for cand in opt_candidates:
                merged = dict(ta)
                merged.update(cand)
                desc = (f"hs_ch={json.dumps(ta.get('hidden_channels',''))} "
                        f"lr={cand.get('learning_rate','')} "
                        f"wd={cand.get('weight_decay','')} "
                        f"do={cand.get('dropout','')} "
                        f"bn={cand.get('use_batch_norm','')}")
                combos.append((merged, desc))
    else:
        raise ValueError(f"No optimization grid for {model_name}")

    return combos


def _smoke_trial_description(
    params: dict[str, Any],
    model_name: str,
) -> str:
    if model_name == "vanillalstm":
        return (f"hs={params['hidden_size']} nl={params['num_layers']} "
                f"do={params['dropout']} lr={params['learning_rate']} "
                f"bs={params['batch_size']}")
    elif model_name == "convlstm":
        hc = json.dumps(params['hidden_channels'])
        ks = json.dumps(params['kernel_size'])
        return (f"hc={hc} ks={ks} nl={params.get('num_encoder_layers','?')} "
                f"lr={params['learning_rate']} bs={params['batch_size']}")
    return str(params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search for spectrum-prediction models."
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config file (default: config.yaml in script dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hp_cfg = config.get("hyperparameter_search")
    if not hp_cfg or not hp_cfg.get("enabled", False):
        print("Hyperparameter search is disabled in config.")
        return

    model_name = str(config["training"]["model_name"]).lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Grid search only supports {sorted(SUPPORTED_MODELS)}, "
            f"got {model_name!r}."
        )

    val_fraction = float(
        config[model_name]["train"].get("val_fraction", 0.15)
    )
    output_base = resolve_path(hp_cfg["output_dir"])
    output_base.mkdir(parents=True, exist_ok=True)

    chunks = chunk_specs(config)
    if not chunks:
        raise ValueError("No chunks defined in config.")
    chunk = chunks[0]

    print(
        f"Loading data for {chunk.chunk_id} "
        f"({chunk.start_mhz}-{chunk.end_mhz} MHz) ..."
    )
    data = load_chunk(config, chunk, val_fraction=val_fraction)
    train_data = data.splits[data.train_split].model_input
    normalization = data.normalization
    print(f"  Model:           {model_name}")
    print(f"  Train data shape: {train_data.shape}")
    print(f"  Normalization:    {'present' if normalization is not None else 'N/A'}")

    _validate_no_nan(train_data, val_fraction)
    print("  NaN / Inf check:  PASSED")

    all_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    runs_log: list[dict[str, Any]] = []
    trial_counter = 0

    def next_trial_id(prefix: str) -> str:
        nonlocal trial_counter
        trial_counter += 1
        return f"{prefix}_{trial_counter:04d}"

    # ── Stage 0: Smoke test ──────────────────────────────────────────────
    smoke_cfg = hp_cfg.get("smoke_test", {})
    if smoke_cfg.get("enabled", False):
        smoke_epochs = int(smoke_cfg.get("epochs", 5))
        smoke_vs = int(
            smoke_cfg.get(
                "validation_stride",
                hp_cfg.get("architecture_search", {}).get("validation_stride", 5),
            )
        )
        smoke_grid = _build_smoke_grid(model_name)
        print(f"\n{'='*60}")
        print(f"Stage 0: Smoke test ({len(smoke_grid)} trials x {smoke_epochs} epochs)")
        print(f"{'='*60}")

        for i, s_params in enumerate(smoke_grid, 1):
            tid = next_trial_id("smoke")
            desc = _smoke_trial_description(s_params, model_name)
            print(f"\n  Smoke trial {i}/{len(smoke_grid)} [{tid}]: {desc}")
            try:
                row = _run_trial(
                    trial_id=tid,
                    search_stage="smoke",
                    base_config=config,
                    model_name=model_name,
                    train_data=train_data,
                    normalization=normalization,
                    params=s_params,
                    epochs=smoke_epochs,
                    val_stride=smoke_vs,
                    output_base=output_base,
                )
                all_rows.append(row)
                runs_log.append({
                    k: row[k]
                    for k in ("trial_id", "status", "mean_horizon_rmse_db", "total_seconds")
                    if k in row
                })
                print(
                    f"    --> Completed.  "
                    f"RMSE={row.get('mean_horizon_rmse_db', '?'):.4f} dB  "
                    f"time={row.get('total_seconds', '?'):.1f}s"
                )
            except Exception as exc:
                ft_row = _failed_trial_row(
                    tid, "smoke", model_name, s_params,
                    traceback.format_exc(), smoke_epochs,
                )
                failed_rows.append(ft_row)
                all_rows.append(ft_row)
                runs_log.append({
                    "trial_id": tid,
                    "status": "failed",
                    "error": str(exc)[:200],
                })
                print(f"    --> FAILED: {exc}")

    # ── Stage 1: Architecture search ─────────────────────────────────────
    arch_cfg = hp_cfg.get("architecture_search", {})
    arch_epochs = int(arch_cfg.get("epochs", 15))
    arch_vs = int(arch_cfg.get("validation_stride", 5))
    fixed = arch_cfg.get("fixed", {})

    arch_combos = _build_architecture_combos(hp_cfg, model_name)
    print(f"\n{'='*60}")
    print(f"Stage 1: Architecture search ({len(arch_combos)} trials x {arch_epochs} epochs)")
    print(f"{'='*60}")

    for i, a_params in enumerate(arch_combos, 1):
        tid = next_trial_id("arch")
        merged_params = dict(a_params)
        for k, v in fixed.items():
            if k not in merged_params:
                merged_params[k] = v

        if model_name == "vanillalstm":
            desc = f"hs={a_params['hidden_size']} nl={a_params['num_layers']} do={a_params['dropout']}"
        else:
            hc = json.dumps(a_params.get('hidden_channels', ''))
            desc = f"hc={hc}"

        print(f"\n  Arch trial {i}/{len(arch_combos)} [{tid}]: {desc}")
        try:
            row = _run_trial(
                trial_id=tid,
                search_stage="architecture",
                base_config=config,
                model_name=model_name,
                train_data=train_data,
                normalization=normalization,
                params=merged_params,
                epochs=arch_epochs,
                val_stride=arch_vs,
                output_base=output_base,
            )
            all_rows.append(row)
            runs_log.append({
                k: row[k]
                for k in ("trial_id", "status", "mean_horizon_rmse_db", "total_seconds")
                if k in row
            })
            print(
                f"    --> Completed.  "
                f"RMSE={row.get('mean_horizon_rmse_db', '?'):.4f} dB  "
                f"time={row.get('total_seconds', '?'):.1f}s"
            )
        except Exception as exc:
            ft_row = _failed_trial_row(
                tid, "architecture", model_name, a_params,
                traceback.format_exc(), arch_epochs,
            )
            failed_rows.append(ft_row)
            all_rows.append(ft_row)
            runs_log.append({
                "trial_id": tid,
                "status": "failed",
                "error": str(exc)[:200],
            })
            print(f"    --> FAILED: {exc}")

    # ── Select top architectures ─────────────────────────────────────────
    completed_arch = [
        r for r in all_rows
        if r.get("search_stage") == "architecture" and r.get("status") == "completed"
    ]
    top_k = int(
        hp_cfg.get("optimization_search", {}).get("top_architectures", 2)
    )
    completed_arch.sort(
        key=lambda r: float(r.get("mean_horizon_rmse_db", float("inf")))
    )
    top_archs = completed_arch[:top_k]

    print(f"\n  Top {len(top_archs)} architecture(s) by mean_horizon_rmse_db:")
    for rank, ta in enumerate(top_archs, 1):
        if model_name == "vanillalstm":
            print(
                f"    #{rank}: hs={ta['hidden_size']} nl={ta['num_layers']} "
                f"do={ta['dropout']} RMSE={ta['mean_horizon_rmse_db']:.4f} dB "
                f"({ta['trial_id']})"
            )
        else:
            print(
                f"    #{rank}: hc={ta.get('hidden_channels','')} "
                f"RMSE={ta['mean_horizon_rmse_db']:.4f} dB "
                f"({ta['trial_id']})"
            )

    # ── Stage 2: Optimization search ─────────────────────────────────────
    opt_cfg = hp_cfg.get("optimization_search", {})
    opt_epochs = int(opt_cfg.get("epochs", 15))
    opt_vs = int(opt_cfg.get("validation_stride", 5))

    opt_combos = _build_optimization_combos(hp_cfg, model_name, top_archs)
    total_opt = len(opt_combos)
    print(f"\n{'='*60}")
    print(f"Stage 2: Optimization search ({total_opt} trials x {opt_epochs} epochs)")
    print(f"{'='*60}")

    for opt_counter, (opt_params, desc) in enumerate(opt_combos, 1):
        tid = next_trial_id("opt")
        print(f"\n  Opt trial {opt_counter}/{total_opt} [{tid}]: {desc}")
        try:
            row = _run_trial(
                trial_id=tid,
                search_stage="optimization",
                base_config=config,
                model_name=model_name,
                train_data=train_data,
                normalization=normalization,
                params=opt_params,
                epochs=opt_epochs,
                val_stride=opt_vs,
                output_base=output_base,
            )
            all_rows.append(row)
            runs_log.append({
                k: row[k]
                for k in ("trial_id", "status", "mean_horizon_rmse_db", "total_seconds")
                if k in row
            })
            print(
                f"    --> Completed.  "
                f"RMSE={row.get('mean_horizon_rmse_db', '?'):.4f} dB  "
                f"time={row.get('total_seconds', '?'):.1f}s"
            )
        except Exception as exc:
            ft_row = _failed_trial_row(
                tid, "optimization", model_name, opt_params,
                traceback.format_exc(), opt_epochs,
            )
            failed_rows.append(ft_row)
            all_rows.append(ft_row)
            runs_log.append({
                "trial_id": tid,
                "status": "failed",
                "error": str(exc)[:200],
            })
            print(f"    --> FAILED: {exc}")

    # ── Write results ────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows, columns=RESULTS_COLUMNS)
    results_csv_path = output_base / "grid_search_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults written to {results_csv_path}")

    runs_jsonl_path = output_base / "grid_search_runs.jsonl"
    with open(runs_jsonl_path, "w") as f:
        for entry in runs_log:
            f.write(json.dumps(entry) + "\n")
    print(f"Runs log written to {runs_jsonl_path}")

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows, columns=RESULTS_COLUMNS)
        failed_csv_path = output_base / "failed_trials.csv"
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"Failed trials written to {failed_csv_path}")

    completed_df = results_df[results_df["status"] == "completed"].copy()
    if len(completed_df) > 0:
        completed_df = completed_df.sort_values(
            "mean_horizon_rmse_db", ascending=True
        )
        best_row = completed_df.iloc[0]

        best_params: dict[str, Any] = {}
        if model_name == "vanillalstm":
            best_params = {
                "hidden_size": (
                    int(best_row["hidden_size"])
                    if best_row["hidden_size"] != "" else None
                ),
                "num_layers": (
                    int(best_row["num_layers"])
                    if best_row["num_layers"] != "" else None
                ),
                "dropout": (
                    float(best_row["dropout"])
                    if best_row["dropout"] != "" else None
                ),
            }
        elif model_name == "convlstm":
            best_params = {
                "hidden_channels": (
                    json.loads(best_row["hidden_channels"])
                    if best_row["hidden_channels"] != "" else None
                ),
                "kernel_size": (
                    json.loads(best_row["kernel_size"])
                    if best_row["kernel_size"] != "" else None
                ),
                "num_encoder_layers": (
                    int(best_row["num_encoder_layers"])
                    if best_row["num_encoder_layers"] != "" else None
                ),
                "decoder_hidden_channels": (
                    int(best_row["decoder_hidden_channels"])
                    if best_row["decoder_hidden_channels"] != "" else None
                ),
                "decoder_lstm_hidden": (
                    int(best_row["decoder_lstm_hidden"])
                    if best_row["decoder_lstm_hidden"] != "" else None
                ),
                "dropout": (
                    float(best_row["dropout"])
                    if best_row["dropout"] != "" else None
                ),
                "use_batch_norm": (
                    bool(best_row["use_batch_norm"])
                    if best_row["use_batch_norm"] != "" else None
                ),
                "cell_activation": (
                    str(best_row["cell_activation"])
                    if best_row["cell_activation"] != "" else None
                ),
            }
        best_params["learning_rate"] = (
            float(best_row["learning_rate"])
            if best_row["learning_rate"] != "" else None
        )
        best_params["batch_size"] = (
            int(best_row["batch_size"])
            if best_row["batch_size"] != "" else None
        )
        best_params["weight_decay"] = (
            float(best_row["weight_decay"])
            if best_row["weight_decay"] != "" else None
        )

        summary = {
            "search_name": hp_cfg.get("search_name", "unnamed_search"),
            "model_name": model_name,
            "total_trials": len(results_df),
            "completed_trials": len(completed_df),
            "failed_trials": len(failed_rows),
            "selection_metric": "mean_horizon_rmse_db",
            "best_trial_id": str(best_row["trial_id"]),
            "best_parameters": best_params,
            "best_metrics": {
                "rmse_t1_db": float(best_row.get("rmse_t1_db", 0.0)),
                "rmse_t5_db": float(best_row.get("rmse_t5_db", 0.0)),
                "rmse_t15_db": float(best_row.get("rmse_t15_db", 0.0)),
                "rmse_t60_db": float(best_row.get("rmse_t60_db", 0.0)),
                "mean_horizon_rmse_db": float(
                    best_row.get("mean_horizon_rmse_db", 0.0)
                ),
            },
        }

        summary_path = output_base / "search_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")

        print(f"\n{'='*60}")
        print(f"BEST CONFIGURATION: {summary['best_trial_id']}")
        print(f"{'='*60}")
        for k, v in summary["best_parameters"].items():
            print(f"  {k}: {v}")
        print(
            f"  mean_horizon_rmse_db: "
            f"{summary['best_metrics']['mean_horizon_rmse_db']:.4f} dB"
        )
    else:
        print("\nNo completed trials.")
        summary = {
            "search_name": hp_cfg.get("search_name", "unnamed_search"),
            "model_name": model_name,
            "total_trials": len(results_df),
            "completed_trials": 0,
            "failed_trials": len(failed_rows),
            "selection_metric": "mean_horizon_rmse_db",
            "best_trial_id": None,
            "best_parameters": None,
            "best_metrics": None,
        }
        summary_path = output_base / "search_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
