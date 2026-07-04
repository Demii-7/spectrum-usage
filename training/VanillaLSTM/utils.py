from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = [
        "data",
        "windowing",
        "split",
        "preprocessing",
        "model",
        "training",
        "evaluation",
        "paths",
        "device",
    ]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    n_frequency_bins = int(config["data"]["n_frequency_bins"])
    input_size = int(config["model"]["input_size"])
    if input_size != n_frequency_bins:
        raise ValueError(f"model.input_size ({input_size}) must match data.n_frequency_bins ({n_frequency_bins}).")

    ratios = (
        float(config["split"]["train_ratio"])
        + float(config["split"]["val_ratio"])
        + float(config["split"]["test_ratio"])
    )
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios:.6f}.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cuda_available = torch.cuda.is_available()
        return torch.device("cuda" if cuda_available else "cpu")
    return torch.device(device_name)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_regression_metrics(pred_dbm: np.ndarray, target_dbm: np.ndarray) -> dict[str, float]:
    pred = pred_dbm.astype(np.float64, copy=False)
    target = target_dbm.astype(np.float64, copy=False)
    error = pred - target
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_per_horizon_rmse(pred_dbm: np.ndarray, target_dbm: np.ndarray) -> dict[str, float]:
    return {
        f"t+{horizon + 1}": float(np.sqrt(np.mean((pred_dbm[:, horizon] - target_dbm[:, horizon]) ** 2)))
        for horizon in range(pred_dbm.shape[1])
    }


def compute_per_frequency_rmse(pred_dbm: np.ndarray, target_dbm: np.ndarray) -> list[float]:
    return np.sqrt(np.mean((pred_dbm - target_dbm) ** 2, axis=(0, 1))).astype(np.float64).tolist()


def save_json(path: str | Path, payload: Any) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_array_csv(path: str | Path, values: np.ndarray) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values.astype(np.float32), delimiter=",", fmt="%.6f")


def flatten_forecasts(values: np.ndarray) -> np.ndarray:
    return values.reshape(values.shape[0] * values.shape[1], values.shape[2])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    best_val_loss: float,
) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def save_normalization_stats(path: str | Path, mean_dbm: np.ndarray, std_dbm: np.ndarray) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean_dbm": torch.from_numpy(mean_dbm.astype(np.float32)),
            "std_dbm": torch.from_numpy(std_dbm.astype(np.float32)),
        },
        path,
    )


def load_normalization_stats(path: str | Path) -> dict[str, np.ndarray]:
    payload = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    return {
        "mean_dbm": payload["mean_dbm"].detach().cpu().numpy().astype(np.float32),
        "std_dbm": payload["std_dbm"].detach().cpu().numpy().astype(np.float32),
    }


def plot_spectrogram(pred_dbm: np.ndarray, target_dbm: np.ndarray, site_name: str, output_path: str | Path) -> None:
    vmin = float(min(np.min(pred_dbm), np.min(target_dbm)))
    vmax = float(max(np.max(pred_dbm), np.max(target_dbm)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ground_truth = axes[0].imshow(target_dbm, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    prediction = axes[1].imshow(pred_dbm, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{site_name} Ground Truth")
    axes[1].set_title(f"{site_name} Prediction")
    axes[0].set_xlabel("Frequency Bin")
    axes[1].set_xlabel("Frequency Bin")
    axes[0].set_ylabel("Forecast Step")
    axes[1].set_ylabel("Forecast Step")

    colorbar = fig.colorbar(prediction, ax=axes, shrink=0.9)
    colorbar.set_label("Power (dBm)")

    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_error_analysis(
    pred_dbm: np.ndarray,
    target_dbm: np.ndarray,
    per_frequency_rmse: list[float],
    output_path: str | Path,
) -> None:
    abs_error = np.abs(pred_dbm - target_dbm)
    mean_error_heatmap = np.mean(abs_error, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    heatmap = axes[0].imshow(mean_error_heatmap, aspect="auto", origin="lower")
    axes[0].set_title("Mean Absolute Error Heatmap")
    axes[0].set_xlabel("Frequency Bin")
    axes[0].set_ylabel("Forecast Step")
    colorbar = fig.colorbar(heatmap, ax=axes[0], shrink=0.9)
    colorbar.set_label("Absolute Error (dB)")

    axes[1].plot(np.arange(len(per_frequency_rmse)), per_frequency_rmse)
    axes[1].set_title("Per-Frequency RMSE")
    axes[1].set_xlabel("Frequency Bin")
    axes[1].set_ylabel("RMSE (dB)")
    axes[1].grid(True, alpha=0.3)

    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
