from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float(1 - ss_res / (ss_tot + 1e-30))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def compute_metrics_per_horizon(pred: np.ndarray, target: np.ndarray) -> dict[int, dict[str, float]]:
    T = pred.shape[1]
    results = {}
    for t in range(T):
        results[t + 1] = compute_metrics(pred[:, t], target[:, t])
    return results


def compute_metrics_per_node(pred: np.ndarray, target: np.ndarray, L: int) -> dict[int, dict[str, float]]:
    F = pred.shape[-1] // L
    results = {}
    for l in range(L):
        cols = slice(l * F, (l + 1) * F)
        results[l] = compute_metrics(pred[..., cols], target[..., cols])
    return results


def compute_metrics_per_frequency(pred: np.ndarray, target: np.ndarray) -> dict[int, dict[str, float]]:
    D = pred.shape[-1]
    results = {}
    for d in range(D):
        results[d] = compute_metrics(pred[..., d:d+1], target[..., d:d+1])
    return results


def plot_spectrogram_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    save_path: str | Path,
    title: str = "Spectrogram Comparison",
) -> None:
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    all_vals = np.concatenate([gt.ravel(), pred.ravel()])
    vmin, vmax = np.percentile(all_vals, [1, 99])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True)
    im0 = axes[0].imshow(gt.T, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Frequency Index")
    im1 = axes[1].imshow(pred.T, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Sample")
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.6, label="Power (dBm)")
    fig.suptitle(title)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_error_analysis(
    gt: np.ndarray,
    pred: np.ndarray,
    save_path: str | Path,
    title: str = "Error Analysis",
) -> None:
    err = pred.astype(np.float32) - gt.astype(np.float32)
    vmax = max(abs(np.percentile(err, 1)), abs(np.percentile(err, 99)))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    im = ax.imshow(err.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("Prediction Error (Pred - GT)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Frequency Index")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="Error (dBm)")
    fig.suptitle(title)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
