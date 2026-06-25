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
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    node_idx: int,
    node_name: str,
    t_in: int,
    save_path: str | Path,
) -> None:
    gt_node = ground_truth[:, node_idx, :].T
    pred_node = prediction[:, node_idx, :].T
    vmin = min(gt_node.min(), pred_node.min())
    vmax = max(gt_node.max(), pred_node.max())
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True,
                              constrained_layout=True)
    im0 = axes[0].imshow(gt_node, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{node_name} — Ground Truth")
    axes[0].set_ylabel("Frequency Bin")
    im1 = axes[1].imshow(pred_node, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{node_name} — Prediction")
    axes[1].set_xlabel("Time Step (future minutes)")
    axes[1].set_ylabel("Frequency Bin")
    fig.colorbar(im1, ax=axes.ravel().tolist(), label="Power (dBm)")
    fig.text(0.01, 0.99, f"Range [{vmin:.0f}, {vmax:.0f}] dBm",
             transform=fig.transFigure, fontsize=8, va="top", ha="left",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_analysis(
    errors: np.ndarray,
    node_names: list[str],
    save_path: str | Path,
) -> None:
    n = len(node_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False,
                              constrained_layout=True)
    im = None
    for i, name in enumerate(node_names):
        ax = axes[0, i]
        err = errors[:, i, :]
        im = ax.imshow(err.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_title(f"{name} — Error (Pred − GT)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Frequency Bin")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Normalized Error")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
