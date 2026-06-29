"""
Utility functions for the DSwinLSTM-I spectrum prediction pipeline.

Includes helpers for seeding, device selection, config loading,
normalization/denormalization, metric computation, checkpoint I/O,
logging, and plotting.
"""

import os
import random
import json
import yaml
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt


def set_seed(seed=42):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str="auto"):
    """Return a torch device. "auto" selects CUDA if available, else CPU."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config(config_path):
    """Load a YAML configuration file and return as a dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def normalize_minmax(data, dmin, dmax, target_range=(-1, 1)):
    """Min-max normalize data from [dmin, dmax] to [target_range[0], target_range[1]]."""
    eps = 1e-8
    data_norm = (data - dmin) / (dmax - dmin + eps)
    t_lo, t_hi = target_range
    data_norm = data_norm * (t_hi - t_lo) + t_lo
    return data_norm


def denormalize(data, stats):
    """Reverse min-max normalization using stored statistics.

    Args:
        data: Normalized data.
        stats: Dict with keys "dmin", "dmax", "range", "method".

    Returns:
        Data restored to original scale.
    """
    method = stats.get("method", "minmax")
    if method == "minmax":
        dmin = stats["dmin"]
        dmax = stats["dmax"]
        lo, hi = stats.get("range", [-1, 1])
        eps = 1e-8
        data_01 = (data - lo) / (hi - lo + eps)
        return data_01 * (dmax - dmin + eps) + dmin
    return data


def rmse(pred, target):
    """Root mean squared error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    """Mean absolute error."""
    return torch.mean(torch.abs(pred - target))


def r2_score(pred, target):
    """Coefficient of determination (R2) score."""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def nrmse_db(pred, target):
    """Normalized RMSE relative to the target's dynamic range (in dBm)."""
    target_range = target.max() - target.min()
    return rmse(pred, target) / (target_range + 1e-8)


def compute_metrics(pred, target):
    """Compute RMSE, MAE, R2, and NRMSE between prediction and target tensors.

    Args:
        pred, target: Tensors of the same shape.

    Returns:
        Dict of metric name -> scalar value.
    """
    pred = pred.detach()
    target = target.detach()
    return {
        "rmse": rmse(pred, target).item(),
        "mae": mae(pred, target).item(),
        "r2": r2_score(pred, target).item(),
        "nrmse_db": nrmse_db(pred, target).item(),
    }


def compute_metrics_per_node(pred, target, node_names=None):
    """Compute per-node metrics by slicing along the height (node) dimension.

    Args:
        pred: Tensor of shape (B, C, H, W).
        target: Tensor of shape (B, C, H, W).
        node_names: List of node name strings.

    Returns:
        Dict with keys like "rmse_CC1", "mae_CC2", etc.
    """
    B, C, H, W = pred.shape
    if node_names is None:
        node_names = [f"node_{i}" for i in range(H)]
    metrics = {}
    for n in range(H):
        m = compute_metrics(pred[:, :, n:n+1, :], target[:, :, n:n+1, :])
        for k, v in m.items():
            metrics[f"{k}_{node_names[n]}"] = v
    return metrics


def compute_metrics_per_horizon(pred, target):
    """Compute per-timestep metrics along the time (prediction horizon) dimension.

    Args:
        pred: Tensor of shape (B, T, C, H, W).
        target: Tensor of shape (B, T, C, H, W).

    Returns:
        Dict with keys like "rmse_t1", "rmse_t2", etc.
    """
    B, T, C, H, W = pred.shape
    metrics = {}
    for t in range(T):
        m = compute_metrics(pred[:, t], target[:, t])
        for k, v in m.items():
            metrics[f"{k}_t{t+1}"] = v
    return metrics


def save_checkpoint(path, model, optimizer, epoch, stats, config, metrics=None):
    """Save a training checkpoint with model weights, optimizer state, and metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "norm_stats": stats,
        "config": config,
        "metrics": metrics,
    }, path)


def load_checkpoint(path, device):
    """Load a training checkpoint from disk. Uses weights_only=False for full state dict compatibility."""
    return torch.load(path, map_location=device, weights_only=False)


def init_logger(log_dir, name="train"):
    """Initialize a logger that writes both to a file and to stdout."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def plot_spectrogram_comparison(ground_truth, prediction, node_idx, node_name, save_path):
    """Plot side-by-side ground truth and predicted spectrograms for a given node.

    Args:
        ground_truth: Array of shape (T, H, W) — full ground truth.
        prediction: Array of shape (T, H, W) — full prediction.
        node_idx: Index of the node to plot.
        node_name: Name label for the node.
        save_path: File path to save the figure.
    """
    gt_node = ground_truth[:, node_idx, :].T
    pred_node = prediction[:, node_idx, :].T
    vmin = min(gt_node.min(), pred_node.min())
    vmax = max(gt_node.max(), pred_node.max())

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
    im1 = axes[0].imshow(gt_node, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{node_name} — Ground Truth")
    im2 = axes[1].imshow(pred_node, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{node_name} — Prediction")
    axes[1].set_xlabel("Frequency Bin")
    axes[1].set_ylabel("Time Step")
    fig.colorbar(im1, ax=axes.ravel().tolist(), label="Power (dBm)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_analysis(errors, node_names, save_path):
    """Plot per-node error maps (prediction - ground truth) for visual analysis.

    Args:
        errors: Array of shape (T, H, W) — prediction errors.
        node_names: List of node name strings.
        save_path: File path to save the figure.
    """
    n = len(node_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False, constrained_layout=True)
    vabs = max(abs(errors.min()), abs(errors.max()))
    for i in range(n):
        err = errors[:, i, :].T
        im = axes[0, i].imshow(err, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        axes[0, i].set_title(f"{node_names[i]} Error")
        axes[0, i].set_xlabel("Time Step")
        axes[0, i].set_ylabel("Frequency Bin")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Error (dBm)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
