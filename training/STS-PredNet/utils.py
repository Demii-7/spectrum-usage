"""
Utility functions for STS-PredNet training, evaluation, and inference.

Includes device management, seeding, metric computation (RMSE, MAE, R²),
per-node and per-frequency breakdowns, checkpoint save/load, and
denormalization back to physical units.
"""
import os
import random
import json
import numpy as np
import torch


def set_seed(seed=42):
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str="auto"):
    """Return a torch.device; 'auto' selects CUDA if available else CPU."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def denormalize(data, stats):
    """Convert normalized predictions back to original physical scale.

    Supports both minmax-[-1,1] and z-score normalization methods.
    """
    method = stats.get("method", "minmax_neg1_pos1")
    if method == "minmax_neg1_pos1":
        dmin = stats["dmin"]
        dmax = stats["dmax"]
        eps = 1e-8
        return 0.5 * (data + 1.0) * (dmax - dmin + eps) + dmin
    elif method == "zscore":
        return data * stats["std"] + stats["mean"]
    return data


def rmse(pred, target):
    """Compute Root Mean Square Error between predictions and targets."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    """Compute Mean Absolute Error between predictions and targets."""
    return torch.mean(torch.abs(pred - target))


def r2_score(pred, target):
    """Compute coefficient of determination (R²).

    R² = 1 - (residual sum of squares) / (total sum of squares)
    """
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def compute_metrics(pred, target):
    """Return a dictionary of RMSE, MAE, and R² for the given tensors."""
    pred = pred.detach()
    target = target.detach()
    return {
        "rmse": rmse(pred, target).item(),
        "mae": mae(pred, target).item(),
        "r2": r2_score(pred, target).item(),
    }


def compute_metrics_per_node(pred, target, node_names=None):
    """Compute metrics separately for each node (spatial location).

    Args:
        pred: Tensor of shape (B, C, H, W).
        target: Tensor of same shape.
        node_names: Optional list of H node labels.

    Returns:
        Dict mapping '{metric}_{node_name}' to value.
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


def compute_metrics_per_frequency(pred, target):
    """Compute metrics separately for each frequency bin.

    Args:
        pred: Tensor of shape (B, C, H, W).
        target: Tensor of same shape.

    Returns:
        Dict mapping '{metric}_freq{w}' to value.
    """
    B, C, H, W = pred.shape
    metrics = {}
    for w in range(W):
        m = compute_metrics(pred[:, :, :, w:w+1], target[:, :, :, w:w+1])
        for k, v in m.items():
            metrics[f"{k}_freq{w}"] = v
    return metrics


def save_checkpoint(path, model, optimizer, epoch, stats, config, metrics):
    """Save a training checkpoint to disk with model state, optimizer, and metadata."""
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
    """Load a training checkpoint from disk and map it to the specified device."""
    return torch.load(path, map_location=device, weights_only=False)
