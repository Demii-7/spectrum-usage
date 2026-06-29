"""Utility functions for the TimeRAN training pipeline.

Provides helpers for reproducibility (set_seed), device management,
common regression metrics (RMSE, MAE, R²), checkpoint I/O, and
metric breakdowns by forecast horizon, per node, and per frequency bin.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set Python, NumPy, and PyTorch random seeds for reproducibility.

    Also sets ``PYTHONHASHSEED`` to stabilise dictionary hash randomisation.

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "auto"):
    """Resolve a device string to a ``torch.device``.

    ``"auto"`` selects CUDA if available, otherwise CPU.

    Args:
        device_str: Device string (``"auto"``, ``"cpu"``, ``"cuda"``, etc.).

    Returns:
        A ``torch.device`` instance.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Reverse z-score normalisation:  x = x_norm * std + mean.

    Args:
        data: Normalised array.
        mean: Per-channel mean (broadcastable to *data*).
        std: Per-channel standard deviation.

    Returns:
        Denormalised array in the original scale.
    """
    return data * std + mean


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(pred - target)))


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination (R²).

    A small epsilon (1e-8) in the denominator avoids division by zero
    when the target is constant.
    """
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def compute_metrics(pred: np.ndarray, target: np.ndarray):
    """Compute RMSE, MAE, and R² in a single call.

    Args:
        pred: Prediction array.
        target: Ground-truth array.

    Returns:
        Dictionary with keys ``"rmse"``, ``"mae"``, ``"r2"``.
    """
    return {
        "rmse": compute_rmse(pred, target),
        "mae": compute_mae(pred, target),
        "r2": compute_r2(pred, target),
    }


def compute_metrics_per_horizon(pred: np.ndarray, target: np.ndarray):
    """Compute metrics for each forecast time step independently.

    Args:
        pred: Array of shape (B, C, H).
        target: Array of shape (B, C, H).

    Returns:
        Dictionary like ``{"rmse_t1": ..., "mae_t1": ..., "r2_t1": ..., ...}``.
    """
    B, C, H = pred.shape
    metrics = {}
    for h in range(H):
        m = compute_metrics(pred[:, :, h], target[:, :, h])
        for k, v in m.items():
            metrics[f"{k}_t{h+1}"] = v
    return metrics


def compute_metrics_per_node(
    pred: np.ndarray, target: np.ndarray, bins_per_node: int, node_names: list[str]
):
    """Compute metrics for each sensor node (contiguous block of frequency bins).

    Args:
        pred: Array of shape (B, C, H) where C = n_nodes * bins_per_node.
        target: Array of shape (B, C, H).
        bins_per_node: Number of frequency bins belonging to each node.
        node_names: List of node labels (length must match n_nodes).

    Returns:
        Dictionary like ``{"rmse_Node_0": ..., "mae_Node_0": ..., ...}``.
    """
    B, C, H = pred.shape
    metrics = {}
    for i, name in enumerate(node_names):
        start = i * bins_per_node
        end = start + bins_per_node
        m = compute_metrics(pred[:, start:end, :], target[:, start:end, :])
        for k, v in m.items():
            metrics[f"{k}_{name}"] = v
    return metrics


def compute_metrics_per_bin(pred: np.ndarray, target: np.ndarray):
    """Compute RMSE and MAE per individual frequency bin, aggregated over
    all samples and time steps.

    Args:
        pred: Array of shape (B, C, H).
        target: Array of shape (B, C, H).

    Returns:
        Tuple of (rmse_per_bin, mae_per_bin), each an array of length C.
    """
    B, C, H = pred.shape
    rmse_per_bin = np.sqrt(np.mean((pred - target) ** 2, axis=(0, 2)))
    mae_per_bin = np.mean(np.abs(pred - target), axis=(0, 2))
    return rmse_per_bin, mae_per_bin


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    train_loss: float,
    val_metrics: dict,
    config: dict,
    norm_stats: dict | None = None,
):
    """Save a training checkpoint to disk.

    Includes model weights, optimizer state, training metadata, the full
    config, and optional normalisation statistics so that inference can
    reverse the normalisation without needing the original training set.

    Args:
        path: Destination file path.
        model: The model whose ``state_dict`` will be saved.
        optimizer: Optimizer (may be ``None``).
        epoch: Current epoch number.
        train_loss: Training loss at this checkpoint.
        val_metrics: Validation metrics dictionary.
        config: Full training configuration (used later by evaluate/inference).
        norm_stats: Optional dict with ``"mean"`` and ``"std"`` arrays.
    """
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "train_loss": train_loss,
        "val_metrics": val_metrics,
        "config": config,
        "norm_stats": norm_stats,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: torch.device):
    """Load a checkpoint saved by ``save_checkpoint``.

    Args:
        path: Path to the checkpoint file.
        device: Device to map the tensors to.

    Returns:
        Dictionary with keys ``"epoch"``, ``"model_state_dict"``,
        ``"optimizer_state_dict"``, ``"train_loss"``, ``"val_metrics"``,
        ``"config"``, and optionally ``"norm_stats"``.
    """
    return torch.load(path, map_location=device, weights_only=False)
