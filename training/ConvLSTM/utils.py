"""
Utility functions for the ConvLSTM spectrum prediction pipeline.

Provides shared helpers used by train.py, evaluate.py, and inference.py:
- Random seed initialization for reproducibility
- Device selection (CPU/CUDA)
- Metric computation (RMSE, MAE, R²) at overall, per-horizon, and per-node levels
- Model checkpoint save/load
- Denormalization of predictions back to physical units
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set all random seeds (Python, NumPy, PyTorch, CUDA) for reproducibility.

    Call this before any model construction or data loading to ensure
    that runs are deterministic given the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str="auto"):
    """
    Resolve a device string to a PyTorch device.

    "auto" selects CUDA if available, otherwise CPU. Any other string
    (e.g., "cpu", "cuda:0") is passed through directly.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def denormalize(data, mean, std):
    """
    Reverse z-score normalization: data * std + mean.

    Also supports min-max denormalization when mean/std are repurposed
    to store (min) and (max - min) respectively (see dataset.create_datasets).
    """
    return data * std + mean


def rmse(pred, target):
    """Root mean squared error between predictions and targets."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    """Mean absolute error between predictions and targets."""
    return torch.mean(torch.abs(pred - target))


def r2_score(pred, target):
    """
    Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot, where SS_res is the residual sum of squares
    and SS_tot is the total sum of squares. A value of 1 indicates perfect
    prediction, 0 means the model performs as well as predicting the mean,
    and negative values indicate worse-than-mean prediction.
    A small epsilon prevents division by zero when the target is constant.
    """
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def compute_metrics(pred, target):
    """
    Compute overall RMSE, MAE, and R² for a batch of predictions.

    Detaches tensors to avoid retaining computation graph memory.
    Returns scalar Python floats suitable for logging.
    """
    pred = pred.detach()
    target = target.detach()
    return {
        "rmse": rmse(pred, target).item(),
        "mae": mae(pred, target).item(),
        "r2": r2_score(pred, target).item(),
    }


def compute_metrics_per_horizon(pred, target):
    """
    Compute RMSE, MAE, R² for each individual prediction time step.

    This is useful for understanding how prediction quality degrades
    as the model forecasts further into the future.

    Returns a flat dict with keys like "rmse_t1", "rmse_t2", etc.
    """
    b, t_out, c, h, w = pred.shape
    metrics = {}
    for t in range(t_out):
        m = compute_metrics(pred[:, t], target[:, t])
        for k, v in m.items():
            metrics[f"{k}_t{t+1}"] = v
    return metrics


def compute_metrics_per_node(pred, target, node_names=None):
    """
    Compute RMSE, MAE, R² for each spatial node independently.

    Slicing along the height dimension (which corresponds to nodes in the
    spectrogram layout) isolates each node's predictions and targets.

    Returns a flat dict with keys like "rmse_Node_0", "mae_Node_0", etc.
    """
    b, t_out, c, h, w = pred.shape
    if node_names is None:
        node_names = [f"node_{i}" for i in range(h)]
    metrics = {}
    for n in range(h):
        m = compute_metrics(pred[:, :, :, n:n+1, :], target[:, :, :, n:n+1, :])
        for k, v in m.items():
            metrics[f"{k}_{node_names[n]}"] = v
    return metrics


def save_checkpoint(path, model, optimizer, epoch, stats, config, metrics):
    """
    Save a training checkpoint to disk.

    Includes model weights, optimizer state (to resume training), epoch number,
    normalization statistics (for inference on new data), the config used for
    training, and the best validation (or test) metrics.
    """
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
    """
    Load a checkpoint from disk.

    ``weights_only=False`` is required because the checkpoint contains
    non-tensor objects (config dict, norm_stats, etc.).
    """
    return torch.load(path, map_location=device, weights_only=False)
