"""
Utility functions for training, evaluation, and visualisation of spectrum-usage
forecasting models.

Includes helpers for:
    - Reproducibility (seeding)
    - Device management
    - Metric computation (RMSE, MAE, R²) at various granularities
    - Checkpoint save/load
    - Spectrogram comparison and error analysis plots (using matplotlib Agg backend)
    - CSV and JSON output.
"""

import os, random, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """Set random seed for Python, NumPy, and PyTorch (CPU + CUDA).

    Call this at the start of training to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str="auto"):
    """Return a ``torch.device`` based on a string descriptor.

    Args:
        device_str: ``"auto"`` (default, picks CUDA if available),
            ``"cpu"``, ``"cuda"``, or a specific device index (e.g. ``"cuda:0"``).
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def denormalize(data, mean, std):
    """Reverse z-score normalisation: ``data * std + mean``.

    Both *mean* and *std* should be broadcastable to *data*.
    """
    return data * std + mean


def rmse(pred, target):
    """Root mean squared error between *pred* and *target* tensors."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    """Mean absolute error between *pred* and *target* tensors."""
    return torch.mean(torch.abs(pred - target))


def r2_score(pred, target):
    """Coefficient of determination (R²) score.

    R² = 1 - SS_res / SS_tot, where SS_res is the residual sum of squares
    and SS_tot is the total sum of squares.  Values close to 1 indicate
    a good fit.  A small epsilon prevents division by zero.
    """
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def compute_metrics(pred, target):
    """Compute RMSE, MAE, and R² between two tensors.

    Tensors are detached before computation to avoid gradient tracking.
    Returns a dictionary of scalar metric values.
    """
    pred = pred.detach()
    target = target.detach()
    return {
        "rmse": rmse(pred, target).item(),
        "mae": mae(pred, target).item(),
        "r2": r2_score(pred, target).item(),
    }


def compute_metrics_per_horizon(pred, target):
    """Compute RMSE/MAE/R² separately for each forecast horizon (time step).

    Args:
        pred, target: Tensors of shape ``(batch, t_out, n_features)``.

    Returns:
        Dictionary with keys like ``"rmse_t1"``, ``"mae_t2"``, etc.
    """
    b, t_out, d = pred.shape
    metrics = {}
    for t in range(t_out):
        m = compute_metrics(pred[:, t], target[:, t])
        for k, v in m.items():
            metrics[f"{k}_t{t+1}"] = v
    return metrics


def compute_metrics_per_node(pred, target, n_nodes, bins_per_node, node_names=None):
    """Compute RMSE/MAE/R² separately for each RF node (set of contiguous bins).

    The feature dimension is assumed to be partitioned into *n_nodes* groups
    of size *bins_per_node* (the frequency bins belonging to each node).

    Args:
        pred, target: Tensors of shape ``(batch, t_out, n_features)``.
        n_nodes: Number of RF nodes.
        bins_per_node: Number of frequency bins per node.
        node_names: Optional list of node name strings (default: ``Node0, Node1, ...``).

    Returns:
        Dictionary with keys like ``"rmse_CC2"``, ``"mae_CC2"``, etc.
    """
    b, t_out, d = pred.shape
    if node_names is None:
        node_names = [f"Node{i}" for i in range(n_nodes)]
    metrics = {}
    for n in range(n_nodes):
        start = n * bins_per_node
        end = start + bins_per_node
        m = compute_metrics(pred[:, :, start:end], target[:, :, start:end])
        for k, v in m.items():
            metrics[f"{k}_{node_names[n]}"] = v
    return metrics


def save_checkpoint(path, model, optimizer, epoch, stats, config, metrics=None):
    """Save a training checkpoint to *path*.

    Stores the model state dict, optimizer state dict, epoch number,
    normalisation statistics, full config, and optional metrics dictionary.
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
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns the full dictionary stored in the checkpoint file.
    ``weights_only=False`` allows loading of legacy PyTorch pickled objects.
    """
    return torch.load(path, map_location=device, weights_only=False)


def plot_spectrogram_comparison(pred_dbm, true_dbm, node_name, output_path, max_time_steps=500):
    """Create a two-panel figure comparing ground-truth and predicted spectrograms.

    Both panels share the same colour scale for direct visual comparison.
    The time axis is truncated to *max_time_steps* to keep the figure readable.

    Args:
        pred_dbm, true_dbm: Arrays of shape ``(time, n_freq_bins)`` in dBm.
        node_name: Label used in the plot titles (e.g. ``"CC2"``).
        output_path: Where to save the PNG figure.
        max_time_steps: Maximum number of time steps to display (default 500).
    """
    T_pred, F = pred_dbm.shape
    T_true, _ = true_dbm.shape
    T = min(T_pred, T_true, max_time_steps)
    pred_dbm = pred_dbm[:T, :]
    true_dbm = true_dbm[:T, :]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    vmin = min(pred_dbm.min(), true_dbm.min())
    vmax = max(pred_dbm.max(), true_dbm.max())

    im0 = axes[0].imshow(true_dbm.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                          interpolation="nearest", origin="lower")
    axes[0].set_title(f"{node_name} — Ground Truth (dBm)")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Frequency Bin")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_dbm.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                          interpolation="nearest", origin="lower")
    axes[1].set_title(f"{node_name} — Prediction (dBm)")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Frequency Bin")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_analysis(pred_dbm, true_dbm, output_path, max_time_steps=500):
    """Create a heatmap of absolute prediction error across time and frequency.

    The "Reds" colour map highlights regions of large error.  The time axis
    is truncated to *max_time_steps*.

    Args:
        pred_dbm, true_dbm: Arrays of shape ``(time, n_freq_bins)`` in dBm.
        output_path: Where to save the PNG figure.
        max_time_steps: Maximum number of time steps to display (default 500).
    """
    T_pred, F = pred_dbm.shape
    T_true, _ = true_dbm.shape
    T = min(T_pred, T_true, max_time_steps)
    error = np.abs(pred_dbm[:T] - true_dbm[:T])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(error.T, aspect="auto", cmap="Reds", interpolation="nearest", origin="lower")
    ax.set_title("Absolute Error (dBm)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Frequency Bin")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_json(metrics, output_path):
    """Write a metrics dictionary to a JSON file (pretty-printed with indent 2)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_csv(data, output_path):
    """Write a 2-D NumPy array to a CSV file with 6 decimal places."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, data, delimiter=",", fmt="%.6f")
