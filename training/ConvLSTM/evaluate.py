"""
Evaluation script for a trained ConvLSTM spectrum-prediction model.

Loads a checkpoint, runs inference on the test set, computes overall and
per-horizon/per-node metrics (RMSE, MAE, R²), and generates diagnostic plots:
- Spectrogram comparisons (ground truth vs. prediction) for each node
- Error analysis heatmaps (prediction minus ground truth)

All results (CSV, metrics JSON, PNG figures) are saved to an output directory.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
# Use non-interactive Agg backend so the script works on headless servers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import create_datasets, create_interpolated_map_datasets
from model import ConvLSTMPredictor
from utils import (
    get_device, compute_metrics, compute_metrics_per_horizon,
    compute_metrics_per_node, load_checkpoint, denormalize,
)


def plot_spectrogram_comparison(ground_truth, prediction, node_idx, node_name,
                                 t_in, save_path):
    """
    Plot a side-by-side spectrogram comparison of ground truth and prediction
    for a single node across the entire prediction horizon.

    A shared color scale (based on the min/max across both panels) ensures
    the visual comparison is fair. The input sequence length is noted in
    the filename but not plotted, as only the future window is shown.
    """
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
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_per_frequency_rmse(pred, target):
    """Compute RMSE per frequency channel across batch, time, and spatial dims.

    Args:
        pred, target: (B, T_out, F, H, W) tensors.
    Returns:
        ndarray of shape (F,) — RMSE per frequency channel.
    """
    diff = (pred - target) ** 2
    return torch.mean(diff, dim=(0, 1, 3, 4)).sqrt().cpu().numpy()


def compute_spatial_rmse_map(pred, target):
    """Compute RMSE per spatial cell across batch, time, and frequency dims.

    Args:
        pred, target: (B, T_out, F, H, W) tensors.
    Returns:
        ndarray of shape (H, W) — RMSE per grid cell.
    """
    diff = (pred - target) ** 2
    return torch.mean(diff, dim=(0, 1, 2)).sqrt().cpu().numpy()


def plot_map_comparison(ground_truth, prediction, freq_idx, t_idx, save_path):
    """Side-by-side spatial map comparison at a given (time, freq) slice.

    Args:
        ground_truth, prediction: (T_out, F, H, W) numpy arrays.
        freq_idx: frequency channel index to plot.
        t_idx: time step index to plot.
        save_path: output PNG path.
    """
    gt_slice = ground_truth[t_idx, freq_idx, :, :]
    pred_slice = prediction[t_idx, freq_idx, :, :]
    vmin = min(gt_slice.min(), pred_slice.min())
    vmax = max(gt_slice.max(), pred_slice.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].imshow(gt_slice, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth (t+{t_idx+1}, freq={freq_idx})")
    axes[0].set_xlabel("Grid X")
    axes[0].set_ylabel("Grid Y")
    im1 = axes[1].imshow(pred_slice, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Prediction (t+{t_idx+1}, freq={freq_idx})")
    axes[1].set_xlabel("Grid X")
    axes[1].set_ylabel("Grid Y")
    fig.colorbar(im1, ax=axes.tolist(), label="Power (dBm)")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_spatial_rmse_map(rmse_map, save_path):
    """Plot a spatial heatmap of RMSE per grid cell.

    Args:
        rmse_map: (H, W) numpy array.
        save_path: output PNG path.
    """
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(rmse_map, aspect="auto", cmap="hot")
    ax.set_title("Spatial RMSE Map")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    fig.colorbar(im, ax=ax, label="RMSE (dBm)")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_frequency_rmse(rmse_per_freq, save_path):
    """Bar/line plot of RMSE per frequency channel.

    Args:
        rmse_per_freq: (F,) numpy array.
        save_path: output PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(rmse_per_freq, marker=".")
    ax.set_title("Per-Frequency RMSE")
    ax.set_xlabel("Frequency Channel Index")
    ax.set_ylabel("RMSE (dBm)")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_analysis(errors, node_names, save_path):
    """
    Plot per-node error heatmaps across all test samples.

    Each heatmap shows (prediction - ground truth) for every frequency bin
    across all test-set windows. The RdBu_r diverging colormap highlights
    overestimation (red) and underestimation (blue), clipped to ±3 normalized
    units for readability.
    """
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
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    """
    Load a trained checkpoint, run evaluation on the test set, compute metrics,
    and generate diagnostic plots.

    CSV mode: spectrogram comparisons + per-node error analysis.
    Interpolated-map mode: spatial map comparisons + per-frequency RMSE +
                           spatial RMSE heatmap.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = get_device("auto")

    # Load checkpoint — contains model weights, config, and normalization stats.
    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    stats = ckpt["norm_stats"]
    # Allow overriding the config baked into the checkpoint with an external file.
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

    dcfg = config["data"]
    wcfg = config["windowing"]
    scfg = config["split"]
    data_format = dcfg.get("format", "csv")

    if data_format == "interpolated_map":
        map_path = dcfg["map_path"]
        if not os.path.exists(map_path):
            map_path = os.path.join(os.path.dirname(__file__), "..", "..", map_path)
        _, _, test_ds, _ = create_interpolated_map_datasets(
            map_path=map_path,
            map_key=dcfg.get("map_key", "map_db"),
            t_in=wcfg["input_sequence_length"],
            t_out=wcfg["prediction_horizon"],
            stride=wcfg.get("stride", 1),
            train_stride=wcfg.get("train_stride"),
            val_stride=wcfg.get("val_stride"),
            test_stride=wcfg.get("test_stride"),
            train_ratio=scfg["train_ratio"],
            val_ratio=scfg["val_ratio"],
            chronological=scfg["chronological_split"],
            normalization=config["preprocessing"]["normalization"],
            fit_on_train_only=config["preprocessing"]["fit_on_train_only"],
            imputation_cfg=config["preprocessing"].get("imputation"),
        )
    else:
        csv_path = dcfg["dataset_path"]
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "..", csv_path)
        _, _, test_ds, _ = create_datasets(
            csv_path=csv_path,
            n_nodes=dcfg["n_nodes"],
            n_bins=dcfg["n_bins_per_node"],
            t_in=wcfg["input_sequence_length"],
            t_out=wcfg["prediction_horizon"],
            stride=wcfg.get("stride", 1),
            train_stride=wcfg.get("train_stride"),
            val_stride=wcfg.get("val_stride"),
            test_stride=wcfg.get("test_stride"),
            train_ratio=scfg["train_ratio"],
            val_ratio=scfg["val_ratio"],
            chronological=scfg["chronological_split"],
            normalization=config["preprocessing"]["normalization"],
            fit_on_train_only=config["preprocessing"]["fit_on_train_only"],
        )

    if test_ds is None or len(test_ds) == 0:
        print("No test set available.")
        return

    loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    model = ConvLSTMPredictor(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run inference on the entire test set.
    all_pred, all_target = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_pred.append(pred)
            all_target.append(y)

    pred = torch.cat(all_pred, dim=0)
    target = torch.cat(all_target, dim=0)

    overall = compute_metrics(pred, target)
    per_horizon = compute_metrics_per_horizon(pred, target)

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    # Denormalize predictions and targets back to dBm for interpretable output.
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])
    pred_dbm = denormalize(pred_np, mean, std)
    target_dbm = denormalize(target_np, mean, std)

    print(f"pred_np range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
    print(f"target_np range: [{target_np.min():.4f}, {target_np.max():.4f}]")
    print(f"pred_dbm range: [{pred_dbm.min():.2f}, {pred_dbm.max():.2f}]")
    print(f"target_dbm range: [{target_dbm.min():.2f}, {target_dbm.max():.2f}]")

    bs, t_out, c, h, w = pred_dbm.shape
    pred_flat = pred_dbm.reshape(bs, t_out, -1)
    target_flat = target_dbm.reshape(bs, t_out, -1)

    # Save only the first test sample's predictions as a human-readable CSV.
    np.savetxt(os.path.join(output_dir, "predictions.csv"),
               pred_flat[0], delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_dir, "ground_truth.csv"),
               target_flat[0], delimiter=",", fmt="%.6f")

    print("=== Evaluation Report ===")
    print(f"Overall RMSE: {overall['rmse']:.4f}")
    print(f"Overall MAE:  {overall['mae']:.4f}")
    print(f"Overall R²:   {overall['r2']:.4f}")
    print()
    print("Per-horizon RMSE:")
    for h_val in args.horizons:
        key = f"rmse_t{h_val}"
        if key in per_horizon:
            print(f"  t={h_val}: {per_horizon[key]:.4f}")

    if data_format == "interpolated_map":
        n_freq = c  # channels = frequency bins in map mode
        rmse_per_freq = compute_per_frequency_rmse(pred, target)
        spatial_rmse = compute_spatial_rmse_map(pred, target)

        print("\nTop-5 worst frequencies:")
        worst_idx = np.argsort(rmse_per_freq)[-5:][::-1]
        for fi in worst_idx:
            print(f"  freq={fi:3d}: RMSE={rmse_per_freq[fi]:.4f}")
        print(f"\nSpatial RMSE: mean={spatial_rmse.mean():.4f}, "
              f"max={spatial_rmse.max():.4f} at ({np.unravel_index(spatial_rmse.argmax(), spatial_rmse.shape)})")

        all_metrics = {**overall, **per_horizon}
        for fi in range(n_freq):
            all_metrics[f"rmse_freq_{fi}"] = float(rmse_per_freq[fi])
        all_metrics["spatial_rmse_mean"] = float(spatial_rmse.mean())
        all_metrics["spatial_rmse_max"] = float(spatial_rmse.max())
    else:
        n_nodes = dcfg.get("n_nodes", 3)
        node_names = dcfg.get("node_names", None)
        if node_names is None:
            node_names = [f"Node_{i}" for i in range(n_nodes)]
            print(f"Warning: no node_names in config; using {node_names}")
        elif len(node_names) != n_nodes:
            print(f"Warning: config has {len(node_names)} names but n_nodes={n_nodes}; "
                  f"falling back to generic names")
            node_names = [f"Node_{i}" for i in range(n_nodes)]
        per_node = compute_metrics_per_node(pred, target, node_names)

        print("\nPer-node RMSE:")
        for name in node_names:
            key = f"rmse_{name}"
            if key in per_node:
                print(f"  {name}: {per_node[key]:.4f}")

        all_metrics = {**overall, **per_horizon, **per_node}

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) if not isinstance(v, (int, float)) else v
                   for k, v in all_metrics.items()}, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate plots based on data format.
    if data_format == "interpolated_map":
        n_freq = c
        # Plot a few frequency slices at the last time step.
        mid_t = t_out // 2
        for freq_idx in [0, n_freq // 4, n_freq // 2, 3 * n_freq // 4, n_freq - 1]:
            if freq_idx >= n_freq:
                continue
            plot_path = os.path.join(output_dir, f"map_comparison_freq{freq_idx}_t{mid_t}.png")
            plot_map_comparison(target_dbm[0], pred_dbm[0], freq_idx, mid_t, plot_path)
            print(f"Map comparison saved to {plot_path}")

        sp_path = os.path.join(output_dir, "spatial_rmse_map.png")
        plot_spatial_rmse_map(spatial_rmse, sp_path)
        print(f"Spatial RMSE map saved to {sp_path}")

        freq_path = os.path.join(output_dir, "per_frequency_rmse.png")
        plot_per_frequency_rmse(rmse_per_freq, freq_path)
        print(f"Per-frequency RMSE saved to {freq_path}")
    else:
        errors = pred_dbm[:, :, 0, :, :] - target_dbm[:, :, 0, :, :]
        for n, name in enumerate(node_names):
            plot_path = os.path.join(output_dir, f"spectrogram_{name}.png")
            plot_spectrogram_comparison(
                target_dbm[0, :, 0, :, :], pred_dbm[0, :, 0, :, :],
                n, name, wcfg["input_sequence_length"], plot_path,
            )
            print(f"Spectrogram saved to {plot_path}")

        error_plot_path = os.path.join(output_dir, "error_analysis.png")
        plot_error_analysis(errors[0], node_names, error_plot_path)
        print(f"Error analysis saved to {error_plot_path}")


if __name__ == "__main__":
    main()
