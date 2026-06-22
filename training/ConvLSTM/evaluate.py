import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import create_datasets
from model import ConvLSTMPredictor
from utils import (
    get_device, compute_metrics, compute_metrics_per_horizon,
    compute_metrics_per_node, load_checkpoint, denormalize,
)


def plot_spectrogram_comparison(ground_truth, prediction, node_idx, node_name,
                                 t_in, save_path):
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
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_analysis(errors, node_names, save_path):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = get_device("auto")

    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    stats = ckpt["norm_stats"]
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

    dcfg = config["data"]
    wcfg = config["windowing"]
    scfg = config["split"]

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

    all_pred, all_target = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_pred.append(pred)
            all_target.append(y)

    pred = torch.cat(all_pred, dim=0)
    target = torch.cat(all_target, dim=0)

    n_nodes = dcfg.get("n_nodes", 3)
    node_names = dcfg.get("node_names", None)
    if node_names is None:
        node_names = [f"Node_{i}" for i in range(n_nodes)]
        print(f"Warning: no node_names in config; using {node_names}")
    elif len(node_names) != n_nodes:
        print(f"Warning: config has {len(node_names)} names but n_nodes={n_nodes}; falling back to generic names")
        node_names = [f"Node_{i}" for i in range(n_nodes)]
    overall = compute_metrics(pred, target)
    per_horizon = compute_metrics_per_horizon(pred, target)
    per_node = compute_metrics_per_node(pred, target, node_names)

    print("=== Evaluation Report ===")
    print(f"Overall RMSE: {overall['rmse']:.4f}")
    print(f"Overall MAE:  {overall['mae']:.4f}")
    print(f"Overall R²:   {overall['r2']:.4f}")
    print()
    print("Per-horizon RMSE:")
    for h in args.horizons:
        key = f"rmse_t{h}"
        if key in per_horizon:
            print(f"  t={h}: {per_horizon[key]:.4f}")
    print()
    print("Per-node RMSE:")
    for name in node_names:
        key = f"rmse_{name}"
        if key in per_node:
            print(f"  {name}: {per_node[key]:.4f}")

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    mean = stats["mean"]
    std = stats["std"]
    if isinstance(mean, np.ndarray):
        pred_dbm = denormalize(pred_np, mean, std)
        target_dbm = denormalize(target_np, mean, std)
    else:
        pred_dbm, target_dbm = pred_np, target_np

    bs, t_out, c, h, w = pred_dbm.shape
    pred_flat = pred_dbm.reshape(bs, t_out, -1)
    target_flat = target_dbm.reshape(bs, t_out, -1)

    np.savetxt(os.path.join(output_dir, "predictions.csv"),
               pred_flat[0], delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_dir, "ground_truth.csv"),
               target_flat[0], delimiter=",", fmt="%.6f")

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({**overall, **per_horizon, **per_node}, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    errors = pred_np[:, :, 0, :, :] - target_np[:, :, 0, :, :]
    for n, name in enumerate(node_names):
        plot_path = os.path.join(output_dir, f"spectrogram_{name}.png")
        plot_spectrogram_comparison(
            target_np[0, :, 0, :, :], pred_np[0, :, 0, :, :],
            n, name, wcfg["input_sequence_length"], plot_path,
        )
        print(f"Spectrogram saved to {plot_path}")

    error_plot_path = os.path.join(output_dir, "error_analysis.png")
    plot_error_analysis(errors[0], node_names, error_plot_path)
    print(f"Error analysis saved to {error_plot_path}")


if __name__ == "__main__":
    main()
