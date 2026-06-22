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

from dataset import create_datasets, collate_branch_samples
from stsprednet import STSPredNet
from utils import (
    get_device, compute_metrics, compute_metrics_per_node,
    compute_metrics_per_frequency, load_checkpoint, denormalize,
)


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def plot_spectrogram_comparison(ground_truth, prediction, node_idx, node_name,
                                 save_path):
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
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Frequency Bin")
    fig.colorbar(im1, ax=axes.ravel().tolist(), label="Normalized Power")
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
        vmax = max(abs(err.min()), abs(err.max()))
        im = ax.imshow(err.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
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
    csv_path = dcfg["dataset_path"]
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", csv_path)

    print("Loading data...")
    _, _, test_ds, _ = create_datasets(csv_path, config)
    if test_ds is None or len(test_ds) == 0:
        print("No test samples available. Check config.")
        return

    test_loader = DataLoader(
        test_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, collate_fn=collate_branch_samples,
    )

    model = STSPredNet(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_pred, all_target = [], []
    with torch.no_grad():
        for batch in test_loader:
            closeness = batch.get("closeness")
            period = batch.get("period")
            trend = batch.get("trend")
            target = batch["target"].to(device)

            if closeness is not None:
                closeness = closeness.to(device)
            if period is not None:
                period = period.to(device)
            if trend is not None:
                trend = trend.to(device)

            pred = model(closeness, period, trend)
            all_pred.append(pred.cpu())
            all_target.append(target.cpu())

    pred = torch.cat(all_pred, dim=0)
    target = torch.cat(all_target, dim=0)

    pred_np = pred.numpy()
    target_np = target.numpy()

    metrics = compute_metrics(pred, target)
    print("=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    node_names = dcfg.get("node_names", None)
    per_node = compute_metrics_per_node(pred, target, node_names)
    print("\nPer-Node Metrics:")
    for k, v in per_node.items():
        print(f"  {k}: {v:.4f}")

    per_freq = compute_metrics_per_frequency(pred, target)
    freq_rmse = [v for k, v in per_freq.items() if "rmse" in k]
    if freq_rmse:
        print(f"\nPer-Frequency RMSE: min={min(freq_rmse):.4f}  max={max(freq_rmse):.4f}")

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    B, C, H, W = pred.shape
    pred_flat = pred_np.reshape(B, -1)
    target_flat = target_np.reshape(B, -1)
    np.savetxt(os.path.join(output_dir, "predictions.csv"), pred_flat, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_dir, "ground_truth.csv"), target_flat, delimiter=",", fmt="%.6f")

    pred_dbm = denormalize(pred_np, stats)
    target_dbm = denormalize(target_np, stats)
    np.savetxt(os.path.join(output_dir, "predictions_dbm.csv"),
               pred_dbm.reshape(B, -1), delimiter=",", fmt="%.2f")
    np.savetxt(os.path.join(output_dir, "ground_truth_dbm.csv"),
               target_dbm.reshape(B, -1), delimiter=",", fmt="%.2f")

    if node_names:
        for n, name in enumerate(node_names):
            plot_spectrogram_comparison(
                target_dbm[:, 0], pred_dbm[:, 0], n, name,
                os.path.join(output_dir, f"spectrogram_{name}.png"),
            )

    errors = pred_dbm - target_dbm
    plot_error_analysis(errors[:, 0], node_names or [f"Node{i}" for i in range(H)],
                        os.path.join(output_dir, "error_analysis.png"))

    all_metrics = {
        "overall": metrics,
        "per_node": per_node,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
