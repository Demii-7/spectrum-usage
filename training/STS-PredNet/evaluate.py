"""
Evaluation script for a trained STS-PredNet checkpoint.

Generates predictions on the test set, computes metrics (RMSE, MAE, R²),
produces per-node and per-frequency breakdowns, saves CSV outputs in
denormalized dBm units, and creates visualisations (spectrograms, error maps).
"""
import copy
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
    """Load YAML configuration file from disk."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def plot_spectrogram_comparison(ground_truth, prediction, node_idx, node_name,
                                 save_path):
    """Plot side-by-side spectrograms of ground truth and prediction for one node.

    Args:
        ground_truth: Array of shape (time, nodes, frequencies).
        prediction: Array of same shape as ground_truth.
        node_idx: Index of the node to plot.
        node_name: Human-readable name for the node.
        save_path: Destination file path for the PNG.
    """
    gt_node = ground_truth[:, node_idx, :].T
    pred_node = prediction[:, node_idx, :].T
    # Use a shared color range so the two plots are directly comparable
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
    """Plot prediction error heatmaps for every node.

    Args:
        errors: Array of (prediction - ground_truth) of shape (samples, nodes, freqs).
        node_names: List of node label strings.
        save_path: Destination PNG path.
    """
    n = len(node_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False,
                              constrained_layout=True)
    im = None
    for i, name in enumerate(node_names):
        ax = axes[0, i]
        err = errors[:, i, :]
        # Symmetric color scale around zero to highlight over-/under-prediction
        vmax = max(abs(err.min()), abs(err.max()))
        im = ax.imshow(err.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name} — Error (Pred − GT)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Frequency Bin")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Normalized Error")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    """Load a trained checkpoint, run evaluation on the test set, and save results."""
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

    prediction_horizon = config.get("windowing", {}).get("prediction_horizon", 1)
    is_multistep = prediction_horizon > 1

    eval_config = copy.deepcopy(config)
    if is_multistep:
        # For multi-step evaluation we shift prediction_offset so the dataset
        # yields the correct alignment for autoregressive-style sampling
        eval_config["branches"]["prediction_offset"] = prediction_horizon

    print("Loading data...")
    _, _, test_ds, _ = create_datasets(csv_path, eval_config)
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
            target_idx = batch.get("target_idx")
            t_tensor = batch.get("t")

            if closeness is not None:
                closeness = closeness.to(device)
            if period is not None:
                period = period.to(device)
            if trend is not None:
                trend = trend.to(device)

            if is_multistep and t_tensor is not None:
                # Multi-step: iteratively predict one step at a time and gather
                # targets from the pre-loaded dataset tensor
                bs = closeness.shape[0] if closeness is not None else period.shape[0]
                bs_preds, bs_targets = [], []
                for o in range(1, prediction_horizon + 1):
                    actual_idx = t_tensor + o
                    target_o = test_ds.data[actual_idx].unsqueeze(1)
                    pred_o = model(closeness, period, trend)
                    bs_preds.append(pred_o.cpu())
                    bs_targets.append(target_o.cpu())
                all_pred.append(torch.stack(bs_preds, dim=1))
                all_target.append(torch.stack(bs_targets, dim=1))
            else:
                # Single-step prediction
                target = batch["target"].to(device)
                pred = model(closeness, period, trend)
                all_pred.append(pred.cpu().unsqueeze(1))
                all_target.append(target.cpu().unsqueeze(1))

    pred = torch.cat(all_pred, dim=0)
    target = torch.cat(all_target, dim=0)

    pred_np = pred.numpy()
    target_np = target.numpy()

    B, T, C, H, W = pred.shape
    # Flatten time and channels for overall metric computation
    pred_2d = pred.reshape(B * T, C * H * W)
    target_2d = target.reshape(B * T, C * H * W)

    flat_metrics = compute_metrics(pred_2d, target_2d)
    print("=== Test Set Metrics ===")
    for k, v in flat_metrics.items():
        print(f"  {k}: {v:.4f}")

    node_names = dcfg.get("node_names", None)
    per_node = compute_metrics_per_node(pred.reshape(B * T, C, H, W), target.reshape(B * T, C, H, W), node_names)
    print("\nPer-Node Metrics:")
    for k, v in per_node.items():
        print(f"  {k}: {v:.4f}")

    per_freq = compute_metrics_per_frequency(pred.reshape(B * T, C, H, W), target.reshape(B * T, C, H, W))
    freq_rmse = [v for k, v in per_freq.items() if "rmse" in k]
    if freq_rmse:
        print(f"\nPer-Frequency RMSE: min={min(freq_rmse):.4f}  max={max(freq_rmse):.4f}")

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # Convert back from normalized space to physical dBm units
    pred_dbm = denormalize(pred_np, stats)
    target_dbm = denormalize(target_np, stats)
    pred_dbm_2d = pred_dbm.reshape(B, T, -1)
    target_dbm_2d = target_dbm.reshape(B, T, -1)

    np.savetxt(os.path.join(output_dir, "predictions.csv"),
               pred_dbm_2d[0], delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_dir, "ground_truth.csv"),
               target_dbm_2d[0], delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_dir, "predictions_dbm.csv"),
               pred_dbm_2d.reshape(-1, pred_dbm_2d.shape[-1]), delimiter=",", fmt="%.2f")
    np.savetxt(os.path.join(output_dir, "ground_truth_dbm.csv"),
               target_dbm_2d.reshape(-1, target_dbm_2d.shape[-1]), delimiter=",", fmt="%.2f")

    if node_names:
        for n, name in enumerate(node_names):
            plot_spectrogram_comparison(
                target_dbm[0, :, 0], pred_dbm[0, :, 0], n, name,
                os.path.join(output_dir, f"spectrogram_{name}.png"),
            )

    errors = pred_dbm - target_dbm
    plot_error_analysis(errors[0, :, 0], node_names or [f"Node{i}" for i in range(H)],
                        os.path.join(output_dir, "error_analysis.png"))

    all_metrics = {
        "overall": flat_metrics,
        "per_node": per_node,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
