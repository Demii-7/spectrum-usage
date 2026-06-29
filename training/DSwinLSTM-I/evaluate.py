"""
Evaluation script for the DSwinLSTM-I spectrum prediction model.

Loads a trained checkpoint, runs inference on the test set, computes
metrics (RMSE, MAE, R2, NRMSE), and generates visualizations including
spectrogram comparisons and error analysis plots.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import (
    get_device, load_config, load_checkpoint, denormalize,
    compute_metrics, compute_metrics_per_node, compute_metrics_per_horizon,
    plot_spectrogram_comparison, plot_error_analysis,
)
from dataset import AERPAWDataset, load_and_split
from model import DSwinLSTM_I


@torch.no_grad()
def evaluate(model, loader, device):
    """Run the model in evaluation mode over a DataLoader and collect predictions.

    Args:
        model: Trained DSwinLSTM-I model.
        loader: DataLoader for evaluation.
        device: Torch device.

    Returns:
        Tuple of (predictions, targets) tensors with shape (B, T_out, C, H, W).
    """
    model.eval()
    all_pred = []
    all_target = []
    for X, mask, Y in loader:
        X = X.permute(0, 1, 4, 2, 3).contiguous().to(device)
        mask = mask.to(device)
        Y = Y.permute(0, 1, 4, 2, 3).contiguous().to(device)
        pred = model(X, mask)
        all_pred.append(pred.cpu())
        all_target.append(Y.cpu())
    return torch.cat(all_pred, dim=0), torch.cat(all_target, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", default=None, help="Path to config YAML (overrides checkpoint config)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--cc2-only", action="store_true", help="CC2-only mode")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, "cpu")
    config = ckpt["config"]
    stats = ckpt["norm_stats"]

    if args.config:
        config = load_config(args.config)

    device = get_device(config.get("device", {}).get("device", "auto"))

    cc2_only = args.cc2_only or config["data"].get("cc2_only_smoke_test", False)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output or os.path.join(script_dir, config["paths"]["evaluation_dir"])
    os.makedirs(output_dir, exist_ok=True)

    train_norm, val_norm, test_norm, _, node_names = load_and_split(config, cc2_only=cc2_only)
    test_dataset = AERPAWDataset(test_norm, config, split="test")
    test_loader = DataLoader(test_dataset, batch_size=config["training"].get("batch_size", 4), shuffle=False)

    model = DSwinLSTM_I(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    pred_norm, target_norm = evaluate(model, test_loader, device)

    B, T_out, C, H, W = pred_norm.shape
    target_np = target_norm.numpy()
    pred_np = pred_norm.numpy()

    target_flat = target_np.reshape(B, T_out, -1)
    pred_flat = pred_np.reshape(B, T_out, -1)

    # Denormalize from [-1,1] back to original dBm scale for interpretability
    target_dbm = denormalize(target_np, stats)
    pred_dbm = denormalize(pred_np, stats)
    target_dbm_flat = target_dbm.reshape(B, T_out, -1)
    pred_dbm_flat = pred_dbm.reshape(B, T_out, -1)

    pred_tensor = torch.from_numpy(pred_np)
    target_tensor = torch.from_numpy(target_np)

    overall = compute_metrics(pred_tensor, target_tensor)
    per_horizon = compute_metrics_per_horizon(pred_tensor, target_tensor)
    per_node = compute_metrics_per_node(
        pred_tensor.view(-1, C, H, W),
        target_tensor.view(-1, C, H, W),
        node_names)

    metrics = {**overall, **per_horizon, **per_node}

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if config["evaluation"].get("export_predictions", True):
        np.savetxt(os.path.join(output_dir, "predictions.csv"),
                   pred_dbm_flat[0], delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(output_dir, "ground_truth.csv"),
                   target_dbm_flat[0], delimiter=",", fmt="%.6f")

        np.savetxt(os.path.join(output_dir, "predictions_dbm.csv"),
                   pred_dbm_flat.reshape(B, -1), delimiter=",", fmt="%.2f")
        np.savetxt(os.path.join(output_dir, "ground_truth_dbm.csv"),
                   target_dbm_flat.reshape(B, -1), delimiter=",", fmt="%.2f")

    plot_dbm = config["evaluation"].get("plot_denormalized_dbm", True)
    if plot_dbm:
        errors = pred_dbm - target_dbm
        for n, name in enumerate(node_names):
            path = os.path.join(output_dir, f"spectrogram_{name}.png")
            plot_spectrogram_comparison(target_dbm[0, :, 0], pred_dbm[0, :, 0], n, name, path)

        plot_error_analysis(errors[0, :, 0], node_names,
                            os.path.join(output_dir, "error_analysis.png"))

    print("\nEvaluation Metrics:")
    for k, v in overall.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  metrics.json")
    for n in node_names:
        print(f"  spectrogram_{n}.png")
    print(f"  error_analysis.png")
    print(f"  predictions.csv / ground_truth.csv")


if __name__ == "__main__":
    main()
