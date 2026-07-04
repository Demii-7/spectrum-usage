"""
Evaluate a trained DeepSPred checkpoint on the test split.

Usage:
    python training/DeepSPred/evaluate.py \
        --checkpoint training/DeepSPred/smoke_test/checkpoints/best_model.pt \
        --config     training/DeepSPred/smoke_test/config.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from dataset import create_datasets
from model import SwinSTB3D
from utils import get_device, compute_metrics, compute_metrics_per_horizon, invert_colormap, load_checkpoint


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu())
        targets.append(y.cpu())
    return torch.cat(preds), torch.cat(targets)


def rgb_to_dbm(rgb_tensor, norm_stats, node_name, cmap_name="jet"):
    """
    Convert (B, T, 3, H, W) RGB predictions back to dBm values.
    Returns numpy array of same shape without the C=3 dimension.
    """
    stats = norm_stats[node_name]
    vmin, vmax = stats["vmin"], stats["vmax"]

    # (B, T, 3, H, W) → (B, T, H, W, 3) for colormap inversion
    rgb_np = rgb_tensor.permute(0, 1, 3, 4, 2).numpy().astype(np.float32)
    scalar = invert_colormap(rgb_np, cmap_name=cmap_name)   # (B, T, H, W)
    return scalar * (vmax - vmin) + vmin


def rgb_to_dbm_per_sample(rgb_tensor, norm_stats, node_names, cmap_name="jet"):
    dbm_samples = []
    for sample_idx, node_name in enumerate(node_names):
        sample_dbm = rgb_to_dbm(rgb_tensor[sample_idx:sample_idx + 1], norm_stats, node_name, cmap_name)
        dbm_samples.append(sample_dbm)
    return np.concatenate(dbm_samples, axis=0)


def save_tensor_csv(path, array, prefix, node_names=None):
    rows = []
    for sample_idx in range(array.shape[0]):
        for horizon_idx in range(array.shape[1]):
            row = {
                "sample_idx": sample_idx,
                "horizon_idx": horizon_idx,
            }
            if node_names is not None:
                row["node_name"] = node_names[sample_idx]
            flat = array[sample_idx, horizon_idx].reshape(-1)
            row.update({f"{prefix}_{i}": float(v) for i, v in enumerate(flat)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--csv",        default=None)
    parser.add_argument("--out-dir",    default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))

    ckpt = load_checkpoint(args.checkpoint, device)
    norm_stats = ckpt["norm_stats"]

    _, _, test_ds, _ = create_datasets(config, csv_path=args.csv)
    if test_ds is None or len(test_ds) == 0:
        print("No test samples.")
        return
    sample_node_names = test_ds.sample_nodes

    loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                        shuffle=False, num_workers=0)

    model = SwinSTB3D(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    pred, tgt = run_inference(model, loader, device)
    # pred/tgt: (N, T, 3, H, W_orig)

    metrics = {}

    # RGB-space metrics.
    metrics["rgb"] = compute_metrics(pred, tgt, include_perceptual=True)
    metrics["rgb_per_horizon"] = compute_metrics_per_horizon(pred, tgt, include_perceptual=True)

    # dBm-space metrics using the correct per-sample normalization stats.
    node_names = list(config["data"]["nodes"].keys())
    cmap = config["preprocessing"]["colormap"]
    metrics["dbm"] = {}
    pred_dbm_all = rgb_to_dbm_per_sample(pred, norm_stats, sample_node_names, cmap)
    tgt_dbm_all = rgb_to_dbm_per_sample(tgt, norm_stats, sample_node_names, cmap)
    for node in node_names:
        try:
            node_indices = [i for i, sample_node in enumerate(sample_node_names) if sample_node == node]
            if not node_indices:
                continue
            pred_t = torch.from_numpy(pred_dbm_all[node_indices])
            tgt_t = torch.from_numpy(tgt_dbm_all[node_indices])
            metrics["dbm"][node] = compute_metrics(pred_t, tgt_t)
        except Exception as e:
            metrics["dbm"][node] = {"error": str(e)}

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "..", "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    pred_np = pred.numpy().astype(np.float32)
    tgt_np = tgt.numpy().astype(np.float32)
    np.save(os.path.join(out_dir, "predictions.npy"), pred_np)
    np.save(os.path.join(out_dir, "targets.npy"), tgt_np)
    save_tensor_csv(os.path.join(out_dir, "predictions_rgb.csv"), pred_np, "rgb", sample_node_names)
    save_tensor_csv(os.path.join(out_dir, "targets_rgb.csv"), tgt_np, "rgb", sample_node_names)

    np.save(os.path.join(out_dir, "predictions_dbm.npy"), pred_dbm_all.astype(np.float32))
    np.save(os.path.join(out_dir, "targets_dbm.npy"), tgt_dbm_all.astype(np.float32))
    save_tensor_csv(os.path.join(out_dir, "predictions_dbm.csv"), pred_dbm_all.astype(np.float32), "dbm", sample_node_names)
    save_tensor_csv(os.path.join(out_dir, "targets_dbm.csv"), tgt_dbm_all.astype(np.float32), "dbm", sample_node_names)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== RGB metrics ===")
    for k, v in metrics["rgb"].items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=== dBm metrics ===")
    for node, m in metrics["dbm"].items():
        print(f"  {node}:", {k: (f"{v:.4f}" if isinstance(v, (int, float)) else v) for k, v in m.items() if k != "error"})
    print(f"Saved metrics: {metrics_path}")

    # Plot a sample prediction vs target (first sample, first time step, first node row).
    _plot_sample(pred, tgt, out_dir)


def _plot_sample(pred, tgt, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    # (B, T, 3, H, W) → take first sample, first time step, permute to (H, W, 3)
    p = pred[0, 0].permute(1, 2, 0).numpy().clip(0, 1)
    t = tgt[0, 0].permute(1, 2, 0).numpy().clip(0, 1)
    axes[0].imshow(t, aspect="auto", origin="lower"); axes[0].set_title("Target")
    axes[1].imshow(p, aspect="auto", origin="lower"); axes[1].set_title("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_prediction.png"), dpi=100)
    plt.close()
    print(f"Saved plot: {out_dir}/sample_prediction.png")


if __name__ == "__main__":
    main()
