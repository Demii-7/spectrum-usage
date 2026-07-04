"""
Standalone inference script for a trained ConvLSTM spectrum-prediction model.

Loads a saved checkpoint, reads an arbitrary input CSV (which may be different
from the training data), normalizes it using the checkpoint's saved statistics,
and runs the model to produce predictions. Results are written to a CSV file.

This script is useful for deploying the model on new, unseen spectrum data
without requiring the full training/evaluation pipeline.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ConvLSTMPredictor
from utils import get_device, load_checkpoint, denormalize
from dataset import load_csv, reshape_to_3d, compute_norm_stats, zscore


def save_prediction_outputs(output_path, pred_dbm, config):
    """Save all inference windows plus metadata."""
    b, t_out, c, h, w = pred_dbm.shape
    pred_rows = pred_dbm.reshape(b * t_out, -1)
    np.savetxt(output_path, pred_rows, delimiter=",", fmt="%.6f")

    metadata = {
        "num_windows": b,
        "prediction_horizon": t_out,
        "channels": c,
        "height": h,
        "width": w,
        "data_format": config["data"].get("format", "csv"),
        "flatten_order": "rows are ordered by window-major then horizon-major; columns flatten (C, H, W)",
    }
    metadata_path = os.path.splitext(output_path)[0] + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def main():
    """
    Run standalone inference: load checkpoint → read & normalize input CSV → predict → save.

    The input CSV can be any spectrum data with the same node/bins layout as training.
    Predictions are denormalized back to dBm before saving.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to input CSV to predict on")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--t-in", type=int, default=None)
    parser.add_argument("--t-out", type=int, default=None)
    args = parser.parse_args()

    device = get_device("auto")
    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    stats = ckpt["norm_stats"]

    dcfg = config["data"]
    # Allow CLI overrides of input/output sequence lengths from the config.
    t_in = args.t_in or config["windowing"]["input_sequence_length"]
    t_out = args.t_out or config["windowing"]["prediction_horizon"]
    n_nodes = dcfg["n_nodes"]
    n_bins = dcfg["n_bins_per_node"]

    print(f"Loading input: {args.input}")
    raw = load_csv(args.input)
    data_3d = reshape_to_3d(raw, n_nodes, n_bins)

    mean = stats["mean"]
    std = stats["std"]
    # Normalize using stored statistics if available (z-score path).
    if isinstance(mean, np.ndarray):
        data_norm = zscore(data_3d, mean, std)
    else:
        # If stats contain scalars (identity normalization), skip normalization.
        data_norm = data_3d

    # Build sliding-window inputs. If the input is too short for even one window,
    # tile it once as a fallback; this handles edge cases like single-row inputs.
    total_windows = len(data_norm) - t_in - t_out + 1
    if total_windows < 1:
        # Duplicate the data along the time axis to guarantee at least one complete window.
        data_norm = np.tile(data_norm, (2, 1, 1))
        total_windows = len(data_norm) - t_in - t_out + 1
        if total_windows < 1:
            raise ValueError(f"Input too short ({len(data_3d)} steps). Need at least {t_in + t_out}.")

    windows = np.stack([data_norm[i:i + t_in] for i in range(total_windows)], axis=0)
    # Match training/evaluation layout: (batch, time, channels, height, width).
    windows = torch.from_numpy(windows).float().unsqueeze(2)
    loader = DataLoader(windows, batch_size=32, shuffle=False)

    model = ConvLSTMPredictor(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run batched inference without gradient tracking.
    all_pred = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            pred = model(x)
            all_pred.append(pred.cpu())

    pred = torch.cat(all_pred, dim=0)
    b, to, c, h, w = pred.shape

    if isinstance(mean, np.ndarray):
        pred_dbm = denormalize(pred.numpy(), mean, std)
    else:
        pred_dbm = pred.numpy()

    metadata_path = save_prediction_outputs(args.output, pred_dbm, config)
    print(f"Predictions saved to {args.output} ({b} windows × {to} time steps × {h * w} columns)")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
