"""Evaluation script for trained TimeRAN models on the test set.

Computes overall, per-horizon, and per-node metrics; saves predictions,
ground truth, spectrogram comparison plots, and an error analysis heatmap.
Supports denormalization back to dBm when training used z-score normalization.
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_datasets
from utils import (
    compute_metrics,
    compute_metrics_per_horizon,
    compute_metrics_per_node,
    compute_metrics_per_bin,
    denormalize,
    get_device,
    load_checkpoint,
    set_seed,
)

from momentfm import MOMENTPipeline


# Mapping from user-facing size labels to HuggingFace Hub model identifiers.
VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


def build_model_from_config(config: dict, device: torch.device):
    """Construct a frozen MOMENT model for inference from a config dict.

    The encoder and embedder are frozen (evaluation only); weights are
    overwritten later by ``load_state_dict`` from the checkpoint.

    Args:
        config: Configuration dictionary with model and windowing keys.
        device: Target torch device.

    Returns:
        A ``torch.nn.Module`` in eval mode.
    """
    variant = config["model"]["checkpoint_size"].lower()
    model_name = VARIANT_TO_MODEL[variant]
    horizon = config["windowing"]["prediction_horizon"]
    t_in = config["windowing"]["input_sequence_length"]

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": horizon,
            "seq_len": t_in,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
    )
    model.init()
    model = model.to(device)
    model.eval()
    return model


def evaluate_checkpoint(checkpoint_path: str, config: dict, test_loader, device: torch.device, bins_per_node: int, node_names: list[str], output_dir: Path | None = None):
    ckpt = load_checkpoint(checkpoint_path, device)
    model = build_model_from_config(config, device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"Missing keys while loading {checkpoint_path}: {missing}")
    if unexpected:
        print(f"Unexpected keys while loading {checkpoint_path}: {unexpected}")
    pred_norm, target_norm, inference_time = evaluate(model, test_loader, device)
    overall = compute_metrics(pred_norm, target_norm)
    per_horizon = compute_metrics_per_horizon(pred_norm, target_norm)
    per_node = compute_metrics_per_node(pred_norm, target_norm, bins_per_node, node_names)
    rmse_per_bin, mae_per_bin = compute_metrics_per_bin(pred_norm, target_norm)
    return {
        "checkpoint": checkpoint_path,
        "overall": overall,
        "per_horizon": per_horizon,
        "per_node": per_node,
        "timing": {
            "inference_time_seconds": inference_time,
            "mean_batch_inference_time_seconds": inference_time / len(test_loader) if len(test_loader) > 0 else 0.0,
            "mean_window_inference_time_seconds": inference_time / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0.0,
        },
        "per_frequency": {
            **{f"rmse_freq{i}": float(v) for i, v in enumerate(rmse_per_bin)},
            **{f"mae_freq{i}": float(v) for i, v in enumerate(mae_per_bin)},
        },
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "norm_stats": ckpt.get("norm_stats"),
    }


def plot_spectrogram_comparison(ground_truth, prediction, node_name, save_path):
    """Plot ground-truth and predicted spectrograms side-by-side for one node.

    Uses a shared colour scale to make visual comparison easier.

    Args:
        ground_truth: Array of shape (time, freq_bins).
        prediction: Array of shape (time, freq_bins).
        node_name: Label for the plot title.
        save_path: Path to save the PNG figure.
    """
    gt_node = ground_truth.T
    pred_node = prediction.T
    # Shared colour limits so the two subplots are directly comparable.
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
    """Plot per-node error heatmaps (prediction minus ground truth).

    Each column shows the error for one node across all samples and frequency
    bins.  The colour scale is clamped to [-3, 3] to highlight moderate errors.

    Args:
        errors: Array of shape (samples, n_nodes, freq_bins).
        node_names: List of node labels.
        save_path: Path to save the PNG figure.
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


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Run the model over a dataloader and collect all predictions/targets.

    Args:
        model: The forecasting model in eval mode.
        dataloader: DataLoader yielding (input, target) pairs.
        device: Target torch device.

    Returns:
        Tuple of (predictions, targets), each a NumPy array of shape
        (N, C, H) where N is the number of windows, C is channels,
        and H is the forecast horizon.
    """
    all_preds = []
    all_targets = []
    inference_time = 0.0
    for timeseries, forecast in tqdm(dataloader, desc="Evaluating"):
        batch_start = time.perf_counter()
        timeseries = timeseries.to(device)
        forecast = forecast.to(device)
        input_mask = torch.ones(timeseries.shape[0], timeseries.shape[-1], device=device)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(x_enc=timeseries, input_mask=input_mask)
        else:
            out = model(x_enc=timeseries, input_mask=input_mask)

        all_preds.append(out.forecast.cpu().numpy())
        all_targets.append(forecast.cpu().numpy())
        inference_time += time.perf_counter() - batch_start

    pred = np.concatenate(all_preds, axis=0)
    target = np.concatenate(all_targets, axis=0)
    return pred, target, inference_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--output", default=None)
    parser.add_argument("--compare-checkpoint", default=None)
    args = parser.parse_args()

    device = get_device("auto")
    print(f"Device: {device}")

    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt.get("config")
    if not config:
        raise ValueError("Checkpoint has no embedded config")
    norm_stats = ckpt.get("norm_stats")

    # Allow overriding the config via a separate YAML (often used in smoke tests).
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config["preprocessing"]["normalization"] = config["preprocessing"].get("normalization", "revin_only")

    dcfg = config["data"]
    wcfg = config["windowing"]
    scfg = config["split"]

    csv_path = dcfg["dataset_path"]
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", csv_path)

    _, _, test_ds, _ = create_datasets(
        csv_path=csv_path,
        t_in=wcfg["input_sequence_length"],
        t_out=wcfg["prediction_horizon"],
        stride=wcfg.get("stride", 1),
        train_stride=wcfg.get("train_stride"),
        val_stride=wcfg.get("val_stride"),
        test_stride=wcfg.get("test_stride"),
        train_ratio=scfg["train_ratio"],
        val_ratio=scfg["val_ratio"],
        normalization=config["preprocessing"]["normalization"],
    )

    if test_ds is None or len(test_ds) == 0:
        print("No test set available.")
        return

    # Use batch_size=1 so the first sample can be directly plotted.
    batch_size = config["training"].get("batch_size", 1)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    n_nodes = dcfg.get("n_nodes", 1)
    bins_per_node = dcfg.get("bins_per_node", 250)
    node_names = dcfg.get("node_names", [f"Node_{i}" for i in range(n_nodes)])

    horizons = args.horizons or config.get("evaluation", {}).get("eval_horizons", [1, 3, 6])
    eval_start = time.perf_counter()
    primary = evaluate_checkpoint(args.checkpoint, config, test_loader, device, bins_per_node, node_names)
    pred_norm = primary.pop("pred_norm")
    target_norm = primary.pop("target_norm")
    norm_stats = primary.pop("norm_stats")
    inference_time = primary["timing"]["inference_time_seconds"]
    total_evaluation_time = time.perf_counter() - eval_start

    overall = primary["overall"]
    per_horizon = primary["per_horizon"]
    per_node = primary["per_node"]

    print("=== Evaluation Report ===")
    print(f"Overall RMSE: {overall['rmse']:.4f}")
    print(f"Overall MAE:  {overall['mae']:.4f}")
    print(f"Overall R\u00b2:   {overall['r2']:.4f}")
    print()
    print("Per-horizon RMSE:")
    for h in horizons:
        key = f"rmse_t{h}"
        if key in per_horizon:
            print(f"  t={h}: {per_horizon[key]:.4f}")
    print()
    print("Per-node RMSE:")
    for name in node_names:
        key = f"rmse_{name}"
        if key in per_node:
            print(f"  {name}: {per_node[key]:.4f}")

    output_dir = Path(args.output or os.path.join(os.path.dirname(__file__), "evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Denormalise back to physical dBm units for saved CSV and plots.
    if norm_stats:
        mean = norm_stats["mean"]
        std = norm_stats["std"]
        pred_dbm = denormalize(pred_norm, mean, std)
        target_dbm = denormalize(target_norm, mean, std)
    else:
        pred_dbm, target_dbm = pred_norm, target_norm

    pred_rows = pred_dbm.transpose(0, 2, 1).reshape(-1, pred_dbm.shape[1])
    target_rows = target_dbm.transpose(0, 2, 1).reshape(-1, target_dbm.shape[1])
    np.savetxt(output_dir / "predictions.csv", pred_rows, delimiter=",", fmt="%.6f")
    np.savetxt(output_dir / "ground_truth.csv", target_rows, delimiter=",", fmt="%.6f")

    rmse_per_bin, mae_per_bin = compute_metrics_per_bin(pred_norm, target_norm)

    metadata = {
        "num_windows": int(pred_norm.shape[0]),
        "prediction_horizon": int(pred_norm.shape[2]),
        "n_features": int(pred_norm.shape[1]),
        "n_nodes": int(n_nodes),
        "bins_per_node": int(bins_per_node),
        "node_names": node_names,
        "row_order": "rows are window-major then horizon-major with columns in node-frequency order",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        metrics = {**overall, **per_horizon, **per_node}
        metrics["timing"] = {
            "inference_time_seconds": inference_time,
            "total_evaluation_time_seconds": total_evaluation_time,
            "mean_batch_inference_time_seconds": inference_time / len(test_loader) if len(test_loader) > 0 else 0.0,
            "mean_window_inference_time_seconds": inference_time / len(test_ds) if len(test_ds) > 0 else 0.0,
        }
        metrics["per_frequency"] = primary["per_frequency"]
        if args.compare_checkpoint:
            comparison = evaluate_checkpoint(args.compare_checkpoint, config, test_loader, device, bins_per_node, node_names)
            metrics["comparison"] = {
                "baseline_checkpoint": args.compare_checkpoint,
                "baseline_overall": comparison["overall"],
                "delta_rmse": overall["rmse"] - comparison["overall"]["rmse"],
                "delta_r2": overall["r2"] - comparison["overall"]["r2"],
            }
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Reshape to (batch, horizon, node, freq_bin) for per-node plotting.
    B, C, H = pred_dbm.shape
    pred_3d = pred_dbm.transpose(0, 2, 1).reshape(B, H, n_nodes, bins_per_node)
    target_3d = target_dbm.transpose(0, 2, 1).reshape(B, H, n_nodes, bins_per_node)

    errors = pred_3d - target_3d
    for n, name in enumerate(node_names):
        plot_path = output_dir / f"spectrogram_{name}.png"
        plot_spectrogram_comparison(
            target_3d[0, :, n, :], pred_3d[0, :, n, :],
            name, plot_path,
        )
        print(f"Spectrogram saved to {plot_path}")

    error_plot_path = output_dir / "error_analysis.png"
    plot_error_analysis(errors[0], node_names, error_plot_path)
    print(f"Error analysis saved to {error_plot_path}")

    print(f"Inference time: {inference_time:.2f}s")
    print(f"Total evaluation time: {total_evaluation_time:.2f}s")

    print(f"\nEvaluation results saved to {output_dir}")


if __name__ == "__main__":
    main()
