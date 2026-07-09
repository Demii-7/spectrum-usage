"""Evaluation script for a trained STS-PredNet checkpoint.

Runs final test evaluation only. The underlying model is a direct single-step
predictor; multi-horizon benchmarking is implemented by rebuilding the test set
with different ``prediction_offset`` values and evaluating each horizon
independently.
"""
import copy
import os
import json
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import create_datasets, create_interpolated_map_datasets, collate_branch_samples
from stsprednet import STSPredNet
from utils import (
    get_device,
    compute_metrics,
    compute_metrics_per_node,
    compute_metrics_per_frequency,
    load_checkpoint,
    denormalize,
)


def plot_spectrogram_comparison(ground_truth, prediction, node_idx, node_name, save_path):
    """Plot side-by-side spectrograms of ground truth and prediction for one node."""
    gt_node = ground_truth[:, node_idx, :].T
    pred_node = prediction[:, node_idx, :].T
    vmin = min(gt_node.min(), pred_node.min())
    vmax = max(gt_node.max(), pred_node.max())

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
    axes[0].imshow(gt_node, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{node_name} - Ground Truth")
    axes[0].set_ylabel("Frequency Bin")
    im1 = axes[1].imshow(pred_node, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{node_name} - Prediction")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Frequency Bin")
    fig.colorbar(im1, ax=axes.ravel().tolist(), label="Power (dBm)")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_analysis(errors, node_names, save_path):
    """Plot prediction error heatmaps for every node."""
    n = len(node_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False, constrained_layout=True)
    im = None
    for i, name in enumerate(node_names):
        ax = axes[0, i]
        err = errors[i, :]
        vmax = max(abs(err.min()), abs(err.max()))
        im = ax.imshow(err.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name} - Error (Pred - GT)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Frequency Bin")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="dBm Error")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def load_test_dataset(config, horizon):
    """Build the test dataset for a specific direct-prediction horizon."""
    eval_config = copy.deepcopy(config)
    eval_config["branches"]["prediction_offset"] = int(horizon)
    dcfg = eval_config["data"]
    data_format = dcfg.get("format", "csv")

    if data_format == "interpolated_map":
        map_path = dcfg["map_path"]
        if not os.path.exists(map_path):
            map_path = os.path.join(os.path.dirname(__file__), "..", "..", map_path)
        _, _, test_ds, _ = create_interpolated_map_datasets(map_path, eval_config)
    else:
        csv_path = dcfg["dataset_path"]
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "..", csv_path)
        _, _, test_ds, _ = create_datasets(csv_path, eval_config)

    return test_ds, eval_config


def prepare_model_config(config, stats):
    """Restore architecture fields needed to instantiate the trained model."""
    model_config = copy.deepcopy(config)
    if model_config["data"].get("format", "csv") == "interpolated_map":
        model_config["model"]["input_channels"] = int(stats["n_freq"])
        model_config["model"]["map_height"] = int(stats["grid_h"])
        model_config["model"]["map_width"] = int(stats["grid_w"])
    return model_config


def run_single_horizon(model, loader, device):
    """Run direct single-step evaluation for one horizon."""
    all_pred, all_target = [], []
    with torch.no_grad():
        for batch in loader:
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
    return pred, target


def save_full_outputs(output_dir, horizon_records, data_format, config):
    """Save all horizon predictions and targets with metadata."""
    prediction_rows = []
    target_rows = []
    row_index = []

    for record in horizon_records:
        horizon = record["horizon"]
        pred_dbm = record["pred_dbm"]
        target_dbm = record["target_dbm"]
        num_samples = pred_dbm.shape[0]
        prediction_rows.append(pred_dbm.reshape(num_samples, -1))
        target_rows.append(target_dbm.reshape(num_samples, -1))
        for sample_idx in range(num_samples):
            row_index.append({"horizon": horizon, "sample_index": sample_idx})

    np.savetxt(os.path.join(output_dir, "predictions.csv"), np.concatenate(prediction_rows, axis=0), delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_dir, "ground_truth.csv"), np.concatenate(target_rows, axis=0), delimiter=",", fmt="%.6f")

    example = horizon_records[0]["pred_dbm"]
    metadata = {
        "data_format": data_format,
        "row_order": "rows are grouped by requested horizon order, then sample index",
        "row_index": row_index,
        "channels": int(example.shape[1]),
        "height": int(example.shape[2]),
        "width": int(example.shape[3]),
    }
    if data_format != "interpolated_map":
        metadata["node_names"] = config["data"].get("node_names")
        metadata["column_layout"] = "channel-major over node x frequency-bin"
    else:
        metadata["column_layout"] = "channel-major over frequency x grid_height x grid_width"

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--horizons", type=int, nargs="+")
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

    data_format = config["data"].get("format", "csv")
    horizons = args.horizons or config.get("evaluation", {}).get("eval_horizons", [config["branches"].get("prediction_offset", 1)])
    output_dir = args.output or os.path.join(os.path.dirname(__file__), "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    model_config = prepare_model_config(config, stats)
    model = STSPredNet(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metrics_by_horizon = {}
    horizon_records = []
    evaluation_start_time = time.perf_counter()
    total_inference_time_seconds = 0.0
    for horizon in horizons:
        print(f"Loading data for horizon t+{horizon}...")
        test_ds, eval_config = load_test_dataset(config, horizon)
        if test_ds is None or len(test_ds) == 0:
            print(f"No test samples available for horizon t+{horizon}. Skipping.")
            continue

        loader = DataLoader(
            test_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_branch_samples,
        )
        horizon_start_time = time.perf_counter()
        pred, target = run_single_horizon(model, loader, device)
        horizon_inference_time = time.perf_counter() - horizon_start_time
        total_inference_time_seconds += horizon_inference_time
        pred_np = pred.numpy()
        target_np = target.numpy()
        pred_dbm = denormalize(pred_np, stats)
        target_dbm = denormalize(target_np, stats)

        overall = compute_metrics(pred, target)
        horizon_key = f"t+{horizon}"
        horizon_metrics = {"overall": overall}
        horizon_metrics["timing"] = {
            "inference_time_seconds": horizon_inference_time,
            "mean_sample_inference_time_seconds": horizon_inference_time / len(test_ds),
        }

        if data_format != "interpolated_map":
            node_names = config["data"].get("node_names")
            if node_names is None:
                node_names = [f"Node{i}" for i in range(pred.shape[2])]
            per_node = compute_metrics_per_node(pred, target, node_names)
            horizon_metrics["per_node"] = per_node

            for node_idx, name in enumerate(node_names):
                plot_spectrogram_comparison(
                    target_dbm[:, 0],
                    pred_dbm[:, 0],
                    node_idx,
                    name,
                    os.path.join(output_dir, f"spectrogram_{name}_{horizon_key}.png"),
                )

            errors = pred_dbm[:, 0] - target_dbm[:, 0]
            plot_error_analysis(
                np.transpose(errors, (1, 0, 2)),
                node_names,
                os.path.join(output_dir, f"error_analysis_{horizon_key}.png"),
            )
        else:
            horizon_metrics["per_node"] = {}

        per_frequency = compute_metrics_per_frequency(pred, target)
        horizon_metrics["per_frequency"] = per_frequency

        metrics_by_horizon[horizon_key] = horizon_metrics
        horizon_records.append({
            "horizon": horizon,
            "pred_dbm": pred_dbm,
            "target_dbm": target_dbm,
        })

        print(f"=== Horizon {horizon_key} ===")
        for key, value in overall.items():
            print(f"  {key}: {value:.4f}")

    if not horizon_records:
        print("No evaluation outputs were generated.")
        return

    save_full_outputs(output_dir, horizon_records, data_format, config)
    total_evaluation_time_seconds = time.perf_counter() - evaluation_start_time
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({
            "summary": {
                "total_inference_time_seconds": total_inference_time_seconds,
                "total_evaluation_time_seconds": total_evaluation_time_seconds,
                "evaluated_horizons": horizons,
            },
            "per_horizon": metrics_by_horizon,
        }, f, indent=2)

    print(f"Total inference time: {total_inference_time_seconds:.2f}s")
    print(f"Total evaluation time: {total_evaluation_time_seconds:.2f}s")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
