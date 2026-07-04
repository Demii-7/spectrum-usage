import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import (
    get_device, load_config, load_checkpoint, denormalize,
    compute_metrics, compute_metrics_per_node, compute_metrics_per_horizon,
    plot_spectrogram_comparison, plot_error_analysis,
)
from dataset import SpectrumMapDataset, load_and_split
from model import DSwinLSTM_I


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_pred = []
    all_target = []
    infer_time = 0.0
    for X, mask, Y in loader:
        X = X.permute(0, 1, 4, 2, 3).contiguous().to(device)
        mask = mask.to(device)
        Y = Y.permute(0, 1, 4, 2, 3).contiguous().to(device)
        batch_start = time.perf_counter()
        pred = model(X, mask)
        infer_time += time.perf_counter() - batch_start
        all_pred.append(pred.cpu())
        all_target.append(Y.cpu())
    return torch.cat(all_pred, dim=0), torch.cat(all_target, dim=0), infer_time


def save_outputs(output_dir, pred_dbm, target_dbm, stats, config, num_samples):
    B, T_out, C, H, W = pred_dbm.shape
    pred_rows = pred_dbm.reshape(B * T_out, -1)
    target_rows = target_dbm.reshape(B * T_out, -1)
    np.savetxt(os.path.join(output_dir, "predictions.csv"), pred_rows, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_dir, "ground_truth.csv"), target_rows, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(output_dir, "predictions_dbm.csv"), pred_rows, delimiter=",", fmt="%.2f")
    np.savetxt(os.path.join(output_dir, "ground_truth_dbm.csv"), target_rows, delimiter=",", fmt="%.2f")

    metadata = {
        "data_format": stats.get("data_format", config["data"].get("format", "csv")),
        "num_samples": num_samples,
        "prediction_horizon": T_out,
        "channels": C,
        "height": H,
        "width": W,
        "node_names": stats.get("node_names"),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


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
    stats["node_names"] = node_names
    config["model"]["map_height"] = int(stats["grid_height"])
    config["model"]["map_width"] = int(stats["grid_width"])
    config["model"]["input_channels"] = int(stats["n_freq_bins"])

    test_dataset = SpectrumMapDataset(test_norm, config, split="test")
    test_loader = DataLoader(test_dataset, batch_size=config["training"].get("batch_size", 4), shuffle=False)

    model = DSwinLSTM_I(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_start = time.perf_counter()
    pred_norm, target_norm, infer_time = evaluate_model(model, test_loader, device)
    total_eval_time = time.perf_counter() - eval_start

    pred_np = pred_norm.numpy()
    target_np = target_norm.numpy()
    pred_dbm = denormalize(pred_np, stats)
    target_dbm = denormalize(target_np, stats)

    overall = compute_metrics(pred_norm, target_norm)
    per_horizon = compute_metrics_per_horizon(pred_norm, target_norm)
    B, T_out, C, H, W = pred_norm.shape
    if stats.get("data_format") == "interpolated_map":
        per_node = {}
    else:
        per_node = compute_metrics_per_node(pred_norm.view(-1, C, H, W), target_norm.view(-1, C, H, W), node_names)
    metrics = {**overall, **per_horizon, **per_node}
    metrics["inference_time_seconds"] = infer_time
    metrics["total_evaluation_time_seconds"] = total_eval_time
    metrics["mean_sample_inference_time_seconds"] = infer_time / len(test_dataset)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if config["evaluation"].get("export_predictions", True):
        save_outputs(output_dir, pred_dbm, target_dbm, stats, config, len(test_dataset))

    if config["evaluation"].get("plot_denormalized_dbm", True):
        errors = pred_dbm - target_dbm
        if stats.get("data_format") != "interpolated_map":
            for n, name in enumerate(node_names):
                plot_spectrogram_comparison(target_dbm[0, :, 0], pred_dbm[0, :, 0], n, name, os.path.join(output_dir, f"spectrogram_{name}.png"))
            plot_error_analysis(errors[0, :, 0], node_names, os.path.join(output_dir, "error_analysis.png"))

    print("\nEvaluation Metrics:")
    for k, v in overall.items():
        print(f"  {k}: {v:.6f}")
    print(f"Inference time: {infer_time:.2f}s")
    print(f"Total evaluation time: {total_eval_time:.2f}s")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
