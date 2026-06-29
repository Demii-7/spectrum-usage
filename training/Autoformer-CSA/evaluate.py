"""
Evaluation script for a trained Autoformer-CSA / AutoformerVanilla model.

Loads a saved checkpoint, runs inference on the test set, computes RMSE/MAE/R2
metrics overall, per horizon, and per node, saves numeric results as JSON/CSV,
and generates diagnostic spectrogram comparison and error-analysis plots.
"""

import os, sys, argparse, json
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import create_datasets
from utils import (
    get_device, compute_metrics, compute_metrics_per_horizon,
    compute_metrics_per_node, load_checkpoint,
    denormalize, save_metrics_json, save_csv,
    plot_spectrogram_comparison, plot_error_analysis,
)
from train import build_model


def main():
    """Entry point: load checkpoint, run test inference, compute metrics, save results.

    The checkpoint must contain ``model_state_dict`` and ``norm_stats``
    (mean/std for denormalisation).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config.get("device", {}).get("device", "auto"))
    print(f"Device: {device}")

    windowing = config["windowing"]
    data_cfg = config["data"]
    split_cfg = config["split"]
    preproc_cfg = config["preprocessing"]
    eval_cfg = config["evaluation"]
    paths = config["paths"]

    output_dir = args.output_dir or os.path.abspath(paths["evaluation_dir"])
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    norm_stats = checkpoint["norm_stats"]

    print(f"Loading data from {data_cfg['dataset_path']}")
    # Only the test split is needed for evaluation
    _, _, test_ds, _ = create_datasets(
        csv_path=data_cfg["dataset_path"],
        seq_len=windowing["seq_len"],
        label_len=windowing["label_len"],
        pred_len=windowing["pred_len"],
        train_stride=windowing["train_stride"],
        val_stride=windowing.get("val_stride"),
        test_stride=windowing.get("test_stride"),
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        chronological=split_cfg.get("chronological_split", True),
        normalization=preproc_cfg.get("normalization", "zscore"),
        fit_on_train_only=preproc_cfg.get("fit_on_train_only", True),
    )

    if test_ds is None or len(test_ds) == 0:
        print("No test samples found.")
        return

    batch_size = config["training"]["batch_size"]
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Test samples: {len(test_ds)}")

    model, model_cfg = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for seq_x, seq_y in test_loader:
            seq_x, seq_y = seq_x.to(device), seq_y.to(device)
            x_mark_enc = torch.zeros(seq_x.shape[0], seq_x.shape[1], 4, device=device)
            x_mark_dec = torch.zeros(seq_y.shape[0], seq_y.shape[1], 4, device=device)
            dec_input = seq_y[:, :model_cfg.label_len + model_cfg.pred_len, :]
            output = model(seq_x, x_mark_enc, dec_input, x_mark_dec)
            all_pred.append(output.cpu())
            all_true.append(seq_y[:, -model_cfg.pred_len:, :].cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)

    # Denormalise using the saved stats (computed during training)
    mean_t = torch.from_numpy(norm_stats["mean"]).float()
    std_t = torch.from_numpy(norm_stats["std"]).float()
    pred_dbm = denormalize(all_pred, mean_t, std_t).numpy()
    true_dbm = denormalize(all_true, mean_t, std_t).numpy()

    n_nodes = len(data_cfg.get("node_names", ["CC2"]))
    bins_per_node = data_cfg.get("bins_per_node", data_cfg["n_features"] // n_nodes)
    node_names = data_cfg.get("node_names", [f"Node{i}" for i in range(n_nodes)])

    overall = compute_metrics(all_pred, all_true)
    per_horizon = compute_metrics_per_horizon(all_pred, all_true)
    per_node = compute_metrics_per_node(all_pred, all_true, n_nodes, bins_per_node, node_names)

    metrics = {}
    metrics.update({f"overall_{k}": v for k, v in overall.items()})
    metrics.update(per_horizon)
    metrics.update(per_node)

    eval_horizons = eval_cfg.get("eval_horizons", [1, 3, 6, 12])
    for h in eval_horizons:
        if h <= all_pred.shape[1]:
            h_idx = h - 1
            m = compute_metrics(all_pred[:, h_idx], all_true[:, h_idx])
            for k, v in m.items():
                metrics[f"h{h}_{k}"] = v

    save_metrics_json(metrics, os.path.join(output_dir, "metrics.json"))

    B, T_out, D = pred_dbm.shape
    pred_flat = pred_dbm.reshape(-1, D)
    true_flat = true_dbm.reshape(-1, D)
    save_csv(pred_flat, os.path.join(output_dir, "predictions.csv"))
    save_csv(true_flat, os.path.join(output_dir, "ground_truth.csv"))

    pred_3d = pred_dbm.reshape(B * T_out, n_nodes, bins_per_node)
    true_3d = true_dbm.reshape(B * T_out, n_nodes, bins_per_node)

    for n in range(n_nodes):
        node_name = node_names[n] if n < len(node_names) else f"Node{n}"
        plot_spectrogram_comparison(
            pred_3d[:, n, :], true_3d[:, n, :], node_name,
            os.path.join(output_dir, f"spectrogram_{node_name}.png"),
        )

    all_nodes_pred = pred_3d.reshape(-1, bins_per_node * n_nodes)
    all_nodes_true = true_3d.reshape(-1, bins_per_node * n_nodes)
    plot_error_analysis(all_nodes_pred, all_nodes_true, os.path.join(output_dir, "error_analysis.png"))

    print(f"Evaluation results saved to {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
