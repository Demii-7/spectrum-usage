import os, sys, argparse, json, time
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import create_datasets
from utils import (
    get_device,
    compute_metrics,
    compute_metrics_per_horizon,
    compute_metrics_per_node,
    compute_metrics_per_frequency,
    load_checkpoint,
    denormalize,
    save_metrics_json,
    save_csv,
    plot_spectrogram_comparison,
    plot_error_analysis,
)
from train import build_model, run_model, resolve_feature_dims


def save_metadata(output_dir, stats, config, num_samples):
    metadata = {
        "data_format": stats.get("data_format", config["data"].get("format", "csv")),
        "num_samples": num_samples,
        "prediction_horizon": config["windowing"]["pred_len"],
        "n_features": stats["n_features"],
    }
    for key in ("n_nodes", "bins_per_node", "selected_nodes", "grid_height", "grid_width", "n_grid"):
        if key in stats:
            metadata[key] = stats[key]
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
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
    data_format = data_cfg.get("format", "csv")
    dataset_path = data_cfg["map_path"] if data_format == "interpolated_map" else data_cfg["dataset_path"]

    output_dir = args.output_dir or os.path.abspath(paths["evaluation_dir"])
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    norm_stats = checkpoint["norm_stats"]

    print(f"Loading data from {dataset_path}")
    _, _, test_ds, stats = create_datasets(
        data_format=data_format,
        dataset_path=dataset_path,
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
        data_cfg=data_cfg,
    )
    stats = norm_stats | {k: v for k, v in stats.items() if k not in norm_stats}

    if test_ds is None or len(test_ds) == 0:
        print("No test samples found.")
        return

    resolve_feature_dims(config, stats)
    batch_size = config["training"]["batch_size"]
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Test samples: {len(test_ds)}")

    model, model_cfg = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_start = time.perf_counter()
    inference_time = 0.0
    all_pred, all_true = [], []
    with torch.no_grad():
        for seq_x, seq_y in test_loader:
            seq_x, seq_y = seq_x.to(device), seq_y.to(device)
            batch_start = time.perf_counter()
            output = run_model(model, model_cfg, seq_x, seq_y, device)
            inference_time += time.perf_counter() - batch_start
            all_pred.append(output.cpu())
            all_true.append(seq_y[:, -model_cfg.pred_len:, :].cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    total_eval_time = time.perf_counter() - eval_start

    mean_t = torch.from_numpy(stats["mean"]).float()
    std_t = torch.from_numpy(stats["std"]).float()
    pred_denorm = denormalize(all_pred, mean_t, std_t).numpy()
    true_denorm = denormalize(all_true, mean_t, std_t).numpy()

    overall = compute_metrics(all_pred, all_true)
    per_horizon = compute_metrics_per_horizon(all_pred, all_true)
    metrics = {"overall": overall, "per_horizon": per_horizon}

    if data_format == "csv":
        n_nodes = stats["n_nodes"]
        bins_per_node = stats["bins_per_node"]
        node_names = stats.get("selected_nodes") or data_cfg.get("node_names")
        metrics["per_node"] = compute_metrics_per_node(all_pred, all_true, n_nodes, bins_per_node, node_names)
        metrics["per_frequency"] = compute_metrics_per_frequency(all_pred, all_true, n_nodes=n_nodes, bins_per_node=bins_per_node)
    else:
        metrics["per_node"] = {}
        metrics["per_frequency"] = compute_metrics_per_frequency(all_pred, all_true, n_nodes=1, bins_per_node=stats["n_features"])

    eval_horizons = eval_cfg.get("eval_horizons", [])
    for h in eval_horizons:
        if h <= all_pred.shape[1]:
            m = compute_metrics(all_pred[:, h - 1], all_true[:, h - 1])
            metrics[f"h{h}"] = m

    metrics["timing"] = {
        "inference_time_seconds": inference_time,
        "total_evaluation_time_seconds": total_eval_time,
        "mean_batch_inference_time_seconds": inference_time / len(test_loader),
        "mean_sample_inference_time_seconds": inference_time / len(test_ds),
    }
    save_metrics_json(metrics, os.path.join(output_dir, "metrics.json"))

    pred_flat = pred_denorm.reshape(-1, pred_denorm.shape[-1])
    true_flat = true_denorm.reshape(-1, true_denorm.shape[-1])
    save_csv(pred_flat, os.path.join(output_dir, "predictions.csv"))
    save_csv(true_flat, os.path.join(output_dir, "ground_truth.csv"))
    save_metadata(output_dir, stats, config, len(test_ds))

    if eval_cfg.get("plot_denormalized_dbm", True):
        if data_format == "csv":
            n_nodes = stats["n_nodes"]
            bins_per_node = stats["bins_per_node"]
            node_names = stats.get("selected_nodes") or data_cfg.get("node_names")
            pred_3d = pred_denorm.reshape(-1, n_nodes, bins_per_node)
            true_3d = true_denorm.reshape(-1, n_nodes, bins_per_node)
            for n in range(n_nodes):
                node_name = node_names[n] if n < len(node_names) else f"Node{n}"
                plot_spectrogram_comparison(pred_3d[:, n, :], true_3d[:, n, :], node_name, os.path.join(output_dir, f"spectrogram_{node_name}.png"))
            plot_error_analysis(pred_3d.reshape(-1, bins_per_node * n_nodes), true_3d.reshape(-1, bins_per_node * n_nodes), os.path.join(output_dir, "error_analysis.png"))
        else:
            plot_error_analysis(pred_denorm.reshape(-1, pred_denorm.shape[-1]), true_denorm.reshape(-1, true_denorm.shape[-1]), os.path.join(output_dir, "error_analysis.png"))

    print(f"Evaluation results saved to {output_dir}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Total evaluation time: {total_eval_time:.2f}s")


if __name__ == "__main__":
    main()
