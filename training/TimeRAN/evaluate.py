import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_datasets, load_csv
from utils import (
    compute_metrics,
    compute_metrics_per_bin,
    compute_metrics_per_horizon,
    compute_metrics_per_node,
    denormalize,
    get_device,
    load_checkpoint,
    set_seed,
)

from momentfm import MOMENTPipeline


VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


def load_model_from_checkpoint(ckpt_path: str, config: dict, device: torch.device):
    variant = config["model"]["checkpoint_size"].lower()
    model_name = VARIANT_TO_MODEL[variant]
    horizon = config["windowing"]["prediction_horizon"]

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": horizon,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
    )
    model.init()

    ckpt_data = load_checkpoint(ckpt_path, device)
    model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model, ckpt_data.get("norm_stats")


@torch.no_grad()
def evaluate(model, dataloader, device):
    all_preds = []
    all_targets = []
    for timeseries, forecast in tqdm(dataloader, desc="Evaluating"):
        timeseries = timeseries.to(device)
        forecast = forecast.to(device)
        input_mask = torch.ones(timeseries.shape[0], timeseries.shape[-1], device=device)

        with torch.cuda.amp.autocast():
            out = model(x_enc=timeseries, input_mask=input_mask)

        all_preds.append(out.forecast.cpu().numpy())
        all_targets.append(forecast.cpu().numpy())

    pred = np.concatenate(all_preds, axis=0)
    target = np.concatenate(all_targets, axis=0)
    return pred, target


def plot_spectrogram(
    pred: np.ndarray,
    target: np.ndarray,
    node_idx: int,
    node_name: str,
    bins_per_node: int,
    sample_idx: int,
    save_path: str,
):
    start = node_idx * bins_per_node
    end = start + bins_per_node
    pred_node = pred[sample_idx, start:end, :]
    target_node = target[sample_idx, start:end, :]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    vmin = min(target_node.min(), pred_node.min())
    vmax = max(target_node.max(), pred_node.max())

    im0 = axes[0].imshow(target_node, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{node_name} — Ground Truth")
    axes[0].set_xlabel("Future time step")
    axes[0].set_ylabel("Frequency bin")

    im1 = axes[1].imshow(pred_node, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{node_name} — Prediction")
    axes[1].set_xlabel("Future time step")
    axes[1].set_ylabel("Frequency bin")

    fig.colorbar(im1, ax=axes, label="Normalized PSD")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="training/TimeRAN/evaluation")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device("auto")
    print(f"Device: {device}")

    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt.get("config")
    if not config:
        raise ValueError("Checkpoint has no embedded config")
    norm_stats = ckpt.get("norm_stats")

    csv_path = config["data"]["dataset_path"]
    t_in = config["windowing"]["input_sequence_length"]
    t_out = config["windowing"]["prediction_horizon"]
    stride = config["windowing"]["stride"]
    train_ratio = config["split"]["train_ratio"]
    val_ratio = config["split"]["val_ratio"]
    normalization = config["preprocessing"]["normalization"]
    bins_per_node = config["data"]["bins_per_node"]
    node_names = config["data"]["node_names"]
    n_nodes = config["data"]["n_nodes"]

    _, _, test_ds, _ = create_datasets(
        csv_path=csv_path,
        t_in=t_in, t_out=t_out, stride=stride,
        train_ratio=train_ratio, val_ratio=val_ratio,
        normalization=normalization,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model, _ = load_model_from_checkpoint(args.checkpoint, config, device)

    pred, target = evaluate(model, test_loader, device)

    if norm_stats:
        mean = norm_stats["mean"]
        std = norm_stats["std"]
        pred = denormalize(pred, mean, std)
        target = denormalize(target, mean, std)

    overall = compute_metrics(pred, target)
    per_horizon = compute_metrics_per_horizon(pred, target)
    per_node = compute_metrics_per_node(pred, target, bins_per_node, node_names)
    rmse_per_bin, mae_per_bin = compute_metrics_per_bin(pred, target)

    results = {
        "overall": overall,
        "per_horizon": per_horizon,
        "per_node": per_node,
        "num_test_windows": len(test_ds),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    np.save(output_dir / "predictions.npy", pred)
    np.save(output_dir / "ground_truth.npy", target)

    B, C, H = pred.shape
    flat_pred = pred.transpose(0, 2, 1).reshape(B * H, C)
    flat_target = target.transpose(0, 2, 1).reshape(B * H, C)
    np.savetxt(output_dir / "predictions.csv", flat_pred, delimiter=",", fmt="%.6f")
    np.savetxt(output_dir / "ground_truth.csv", flat_target, delimiter=",", fmt="%.6f")

    for node_idx, node_name in enumerate(node_names):
        for si in range(min(3, B)):
            plot_spectrogram(
                pred, target, node_idx, node_name, bins_per_node, si,
                str(output_dir / f"spectrogram_{node_name}_sample{si}.png"),
            )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rmse_per_bin, label="RMSE per bin")
    ax.plot(mae_per_bin, label="MAE per bin")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Error")
    ax.legend()
    ax.set_title("Per-frequency-bin Error")
    plt.tight_layout()
    plt.savefig(output_dir / "error_per_bin.png", dpi=150)
    plt.close(fig)

    print(f"Evaluation results saved to {output_dir}")


if __name__ == "__main__":
    main()
