import os, random, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str="auto"):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def denormalize(data, mean, std):
    return data * std + mean


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def r2_score(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def compute_metrics(pred, target):
    pred = pred.detach()
    target = target.detach()
    return {
        "rmse": rmse(pred, target).item(),
        "mae": mae(pred, target).item(),
        "r2": r2_score(pred, target).item(),
    }


def compute_metrics_per_horizon(pred, target):
    b, t_out, d = pred.shape
    metrics = {}
    for t in range(t_out):
        m = compute_metrics(pred[:, t], target[:, t])
        for k, v in m.items():
            metrics[f"{k}_t{t+1}"] = v
    return metrics


def compute_metrics_per_node(pred, target, n_nodes, bins_per_node, node_names=None):
    b, t_out, d = pred.shape
    if node_names is None:
        node_names = [f"Node{i}" for i in range(n_nodes)]
    metrics = {}
    for n in range(n_nodes):
        start = n * bins_per_node
        end = start + bins_per_node
        m = compute_metrics(pred[:, :, start:end], target[:, :, start:end])
        for k, v in m.items():
            metrics[f"{k}_{node_names[n]}"] = v
    return metrics


def compute_metrics_per_frequency(pred, target, n_nodes=1, bins_per_node=None):
    b, t_out, d = pred.shape
    if bins_per_node is None:
        bins_per_node = d
    metrics = {}
    for f in range(bins_per_node):
        idx = list(range(f, d, bins_per_node)) if n_nodes > 1 else [f]
        m = compute_metrics(pred[:, :, idx], target[:, :, idx])
        for k, v in m.items():
            metrics[f"{k}_freq{f}"] = v
    return metrics


def save_checkpoint(path, model, optimizer, epoch, stats, config, metrics=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "norm_stats": stats,
        "config": config,
        "metrics": metrics,
    }, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def plot_spectrogram_comparison(pred_dbm, true_dbm, node_name, output_path, max_time_steps=500):
    T_pred, F = pred_dbm.shape
    T_true, _ = true_dbm.shape
    T = min(T_pred, T_true, max_time_steps)
    pred_dbm = pred_dbm[:T, :]
    true_dbm = true_dbm[:T, :]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    vmin = min(pred_dbm.min(), true_dbm.min())
    vmax = max(pred_dbm.max(), true_dbm.max())

    im0 = axes[0].imshow(true_dbm.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                          interpolation="nearest", origin="lower")
    axes[0].set_title(f"{node_name} — Ground Truth (dBm)")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Frequency Bin")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_dbm.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                          interpolation="nearest", origin="lower")
    axes[1].set_title(f"{node_name} — Prediction (dBm)")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Frequency Bin")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_analysis(pred_dbm, true_dbm, output_path, max_time_steps=500):
    T_pred, F = pred_dbm.shape
    T_true, _ = true_dbm.shape
    T = min(T_pred, T_true, max_time_steps)
    error = np.abs(pred_dbm[:T] - true_dbm[:T])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(error.T, aspect="auto", cmap="Reds", interpolation="nearest", origin="lower")
    ax.set_title("Absolute Error (dBm)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Frequency Bin")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_json(metrics, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_csv(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, data, delimiter=",", fmt="%.6f")
