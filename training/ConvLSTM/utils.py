import os
import random
import numpy as np
import torch


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
    b, t_out, c, h, w = pred.shape
    metrics = {}
    for t in range(t_out):
        m = compute_metrics(pred[:, t], target[:, t])
        for k, v in m.items():
            metrics[f"{k}_t{t+1}"] = v
    return metrics


def compute_metrics_per_node(pred, target, node_names=None):
    b, t_out, c, h, w = pred.shape
    if node_names is None:
        node_names = [f"node_{i}" for i in range(h)]
    metrics = {}
    for n in range(h):
        m = compute_metrics(pred[:, :, :, n:n+1, :], target[:, :, :, n:n+1, :])
        for k, v in m.items():
            metrics[f"{k}_{node_names[n]}"] = v
    return metrics


def save_checkpoint(path, model, optimizer, epoch, stats, config, metrics):
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
