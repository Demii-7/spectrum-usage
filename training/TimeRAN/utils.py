import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto"):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def compute_metrics(pred: np.ndarray, target: np.ndarray):
    return {
        "rmse": compute_rmse(pred, target),
        "mae": compute_mae(pred, target),
    }


def compute_metrics_per_horizon(pred: np.ndarray, target: np.ndarray):
    B, C, H = pred.shape
    metrics = {}
    for h in range(H):
        rmse = compute_rmse(pred[:, :, h], target[:, :, h])
        mae = compute_mae(pred[:, :, h], target[:, :, h])
        metrics[f"t={h+1}"] = {"rmse": rmse, "mae": mae}
    return metrics


def compute_metrics_per_node(
    pred: np.ndarray, target: np.ndarray, bins_per_node: int, node_names: list[str]
):
    B, C, H = pred.shape
    n_nodes = len(node_names)
    metrics = {}
    for i, name in enumerate(node_names):
        start = i * bins_per_node
        end = start + bins_per_node
        rmse = compute_rmse(pred[:, start:end, :], target[:, start:end, :])
        mae = compute_mae(pred[:, start:end, :], target[:, start:end, :])
        metrics[name] = {"rmse": rmse, "mae": mae}
    return metrics


def compute_metrics_per_bin(pred: np.ndarray, target: np.ndarray):
    B, C, H = pred.shape
    rmse_per_bin = np.sqrt(np.mean((pred - target) ** 2, axis=(0, 2)))
    mae_per_bin = np.mean(np.abs(pred - target), axis=(0, 2))
    return rmse_per_bin, mae_per_bin


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: dict,
    norm_stats: dict | None = None,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        "norm_stats": norm_stats,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return data * std + mean
