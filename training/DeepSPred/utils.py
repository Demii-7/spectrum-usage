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


def compute_metrics(pred, target):
    pred = pred.detach().float()
    target = target.detach().float()
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_metrics_per_horizon(pred, target):
    # pred/target: (B, T_out, C, H, W)
    metrics = {}
    for t in range(pred.shape[1]):
        m = compute_metrics(pred[:, t], target[:, t])
        for k, v in m.items():
            metrics[f"{k}_t{t+1}"] = v
    return metrics


def save_checkpoint(path, model, optimizer, epoch, norm_stats, config, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "norm_stats": norm_stats,
        "config": config,
        "metrics": metrics,
    }, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def invert_colormap(rgb_np, cmap_name="jet", n_lut=1024):
    """
    Approximate inverse of a matplotlib colormap.
    rgb_np: (..., 3) float32 in [0,1]
    Returns: (...,) float32 in [0,1]
    """
    import matplotlib
    cmap = matplotlib.colormaps[cmap_name]
    scalars = np.linspace(0, 1, n_lut)
    lut = cmap(scalars)[:, :3].astype(np.float32)          # (n_lut, 3)
    flat = rgb_np.reshape(-1, 3)                             # (N, 3)
    diffs = flat[:, None, :] - lut[None, :, :]              # (N, n_lut, 3)
    dists = np.sum(diffs ** 2, axis=-1)                     # (N, n_lut)
    idx = np.argmin(dists, axis=-1)                         # (N,)
    return scalars[idx].astype(np.float32).reshape(rgb_np.shape[:-1])
