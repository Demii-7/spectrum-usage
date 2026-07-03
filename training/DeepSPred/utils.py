import os
import random
import numpy as np
import torch
import torch.nn.functional as F


try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None


_LPIPS_MODEL = None


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


def _compute_psnr(mse):
    return float((-10.0 * torch.log10(mse + 1e-8)).item())


def _compute_ssim(pred, target):
    # Global SSIM over each frame, averaged across batch and horizon.
    pred = pred.float()
    target = target.float()
    dims = tuple(range(2, pred.ndim))
    mu_x = pred.mean(dim=dims, keepdim=True)
    mu_y = target.mean(dim=dims, keepdim=True)
    sigma_x = ((pred - mu_x) ** 2).mean(dim=dims, keepdim=True)
    sigma_y = ((target - mu_y) ** 2).mean(dim=dims, keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=dims, keepdim=True)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return float(ssim_map.mean())


def _get_lpips_model(device):
    global _LPIPS_MODEL
    if lpips_lib is None:
        return None
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips_lib.LPIPS(net="alex")
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL.to(device)


@torch.no_grad()
def _compute_lpips(pred, target):
    model = _get_lpips_model(pred.device)
    if model is None:
        return None
    if pred.ndim == 5:
        pred = pred.reshape(-1, pred.shape[2], pred.shape[3], pred.shape[4])
        target = target.reshape(-1, target.shape[2], target.shape[3], target.shape[4])
    pred = pred.float() * 2.0 - 1.0
    target = target.float() * 2.0 - 1.0
    # LPIPS expects at least ~32x32 spatial support; upsample narrow maps if needed.
    if pred.shape[-2] < 32 or pred.shape[-1] < 32:
        size = (max(32, pred.shape[-2]), max(32, pred.shape[-1]))
        pred = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
    return float(model(pred, target).mean().item())


def compute_metrics(pred, target, include_perceptual=False):
    pred = pred.detach().float()
    target = target.detach().float()
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    metrics = {
        "mse": float(mse.item()),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "psnr": _compute_psnr(mse),
        "ssim": _compute_ssim(pred, target),
    }
    if include_perceptual:
        lpips_value = _compute_lpips(pred, target)
        if lpips_value is None:
            metrics["lpips"] = None
            metrics["lpips_status"] = "lpips package not installed"
        else:
            metrics["lpips"] = lpips_value
    return metrics


def compute_metrics_per_horizon(pred, target, include_perceptual=False):
    # pred/target: (B, T_out, C, H, W)
    metrics = {}
    for t in range(pred.shape[1]):
        m = compute_metrics(pred[:, t], target[:, t], include_perceptual=include_perceptual)
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
