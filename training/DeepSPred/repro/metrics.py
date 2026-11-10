"""Metrics in the paper's own space. No skimage / lpips / torchmetrics in the core image, so
SSIM is implemented here (Wang et al. 2004) and LPIPS is reported as unavailable rather than skipped.
DEEPSPRED_REFERENCE.md rows 4.8 and 4.9 record both as gaps the paper leaves open.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import gaussian_filter

SPACES = ('uint8_mean_per_channel', 'uint8_mean_channel_sum', 'unit_mean_per_channel', 'uint8_sum_literal')


def _as(space: str, a: np.ndarray) -> np.ndarray:
    """Return the array scaled for `space`. Inputs are uint8-valued floats in 0..255."""
    if space.startswith('unit'):
        return a / 255.0
    return a


def mse_in_space(pred: np.ndarray, true: np.ndarray, space: str) -> float:
    """pred/true: (n, H, W, 3) float arrays holding 0..255 values."""
    if space not in SPACES:
        raise ValueError(f'''unknown space {space!r}; expected one of {SPACES}''')
    se = (_as(space, pred.astype(np.float64)) - _as(space, true.astype(np.float64))) ** 2
    if space == 'uint8_mean_channel_sum':
        return float(se.sum(axis=-1).mean())
    if space == 'uint8_sum_literal':
        return float(se.mean(axis=-1).sum(axis=(-2, -1)).mean())
    return float(se.mean())


def psnr_from_mse(mse: float, max_i: float = 255.0) -> float:
    """Eq. (31). Returns inf for an exact match."""
    if mse <= 0:
        return float('inf')
    return 10.0 * math.log10(max_i ** 2 / mse)


def ssim(pred: np.ndarray, true: np.ndarray, max_i: float = 255.0, sigma: float = 1.5,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Mean SSIM over a batch of HxWx3 images, computed per channel then averaged.

    11x11 Gaussian window, sigma 1.5 - the Wang et al. defaults the paper's eq. (32) implies via
    rho1 = 0.01, rho2 = 0.03. The paper does not state window or border handling (gap 4.8).
    """
    c1, c2 = (k1 * max_i) ** 2, (k2 * max_i) ** 2
    out = []
    for p, t in zip(pred.astype(np.float64), true.astype(np.float64)):
        per_channel = []
        for ch in range(p.shape[-1]):
            x, y = p[..., ch], t[..., ch]
            mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
            mxx = gaussian_filter(x * x, sigma) - mx * mx
            myy = gaussian_filter(y * y, sigma) - my * my
            mxy = gaussian_filter(x * y, sigma) - mx * my
            s = (2 * mx * my + c1) * (2 * mxy + c2) / ((mx ** 2 + my ** 2 + c1) * (mxx + myy + c2))
            per_channel.append(float(s.mean()))
        out.append(sum(per_channel) / len(per_channel))
    return float(np.mean(out))


def lpips_available() -> tuple[bool, str]:
    try:
        import lpips
        return True, 'lpips package present'
    except Exception as exc:
        return False, (f'''lpips unavailable ({type(exc).__name__}); the core image ships neither lpips nor torchmetrics. LPIPS rows are recorded as null, never silently omitted. Install with: pip install lpips''')
