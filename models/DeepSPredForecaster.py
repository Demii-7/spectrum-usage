"""3D-SwinSTB (DeepSPred, Pan et al. 2025) as a forecaster in the shared POWDER pipeline.

The shared pipeline hands every 2D model the same windows: an input ``(B, L, F)`` of per-frequency z-scored dB values (training-split
statistics) and a target ``(B, h, F)``. This wrapper keeps DeepSPred's representation and loss but speaks that contract:

1. **Render** (fixed, not tuned): z is mapped to [0, 1] with one affine ``v = (z + k) / (2k)``, clipped, where ``k`` is the 99.99th
   percentile of the training split's |z| (see ``z_clip_from``), then through a colormap: ``jet`` (3 channels, the paper's) or
   ``gray`` (1 channel, its Appendix-A ablation).
2. **Frames**: the L input steps become ``T = input_frames`` frames of ``H = L / T`` rows (time) x F columns (frequency); rows are padded
   to a multiple of 4 and columns to ``w_pad`` (256) with zeros. The paper's projection head (eq. 22) emits ``K = T`` frames, i.e. the
   next L steps; horizon h is row h - 1 of that output (the first ``prediction_horizon`` rows are kept).
3. **Loss**: the paper's image-space MSE on the rendered target (``training_loss``; used by the shared trainer when a model defines it).
4. **Back-out**: ``forward`` inverts the predicted colours (nearest point of a 4096-entry colormap table; exact for gray) and undoes the
   affine, returning z-scored dB, which the pipeline denormalizes to dBm. Validation, model selection and every reported metric therefore
   use real dB errors, never the image score.
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DeepSPred import SwinSTB3D

LUT_SIZE = 4096
COLORMAPS = {"jet": 3, "gray": 1}


def z_clip_from(train_data: np.ndarray, quantile: float = 99.99) -> float:
    """Half-width k of the rendered z range: the 99.99th percentile of |z| on the training split (clips 0.01% of training values).

    Not the maximum: per-bin z-scores on POWDER reach |z| = 68-92 while 99.9% lie within 5-7, so a max-based range squeezes almost
    all data into ~5% of the colormap (the same failure as dataset min-max scaling when a few extreme pixels set the range)."""
    finite = np.asarray(train_data, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(max(1.0, np.percentile(np.abs(finite), quantile))) if finite.size else 4.0


class DeepSPredForecaster(nn.Module):
    def __init__(self, config: dict, train_data: np.ndarray | None = None):
        super().__init__()
        m = config["model"]
        self.input_sequence_length = L = int(m["input_sequence_length"])
        self.prediction_horizon = int(m["prediction_horizon"])
        self.n_bins = int(m["input_size"])
        self.colormap = str(m.get("colormap", "jet"))
        if self.colormap not in COLORMAPS:
            raise ValueError(f"colormap must be one of {sorted(COLORMAPS)}, got {self.colormap!r}")
        self.channels = COLORMAPS[self.colormap]
        self.input_frames = T = int(m.get("input_frames", 4))
        if L % T:
            raise ValueError(f"input_sequence_length {L} is not divisible by input_frames {T}")
        self.frame_rows = H = L // T
        if self.prediction_horizon > L:
            raise ValueError(f"prediction_horizon {self.prediction_horizon} exceeds the {L} rows the paper head emits (K = T frames)")
        self.rows_padded = int(math.ceil(H / 4) * 4)
        self.w_pad = int(m.get("w_pad", 256))
        if self.w_pad < self.n_bins or self.w_pad % 16:
            raise ValueError(f"w_pad {self.w_pad} must be >= {self.n_bins} bins and divisible by 16")

        embed_dim = int(m.get("embed_dim", 96))
        depths = list(m.get("depths", [2, 4, 2]))
        heads = list(m.get("num_heads", [4, 8, 16]))
        window = list(m.get("window_size", [2, 7, 7]))
        patch_t = 2 if T % 2 == 0 else 1                                 # the paper's temporal patch 2 needs an even frame count
        self.swin = SwinSTB3D({
            "model": {"embed_dim": embed_dim, "depths": depths, "num_heads": heads, "patch_size": [patch_t, 4, 4],
                      "window_size": [min(window[0], max(1, T // patch_t)), window[1], window[2]], "head": "paper",
                      "in_chans": self.channels, "out_chans": self.channels,
                      "drop_rate": float(m.get("drop_rate", 0.0)), "attn_drop_rate": float(m.get("attn_drop_rate", 0.0)),
                      "drop_path_rate": float(m.get("drop_path_rate", 0.0))},
            "frames": {"minutes_per_frame": self.rows_padded, "w_pad": self.w_pad, "w_orig": self.n_bins},
            "windowing": {"input_frames": T, "output_frames": T},
        })
        k = float(m["z_clip"]) if m.get("z_clip") is not None else (z_clip_from(train_data) if train_data is not None else 4.0)
        self.register_buffer("z_clip", torch.tensor(k, dtype=torch.float32))
        # resampled: matplotlib's registered colormaps are 256-entry tables, which would quantize z to 2k/255 steps
        cmap = matplotlib.colormaps[self.colormap].resampled(LUT_SIZE)
        lut = cmap(np.linspace(0.0, 1.0, LUT_SIZE))[:, : self.channels].astype(np.float32)
        self.register_buffer("lut", torch.from_numpy(lut), persistent=False)

    # ---- representation -------------------------------------------------------------------------------------------------------------
    def to_unit(self, z: torch.Tensor) -> torch.Tensor:
        return ((z + self.z_clip) / (2 * self.z_clip)).clamp(0.0, 1.0)

    def render(self, z: torch.Tensor) -> torch.Tensor:
        """(B, n, F) z-scores -> (B, n, C, F) colours in [0, 1] (linear interpolation in a 4096-entry table = the colormap itself)."""
        v = self.to_unit(z)
        if self.colormap == "gray":
            return v.unsqueeze(2)
        pos = v * (LUT_SIZE - 1)
        lo = pos.floor().long().clamp(max=LUT_SIZE - 2)
        frac = (pos - lo).unsqueeze(-1)
        rgb = self.lut[lo] * (1 - frac) + self.lut[lo + 1] * frac        # (B, n, F, 3)
        return rgb.permute(0, 1, 3, 2)

    def invert(self, img: torch.Tensor) -> torch.Tensor:
        """(B, n, C, F) colours -> (B, n, F) z-scores. Gray: exact. Jet: nearest table entry (step 2k/4095 in z)."""
        if self.colormap == "gray":
            v = img[:, :, 0]
        else:
            flat = img.permute(0, 1, 3, 2).reshape(-1, 3)
            idx = torch.empty(flat.shape[0], dtype=torch.long, device=img.device)
            for start in range(0, flat.shape[0], 32768):
                chunk = flat[start:start + 32768]
                idx[start:start + 32768] = torch.cdist(chunk, self.lut).argmin(dim=1)
            v = (idx.float() / (LUT_SIZE - 1)).view(img.shape[0], img.shape[1], img.shape[3])
        return v * (2 * self.z_clip) - self.z_clip

    def off_curve_distance(self, img: torch.Tensor) -> torch.Tensor:
        """Mean Euclidean distance of predicted colours from the colormap curve (0 for gray): a diagnostic of the jet back-out."""
        if self.colormap == "gray":
            return img.new_zeros(())
        flat = img.permute(0, 1, 3, 2).reshape(-1, 3)
        return torch.cdist(flat[:65536], self.lut).min(dim=1).values.mean()

    # ---- model ----------------------------------------------------------------------------------------------------------------------
    def predict_image(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, F) z-scores -> (B, L, C, F) predicted colours for the next L steps (K = T frames of H rows)."""
        B, L, Fb = x.shape
        T, H = self.input_frames, self.frame_rows
        img = self.render(x).view(B, T, H, self.channels, Fb).permute(0, 1, 3, 2, 4)          # (B, T, C, H, F)
        img = F.pad(img, (0, self.w_pad - Fb, 0, self.rows_padded - H))                       # pad F -> w_pad, H -> rows_padded
        out = self.swin(img)                                                                   # (B, T, C, rows_padded, F)
        out = out[:, :, :, :H]                                                                 # drop pad rows
        return out.permute(0, 1, 3, 2, 4).reshape(B, T * H, self.channels, Fb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.invert(self.predict_image(x)[:, : self.prediction_horizon])

    def training_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The paper's objective: MSE between predicted and rendered target frames (image space), on the target's rows only."""
        pred = self.predict_image(x)[:, : y.shape[1]]
        return F.mse_loss(pred, self.render(y))
