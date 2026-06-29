"""
Dataset pipeline for DeepSPred.

Each node's dBm power data is independently:
  1. Normalized to [0,1] using train-set min/max
  2. Converted to RGB via a matplotlib colormap (jet by default)
  3. Grouped into H-minute spectrogram frames
  4. Width-padded from 250 to 256 for the model
  5. Split chronologically into train/val/test
  6. Windowed into (x, y) pairs of T_in/T_out consecutive frames

Multiple nodes are pooled as independent samples (windows never straddle nodes).

Sample shapes:
  x: (T_in, 3, H, W_pad)   — padded input, channel-first
  y: (T_in, 3, H, W_orig)  — unpadded target (same length as x; T_out == T_in)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_node(csv_path, col_start, col_end):
    """Load one node's columns from CSV. Returns (T, W) float32."""
    # Use numpy for speed; assume no header row.
    data = np.loadtxt(csv_path, delimiter=",", usecols=range(col_start, col_end),
                      dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return data


def _minmax_stats(data):
    return float(data.min()), float(data.max())


def _normalize(data, vmin, vmax):
    return np.clip((data - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0).astype(np.float32)


def _colormap(data_01, cmap_name):
    """(T, W) float in [0,1] → (T, W, 3) float in [0,1] via colormap."""
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(data_01)                        # (T, W, 4) float64
    return rgba[:, :, :3].astype(np.float32)   # drop alpha


def _make_frames(rgb, H):
    """(T, W, 3) → (N_frames, H, W, 3). Trims tail so T is divisible by H."""
    T, W, C = rgb.shape
    n = T // H
    return rgb[: n * H].reshape(n, H, W, C)


def _pad_w(frames, w_pad):
    """(N, H, W, 3) → (N, H, w_pad, 3) by zero-padding on the right."""
    N, H, W, C = frames.shape
    if W == w_pad:
        return frames
    pad = np.zeros((N, H, w_pad - W, C), dtype=np.float32)
    return np.concatenate([frames, pad], axis=2)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SpectrumFrameDataset(Dataset):
    """
    Sliding-window dataset over spectrogram frames.

    Indices are pre-validated so that windows never straddle segment boundaries
    (i.e., they never mix different nodes or different split partitions).
    """

    def __init__(self, frames_padded, frames_orig, t_in, indices):
        # Convert to (N, 3, H, W) once — channel first.
        self.pad  = torch.from_numpy(frames_padded.transpose(0, 3, 1, 2))  # (N, 3, H, W_pad)
        self.orig = torch.from_numpy(frames_orig.transpose(0, 3, 1, 2))    # (N, 3, H, W_orig)
        self.t_in = t_in
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.pad [i : i + self.t_in]                      # (T_in, 3, H, W_pad)
        y = self.orig[i + self.t_in : i + 2 * self.t_in]     # (T_in, 3, H, W_orig)
        return x, y


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_datasets(config, csv_path=None):
    """
    Build train/val/test SpectrumFrameDatasets from config.

    Returns:
        train_ds, val_ds, test_ds  — may be None if not enough frames
        stats                      — dict of per-node {"vmin", "vmax"}
    """
    dcfg = config["data"]
    pcfg = config["preprocessing"]
    fcfg = config["frames"]
    wcfg = config["windowing"]
    scfg = config["split"]

    if csv_path is None:
        csv_path = dcfg["dataset_path"]

    # Resolve relative path from repo root (two levels above this file).
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", csv_path)

    t_in             = wcfg["input_frames"]
    stride           = wcfg.get("stride", 1)
    H                = fcfg["minutes_per_frame"]
    w_pad            = fcfg["w_pad"]
    cmap_name        = pcfg["colormap"]
    train_ratio      = scfg["train_ratio"]
    val_ratio        = scfg["val_ratio"]

    nodes = dcfg["nodes"]
    stats = {}

    # Accumulate per-split frame arrays for all nodes.
    tr_pad, tr_orig = [], []
    va_pad, va_orig = [], []
    te_pad, te_orig = [], []

    for node_name, ncfg in nodes.items():
        col_start = ncfg["col_start"]
        col_end   = ncfg["col_end"]

        raw = _load_node(csv_path, col_start, col_end)   # (T, W)
        T, W = raw.shape

        # Fit normalization on training rows only.
        n_tr_rows = int(T * train_ratio)
        vmin, vmax = _minmax_stats(raw[:n_tr_rows])
        stats[node_name] = {"vmin": vmin, "vmax": vmax}

        # Full pipeline.
        norm  = _normalize(raw, vmin, vmax)               # (T, W)
        rgb   = _colormap(norm, cmap_name)                # (T, W, 3)
        frames = _make_frames(rgb, H)                     # (N, H, W, 3)
        fp    = _pad_w(frames, w_pad)                     # (N, H, w_pad, 3)
        fo    = frames                                     # (N, H, W, 3)  — unpadded target

        N = len(frames)
        n_tr = int(N * train_ratio)
        n_va = int(N * val_ratio)

        tr_pad.append(fp[:n_tr])
        tr_orig.append(fo[:n_tr])
        va_pad.append(fp[n_tr : n_tr + n_va])
        va_orig.append(fo[n_tr : n_tr + n_va])
        te_pad.append(fp[n_tr + n_va :])
        te_orig.append(fo[n_tr + n_va :])

    def _build_ds(pad_list, orig_list):
        """Stack segments, build per-segment window indices (no cross-node windows)."""
        if not any(len(p) >= 2 * t_in for p in pad_list):
            return None
        pad  = np.concatenate(pad_list,  axis=0)
        orig = np.concatenate(orig_list, axis=0)
        indices = []
        offset = 0
        for p in pad_list:
            n = len(p)
            # Window requires t_in input frames + t_in output frames.
            valid = list(range(0, n - 2 * t_in + 1, stride))
            indices.extend(i + offset for i in valid)
            offset += n
        if not indices:
            return None
        return SpectrumFrameDataset(pad, orig, t_in, indices)

    train_ds = _build_ds(tr_pad, tr_orig)
    val_ds   = _build_ds(va_pad, va_orig)
    test_ds  = _build_ds(te_pad, te_orig)

    return train_ds, val_ds, test_ds, stats
