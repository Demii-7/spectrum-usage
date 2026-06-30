"""
Dataset loading, normalization, and PyTorch Dataset creation for spectrum data.

This module handles:
- Reading raw CSV spectrum traces (CSV mode)
- Loading pre-interpolated .npz map data (interpolated_map mode)
- Computing normalization statistics (z-score or min-max)
- Creating train/val/test PyTorch Datasets with sliding-window sequences

Data shape conventions:

CSV mode:
- Raw CSV: (total_time_steps, n_nodes * n_bins)
- Reshaped 3D: (total_time_steps, n_nodes, n_bins)
- Dataset sample x: (t_in, 1, n_nodes, n_bins)  -- model input sequence
- Dataset sample y: (t_out, 1, n_nodes, n_bins) -- target sequence

Interpolated-map mode:
- Raw NPZ: (T, H, W, F)  -- time, grid_y, grid_x, freq/channel
- After transpose: (T, F, H, W) -- time, channel, grid_y, grid_x
- Dataset sample x: (t_in, F, H, W)
- Dataset sample y: (t_out, F, H, W)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    """PyTorch Dataset that yields (input, target) sliding-window pairs from 3D spectrum data.

    Each sample consists of ``t_in`` consecutive time steps as input and the following
    ``t_out`` consecutive time steps as the prediction target. The data tensor has shape
    (T, C=1, H=n_nodes, W=n_bins) after the unsqueeze/transpose transformation.
    """

    def __init__(self, data_3d, t_in, t_out, start_indices):
        """
        Args:
            data_3d: Normalized spectrum data, shape (T, n_nodes, n_bins).
            t_in: Number of input time steps per sample.
            t_out: Number of target (prediction) time steps per sample.
            start_indices: List of time indices marking the start of each window.
        """
        self.data = torch.from_numpy(data_3d).float()
        self.t_in = t_in
        self.t_out = t_out
        self.indices = start_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            x: Input sequence, shape (t_in, 1, n_nodes, n_bins).
            y: Target sequence, shape (t_out, 1, n_nodes, n_bins).
        """
        i = self.indices[idx]
        # Slice input window: [i : i + t_in] and target window immediately following.
        x = self.data[i : i + self.t_in]
        y = self.data[i + self.t_in : i + self.t_in + self.t_out]
        # Unsqueeze adds a channel dimension at dim 0, then transpose(0,1) moves
        # it to dim 1, yielding (T, C=1, n_nodes, n_bins) as expected by the model.
        x = x.unsqueeze(0).transpose(0, 1)
        y = y.unsqueeze(0).transpose(0, 1)
        return x, y


class InterpolatedMapDataset(Dataset):
    """PyTorch Dataset for pre-interpolated spectrogram maps stored as NPZ archives.

    The NPZ file contains a 4D array with shape (T, H, W, F) where:
    - T: number of time steps (minutes)
    - H: spatial grid height
    - W: spatial grid width
    - F: number of frequency channels

    After loading and transposing, the internal shape is (T, F, H, W).
    Each sample yields:
        x: (t_in, F, H, W)  -- input sequence
        y: (t_out, F, H, W) -- target sequence
    """

    def __init__(self, data_4d, t_in, t_out, start_indices):
        """
        Args:
            data_4d: Normalized map data, shape (T, F, H, W).
            t_in: Number of input time steps per sample.
            t_out: Number of target time steps per sample.
            start_indices: List of time indices marking the start of each window.
        """
        self.data = torch.from_numpy(data_4d).float()
        self.t_in = t_in
        self.t_out = t_out
        self.indices = start_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i : i + self.t_in]
        y = self.data[i + self.t_in : i + self.t_in + self.t_out]
        return x, y


def load_map_npz(npz_path, map_key):
    """Load an interpolated map from a .npz file.

    Expected raw shape: (T, H, W, F).
    Returns array transposed to (T, F, H, W) so that F maps to the channel dimension.
    """
    data = np.load(npz_path)[map_key].astype(np.float32)
    return data.transpose(0, 3, 1, 2)


def load_csv(csv_path):
    """
    Load a CSV file of spectrum measurements as a float32 NumPy array.

    Expected format: comma-delimited, one time step per row, each column
    corresponds to a flattened (node, frequency_bin) pair.
    """
    return np.loadtxt(csv_path, delimiter=",").astype(np.float32)


def reshape_to_3d(arr, n_nodes, n_bins):
    """
    Reshape a 2D array (T, n_nodes * n_bins) into 3D (T, n_nodes, n_bins).

    Each row of the original CSV is a flattened concatenation of spectra
    from all nodes; this reverses that flattening.
    """
    return arr.reshape(-1, n_nodes, n_bins)


def compute_norm_stats_freq(data_4d):
    """Compute per-frequency-channel mean and std, broadcastable as (F, 1, 1).

    Args:
        data_4d: array of shape (T, F, H, W).

    Returns:
        mean: (F, 1, 1) float32.
        std:  (F, 1, 1) float32, floor at 1e-8.
    """
    axes = (0, 2, 3)
    mean = np.mean(data_4d, axis=axes, keepdims=True).astype(np.float32)
    std = np.std(data_4d, axis=axes, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean, std


def impute_nan_local_time(data_4d: np.ndarray, window_steps: int = 2) -> np.ndarray:
    """Impute NaN values along the time axis using local neighbours.

    For each (F, H, W) cell, for each NaN at time t, compute the mean of
    valid values in the window [t-window_steps, t-1] ∪ [t+1, t+window_steps].
    All imputation values are computed from the *original* neighbors at each
    position and applied simultaneously, avoiding cascading propagation.
    If no valid neighbours exist within the window, the NaN is left as-is
    (no fallback imputation).

    Operates on (T, F, H, W) layout.  Does not drop any timesteps.
    """
    T, F, H, W = data_4d.shape
    total_nan = int(np.isnan(data_4d).sum())
    if total_nan == 0:
        return data_4d

    fill_coords = []
    fill_vals = []

    for f in range(F):
        for h in range(H):
            for w in range(W):
                series = data_4d[:, f, h, w]
                if not np.isnan(series).any():
                    continue
                for t in np.where(np.isnan(series))[0]:
                    left = slice(max(0, t - window_steps), t)
                    right = slice(t + 1, min(T, t + window_steps + 1))
                    vals = np.concatenate([series[left], series[right]])
                    valid = vals[~np.isnan(vals)]
                    if len(valid) > 0:
                        fill_coords.append((t, f, h, w))
                        fill_vals.append(valid.mean())

    for (t, f, h, w), v in zip(fill_coords, fill_vals):
        data_4d[t, f, h, w] = v

    remaining = int(np.isnan(data_4d).sum())
    print(f"  Imputed {len(fill_coords)} NaN cell(s) via local time-axis (window={window_steps})."
          + (f"  Remaining: {remaining}" if remaining else ""))
    return data_4d


def compute_norm_stats(data_3d):
    """
    Compute per-(node, frequency-bin) mean and standard deviation.

    Standard deviations below 1e-8 are clamped to avoid division-by-zero
    during z-score normalization (e.g., for constant or silent channels).
    """
    mean = np.mean(data_3d, axis=0, keepdims=True)
    std = np.std(data_3d, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore(data_3d, mean, std):
    """Apply z-score normalization: (data - mean) / std."""
    return ((data_3d - mean) / std).astype(np.float32)


def denormalize(data, mean, std):
    """Reverse z-score normalization: data * std + mean.

    Also used for min-max denormalization when `mean`/`std` are repurposed
    to hold the min and (max - min) respectively.
    """
    return data * std + mean


def _make_windows(data_len, t_in, t_out, stride):
    """Generate a list of starting indices for sliding windows.

    Each window covers ``t_in + t_out`` steps. The stride controls overlap:
    stride=1 yields maximum overlap; stride=t_in+t_out yields non-overlapping windows.
    The list may be empty if the data is too short for even one window.
    """
    return list(range(0, data_len - t_in - t_out + 1, max(stride, 1)))


def _split_time_ranges(total_len, train_ratio, val_ratio, chronological=True):
    """Partition the time index range into train/val/test index lists.

    When ``chronological=True`` (default for time-series), the split is contiguous:
    the first ``train_ratio`` fraction of time steps go to training, the next
    ``val_ratio`` to validation, and the remainder to testing.

    When ``chronological=False``, indices are randomly permuted before splitting,
    which is useful when the data is not a true time series (e.g., independent samples).
    A fixed seed (42) ensures reproducibility of the random split.
    """
    if total_len <= 0:
        return [], [], []
    if chronological:
        n_train = int(total_len * train_ratio)
        n_val = int(total_len * val_ratio)
        train_range = list(range(0, n_train))
        val_range = list(range(n_train, n_train + n_val))
        test_range = list(range(n_train + n_val, total_len))
    else:
        perm = np.random.RandomState(42).permutation(total_len)
        n_train = int(total_len * train_ratio)
        n_val = int(total_len * val_ratio)
        train_range = perm[:n_train].tolist()
        val_range = perm[n_train:n_train + n_val].tolist()
        test_range = perm[n_train + n_val:].tolist()
    return train_range, val_range, test_range


def create_datasets(csv_path, n_nodes, n_bins, t_in, t_out, stride=1,
                    train_stride=None, val_stride=None, test_stride=None,
                    train_ratio=0.8, val_ratio=0.1, chronological=True,
                    normalization="zscore", fit_on_train_only=True):
    """
    Full pipeline: load CSV, reshape, normalize, split, and create Datasets.

    Normalization statistics are computed either on the training segment only
    (``fit_on_train_only=True``, the default) or on the entire dataset, to avoid
    data leakage from validation/test sets into training.

    The returned ``stats`` dict contains the normalization parameters so they
    can be saved alongside the model checkpoint for correct denormalization
    during evaluation or inference.

    Returns:
        train_ds, val_ds, test_ds: SpectrumDataset instances (may be None if no windows).
        stats: dict with keys "mean", "std", "n_nodes", "n_bins".
    """
    train_stride = stride if train_stride is None else train_stride
    val_stride = stride if val_stride is None else val_stride
    test_stride = stride if test_stride is None else test_stride

    raw = load_csv(csv_path)
    data_3d = reshape_to_3d(raw, n_nodes, n_bins)
    T = len(data_3d)

    # Compute normalization statistics on the training portion only to prevent leakage.
    if fit_on_train_only:
        n_train_raw = int(T * train_ratio)
        train_segment_end = n_train_raw
        train_segment = data_3d[:train_segment_end] if train_segment_end > 0 else data_3d[:1]
        mean, std = compute_norm_stats(train_segment)
    else:
        mean, std = compute_norm_stats(data_3d)

    # Apply the chosen normalization method.
    if normalization == "zscore":
        data_norm = zscore(data_3d, mean, std)
    elif normalization == "minmax":
        dmin = data_3d.min(axis=0, keepdims=True)
        dmax = data_3d.max(axis=0, keepdims=True)
        data_norm = ((data_3d - dmin) / (dmax - dmin + 1e-8)).astype(np.float32)
        # Repurpose mean/std to store min/max-info for later denormalization.
        mean, std = dmin.astype(np.float32), (dmax - dmin + 1e-8).astype(np.float32)
    else:
        # Identity normalization — pass through as-is with dummy stats.
        data_norm = data_3d.astype(np.float32)
        mean = np.zeros((1, n_nodes, n_bins), dtype=np.float32)
        std = np.ones((1, n_nodes, n_bins), dtype=np.float32)

    train_range, val_range, test_range = _split_time_ranges(
        T, train_ratio, val_ratio, chronological,
    )

    train_starts = _make_windows(len(train_range), t_in, t_out, train_stride)
    val_starts = _make_windows(len(val_range), t_in, t_out, val_stride)
    test_starts = _make_windows(len(test_range), t_in, t_out, test_stride)

    # Create Datasets: translate window start offsets from split-relative to absolute indices.
    # e.g., if val_range = [100, 101, ...], a window starting at offset 0 in val_range
    # corresponds to absolute index 100 in data_norm.
    train_ds = SpectrumDataset(data_norm, t_in, t_out,
                               [train_range[s] for s in train_starts]) if train_starts else None
    val_ds = SpectrumDataset(data_norm, t_in, t_out,
                             [val_range[s] for s in val_starts]) if val_starts else None
    test_ds = SpectrumDataset(data_norm, t_in, t_out,
                              [test_range[s] for s in test_starts]) if test_starts else None

    stats = {"mean": mean, "std": std, "n_nodes": n_nodes, "n_bins": n_bins}
    return train_ds, val_ds, test_ds, stats


def create_interpolated_map_datasets(map_path, map_key, t_in, t_out, stride=1,
                                     train_stride=None, val_stride=None, test_stride=None,
                                     train_ratio=0.8, val_ratio=0.1, chronological=True,
                                     normalization="zscore", fit_on_train_only=True,
                                     imputation_cfg=None):
    """Full pipeline for interpolated-map mode: load NPZ, impute NaNs, normalize, split, create Datasets.

    NaN handling (via ``impute_nan_local_time``):
      Fills NaN cells along the time axis using the mean of neighbouring
      valid values within a configurable window.  No timesteps are dropped.

    Args:
        imputation_cfg: dict with keys ``impute`` (bool) and ``window_steps`` (int).

    Returns:
        train_ds, val_ds, test_ds: InterpolatedMapDataset instances (may be None).
        stats: dict with keys "mean", "std", "n_freq", "grid_h", "grid_w".
    """
    train_stride = stride if train_stride is None else train_stride
    val_stride = stride if val_stride is None else val_stride
    test_stride = stride if test_stride is None else test_stride

    data_4d = load_map_npz(map_path, map_key)
    print(f"[create_interpolated_map_datasets] Loaded map: shape {data_4d.shape}")

    # Impute NaN values along time axis (configurable window size).
    ipcfg = imputation_cfg or {}
    should_impute = ipcfg.get("impute", ipcfg.get("enabled", True))
    if should_impute:
        window = int(ipcfg.get("window_steps", 2))
        data_4d = impute_nan_local_time(data_4d, window)
    else:
        nan_count = int(np.isnan(data_4d).sum())
        if nan_count:
            print(f"  WARNING: impute=false, {nan_count} NaN(s) remain in data")

    T, F, H, W = data_4d.shape

    # Compute per-frequency normalization stats on training portion only to prevent leakage.
    if fit_on_train_only:
        n_train_raw = int(T * train_ratio)
        train_segment_end = n_train_raw
        train_segment = data_4d[:train_segment_end] if train_segment_end > 0 else data_4d[:1]
        mean, std = compute_norm_stats_freq(train_segment)
    else:
        mean, std = compute_norm_stats_freq(data_4d)

    if normalization == "zscore":
        data_norm = zscore(data_4d, mean, std)
    else:
        data_norm = data_4d.astype(np.float32)
        mean = np.zeros((1, F, 1, 1), dtype=np.float32)
        std = np.ones((1, F, 1, 1), dtype=np.float32)

    train_range, val_range, test_range = _split_time_ranges(
        T, train_ratio, val_ratio, chronological,
    )

    train_starts = _make_windows(len(train_range), t_in, t_out, train_stride)
    val_starts = _make_windows(len(val_range), t_in, t_out, val_stride)
    test_starts = _make_windows(len(test_range), t_in, t_out, test_stride)

    train_ds = InterpolatedMapDataset(data_norm, t_in, t_out,
                                      [train_range[s] for s in train_starts]) if train_starts else None
    val_ds = InterpolatedMapDataset(data_norm, t_in, t_out,
                                    [val_range[s] for s in val_starts]) if val_starts else None
    test_ds = InterpolatedMapDataset(data_norm, t_in, t_out,
                                     [test_range[s] for s in test_starts]) if test_starts else None

    stats = {"mean": mean, "std": std, "n_freq": F, "grid_h": H, "grid_w": W}
    return train_ds, val_ds, test_ds, stats
