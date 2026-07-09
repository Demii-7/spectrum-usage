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


def _trim_trailing_all_nan_timesteps(data: np.ndarray):
    """Drop only all-NaN timesteps at the end of the series.

    Internal missing timesteps are preserved so chronology stays intact and can
    be handled by local interpolation.
    """
    if data.size == 0:
        return data, 0

    valid_mask = ~np.isnan(data).all(axis=tuple(range(1, data.ndim)))
    if valid_mask.all():
        return data, 0

    last_valid = np.where(valid_mask)[0]
    if len(last_valid) == 0:
        return data[:0], data.shape[0]

    trimmed_len = int(data.shape[0] - (last_valid[-1] + 1))
    if trimmed_len <= 0:
        return data, 0
    return data[:last_valid[-1] + 1], trimmed_len


def _impute_local_temporal_and_frequency(data: np.ndarray, freq_axis: int, window_steps: int = 2):
    """Impute NaNs with time-neighbour fill, then frequency-neighbour fallback.

    The first pass fills each missing cell from nearby timesteps while keeping
    the frequency/spatial location fixed. Remaining NaNs are then filled from
    nearby frequencies at the same timestep and spatial location. Any NaNs that
    still cannot be resolved are left in place and reported.
    """
    total_nan = int(np.isnan(data).sum())
    if total_nan == 0:
        return data, {
            "initial_nan_count": 0,
            "temporal_imputed": 0,
            "frequency_imputed": 0,
            "remaining_nan_count": 0,
        }

    temporal_fill_coords = []
    temporal_fill_vals = []
    trailing_axes = tuple(range(1, data.ndim))
    spatial_shape = data.shape[2:]

    for freq_idx in range(data.shape[freq_axis]):
        for spatial_idx in np.ndindex(spatial_shape):
            series = data[(slice(None), freq_idx, *spatial_idx)]
            if not np.isnan(series).any():
                continue
            for t in np.where(np.isnan(series))[0]:
                left = slice(max(0, t - window_steps), t)
                right = slice(t + 1, min(data.shape[0], t + window_steps + 1))
                vals = np.concatenate([series[left], series[right]])
                valid = vals[~np.isnan(vals)]
                if len(valid) > 0:
                    temporal_fill_coords.append((t, freq_idx, *spatial_idx))
                    temporal_fill_vals.append(valid.mean())

    for coord, value in zip(temporal_fill_coords, temporal_fill_vals):
        data[coord] = value

    frequency_fill_coords = []
    frequency_fill_vals = []
    remaining_nan_coords = np.argwhere(np.isnan(data))
    max_freq = data.shape[freq_axis]
    for coord in remaining_nan_coords:
        t = int(coord[0])
        freq_idx = int(coord[freq_axis])
        spatial_idx = tuple(int(v) for v in coord[2:])
        low = max(0, freq_idx - window_steps)
        high = min(max_freq, freq_idx + window_steps + 1)
        freq_values = data[(t, slice(low, high), *spatial_idx)]
        valid = freq_values[~np.isnan(freq_values)]
        if len(valid) > 0:
            frequency_fill_coords.append((t, freq_idx, *spatial_idx))
            frequency_fill_vals.append(valid.mean())

    for coord, value in zip(frequency_fill_coords, frequency_fill_vals):
        data[coord] = value

    stats = {
        "initial_nan_count": total_nan,
        "temporal_imputed": len(temporal_fill_coords),
        "frequency_imputed": len(frequency_fill_coords),
        "remaining_nan_count": int(np.isnan(data).sum()),
    }
    return data, stats


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


def clean_nan_csv(data_3d: np.ndarray, window_steps: int = 2) -> tuple[np.ndarray, dict]:
    """Clean CSV-mode NaNs while preserving chronology.

    The data is shaped as (T, nodes, bins). Only trailing all-NaN timesteps are
    removed. Internal missing values are filled locally in time, then by nearby
    frequency bins within the same node and timestep.
    """
    trimmed, trimmed_count = _trim_trailing_all_nan_timesteps(data_3d)
    transposed = trimmed.transpose(0, 2, 1)
    cleaned_t, stats = _impute_local_temporal_and_frequency(transposed.copy(), freq_axis=1, window_steps=window_steps)
    cleaned = cleaned_t.transpose(0, 2, 1)
    stats["trimmed_trailing_timesteps"] = trimmed_count
    return cleaned, stats


def clean_nan_map(data_4d: np.ndarray, window_steps: int = 2) -> tuple[np.ndarray, dict]:
    """Clean map-mode NaNs while preserving chronology.

    The data is shaped as (T, F, H, W). Only trailing all-NaN timesteps are
    removed. Internal missing values are filled locally in time, then by nearby
    frequency channels at the same spatial location and timestep.
    """
    trimmed, trimmed_count = _trim_trailing_all_nan_timesteps(data_4d)
    cleaned, stats = _impute_local_temporal_and_frequency(trimmed.copy(), freq_axis=1, window_steps=window_steps)
    stats["trimmed_trailing_timesteps"] = trimmed_count
    return cleaned, stats


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
                    normalization="zscore", fit_on_train_only=True,
                    nan_window_steps=2):
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
    data_3d, nan_stats = clean_nan_csv(data_3d, window_steps=nan_window_steps)
    print(
        "  CSV NaN cleanup: "
        f"trimmed_trailing_timesteps={nan_stats['trimmed_trailing_timesteps']}, "
        f"initial_nan_count={nan_stats['initial_nan_count']}, "
        f"temporal_imputed={nan_stats['temporal_imputed']}, "
        f"frequency_imputed={nan_stats['frequency_imputed']}, "
        f"remaining_nan_count={nan_stats['remaining_nan_count']}"
    )
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

    stats = {"mean": mean, "std": std, "n_nodes": n_nodes, "n_bins": n_bins, "nan_stats": nan_stats}
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
        data_4d, nan_stats = clean_nan_map(data_4d, window)
        print(
            "  Map NaN cleanup: "
            f"trimmed_trailing_timesteps={nan_stats['trimmed_trailing_timesteps']}, "
            f"initial_nan_count={nan_stats['initial_nan_count']}, "
            f"temporal_imputed={nan_stats['temporal_imputed']}, "
            f"frequency_imputed={nan_stats['frequency_imputed']}, "
            f"remaining_nan_count={nan_stats['remaining_nan_count']}"
        )
    else:
        nan_stats = {
            "trimmed_trailing_timesteps": 0,
            "initial_nan_count": int(np.isnan(data_4d).sum()),
            "temporal_imputed": 0,
            "frequency_imputed": 0,
            "remaining_nan_count": int(np.isnan(data_4d).sum()),
        }
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

    stats = {"mean": mean, "std": std, "n_freq": F, "grid_h": H, "grid_w": W, "nan_stats": nan_stats}
    return train_ds, val_ds, test_ds, stats
