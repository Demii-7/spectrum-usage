"""Dataset loading and PyTorch Dataset for spectrum occupancy time-series forecasting.

Provides AERPAWDataset, a PyTorch Dataset that slices multi-variate time series
into (input, target) windows, and create_datasets which handles CSV loading,
train/val/test splitting, and optional z-score normalization.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def load_csv(path: str) -> np.ndarray:
    """Load a 2D CSV file (time steps x channels) as a NumPy array.

    Args:
        path: Path to the CSV file.

    Returns:
        2D array of shape (T, C) where T is time steps and C is channels.

    Raises:
        ValueError: If the CSV does not have exactly 2 dimensions.
    """
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2:
        raise ValueError(f"Expected 2D CSV, got shape {data.shape}")
    return data


class AERPAWDataset(Dataset):
    """PyTorch Dataset for sliding-window forecasting on spectrum data.

    Uses an index array (indices) that maps logical positions to actual rows
    in *data*, enabling clean time-series splits without copying large arrays.
    Each sample returns a (t_in, C) input and (t_out, C) target window.
    """

    def __init__(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        t_in: int,
        t_out: int,
        stride: int,
    ):
        """
        Args:
            data: Raw time-series array of shape (T_full, C).
            indices: Logical-to-physical row mapping for this split.
            t_in: Number of input time steps per sample.
            t_out: Number of forecast time steps per sample.
            stride: Step size between consecutive window starts.
        """
        self.t_in = t_in
        self.t_out = t_out
        self.stride = stride
        self.window_len = t_in + t_out

        # Pre-compute all valid window start positions along the index array.
        starts = []
        for start in range(0, len(indices) - self.window_len + 1, stride):
            starts.append(start)
        self.starts = np.array(starts, dtype=np.int64)
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        # Map the logical window start to the physical row range in data.
        start = self.starts[idx]
        global_start = self.indices[start]
        global_end = self.indices[start + self.window_len - 1] + 1

        window = self.data[global_start:global_end]
        # Transpose from (time, channels) to (channels, time) for the model.
        x = window[: self.t_in].T
        y = window[self.t_in :].T
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        return x, y


def create_datasets(
    csv_path: str,
    t_in: int = 128,
    t_out: int = 16,
    stride: int = 16,
    train_stride: Optional[int] = None,
    val_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    normalization: str = "revin_only",
):
    """Load a CSV, split temporally, and return train/val/test Datasets.

    The split is contiguous: the first *train_ratio* fraction is training,
    the next *val_ratio* is validation, and the remainder is test.

    Args:
        csv_path: Path to the input CSV (T x C).
        t_in: Input sequence length.
        t_out: Forecast horizon.
        stride: Default stride between windows.
        train_stride: Override stride for training set (defaults to *stride*).
        val_stride: Override stride for validation set.
        test_stride: Override stride for test set.
        train_ratio: Fraction of time steps used for training.
        val_ratio: Fraction of time steps used for validation.
        normalization: ``"train_zscore"`` to normalize using training-set
            statistics; any other value skips normalization.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, norm_stats). Each dataset is
        ``None`` if its split has no samples. *norm_stats* is a dict with
        keys ``"mean"`` and ``"std"`` when ``normalization="train_zscore"``,
        otherwise ``None``.
    """
    train_stride = stride if train_stride is None else train_stride
    val_stride = stride if val_stride is None else val_stride
    test_stride = stride if test_stride is None else test_stride

    data = load_csv(csv_path)
    T, C = data.shape
    assert val_ratio >= 0 and train_ratio + val_ratio <= 1.0
    test_ratio = 1.0 - train_ratio - val_ratio

    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    n_test = T - n_train - n_val

    # Build index arrays that define each split's coverage of the full data.
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, T)

    norm_stats = None
    if normalization == "train_zscore":
        # Compute per-channel mean/std on training data only to avoid leakage.
        train_data = data[train_idx]
        mean = np.mean(train_data, axis=0, keepdims=True)
        std = np.std(train_data, axis=0, keepdims=True)
        # Clamp zero-variance channels to unit std to avoid division by zero.
        std = np.where(std < 1e-8, 1.0, std)
        norm_stats = {"mean": mean, "std": std}

        train_data_norm = (train_data - mean) / std
        val_data_norm = (data[val_idx] - mean) / std
        test_data_norm = (data[test_idx] - mean) / std

        train_ds = AERPAWDataset(train_data_norm, train_idx, t_in, t_out, train_stride)
        val_ds = AERPAWDataset(val_data_norm, val_idx, t_in, t_out, val_stride) if n_val > 0 else None
        test_ds = AERPAWDataset(test_data_norm, test_idx, t_in, t_out, test_stride) if n_test > 0 else None
    else:
        # No normalization — feed raw data directly (RevIN handles it at runtime).
        train_ds = AERPAWDataset(data, train_idx, t_in, t_out, train_stride)
        val_ds = AERPAWDataset(data, val_idx, t_in, t_out, val_stride) if n_val > 0 else None
        test_ds = AERPAWDataset(data, test_idx, t_in, t_out, test_stride) if n_test > 0 else None

    return train_ds, val_ds, test_ds, norm_stats
