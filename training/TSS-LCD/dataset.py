"""
Dataset loading, preprocessing, and windowing for TSS-LCD.

Handles CSV loading, train/val/test splitting (chronological or random),
normalization (min-max or z-score), sliding-window creation, missing
data masking, and PyTorch DataLoader construction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_csv_numpy(path: str, n_nodes: int, n_bins_per_node: int,
                   cc2_only: bool = False,
                   selected_nodes: list[str] | None = None,
                   node_names: list[str] | None = None) -> np.ndarray:
    """Load a spectrogram CSV into a 2D NumPy array (time, features).

    Supports optionally selecting a single CC2 node or specific named
    nodes. The output is always flattened so that each row is a flat
    concatenation of all node-frequency bins for one time step.

    Args:
        path: Path to the CSV file (no header).
        n_nodes: Total number of nodes in the CSV.
        n_bins_per_node: Frequency bins per node.
        cc2_only: If True, extract only the second node (CC2).
        selected_nodes: Subset of node names to keep.
        node_names: Ordered list of all node names.

    Returns:
        Array of shape (T, L*F) where L is the (possibly filtered) node count.
    """
    data = pd.read_csv(path, header=None).values.astype(np.float32)
    L = n_nodes
    F = n_bins_per_node
    assert data.shape[1] == L * F, f"Expected {L*F} columns, got {data.shape[1]}"
    if cc2_only:
        start = 1 * F
        end = 2 * F
        data = data[:, start:end]
        L_new = 1
        data = data.reshape(-1, L_new, F).reshape(-1, L_new * F)
        return data
    if selected_nodes is not None and node_names is not None:
        indices = []
        for name in selected_nodes:
            idx = node_names.index(name)
            indices.extend(range(idx * F, (idx + 1) * F))
        data = data[:, indices]
        L_new = len(selected_nodes)
        data = data.reshape(-1, L_new, F).reshape(-1, L_new * F)
        return data
    data = data.reshape(-1, L, F).reshape(-1, L * F)
    return data


def split_series_chronological(
    data: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a time series chronologically into train/val/test.

    Preserves temporal order. The remainder after train+val goes to test.
    """
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end = train_end + int(T * val_ratio)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data


def build_windows(
    series: np.ndarray,
    T_in: int,
    T_out: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input-target sliding windows from a time series.

    Each window pair consists of T_in past steps (input) and T_out
    future steps (target), sampled every `stride` time steps.

    Args:
        series: 2D array (T_total, D).
        T_in: Number of input time steps.
        T_out: Number of output (prediction) time steps.
        stride: Step size between consecutive windows.

    Returns:
        X: (num_windows, T_in, D), Y: (num_windows, T_out, D).
    """
    X_list, Y_list = [], []
    D = series.shape[1]
    for i in range(0, len(series) - T_in - T_out + 1, stride):
        X_list.append(series[i:i + T_in])
        Y_list.append(series[i + T_in:i + T_in + T_out])
    if len(X_list) == 0:
        return np.empty((0, T_in, D), dtype=np.float32), np.empty((0, T_out, D), dtype=np.float32)
    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)


class Normalizer:
    """Per-feature min-max or z-score normalizer.

    Fitted on training data and applied to train/val/test. Handles
    edge cases where min==max (min-max) or std==0 (z-score) by
    substituting a small positive offset.
    """

    def __init__(self, method: str = "minmax"):
        self.method = method
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray) -> None:
        """Compute normalization statistics from data (per-feature).

        Args:
            data: Array of any shape; last axis is the feature dimension.
        """
        orig_shape = data.shape
        flat = data.reshape(-1, data.shape[-1])
        if self.method == "minmax":
            self.min_ = flat.min(axis=0, keepdims=True)
            self.max_ = flat.max(axis=0, keepdims=True)
            # Avoid division by zero for constant features
            self.max_[self.max_ == self.min_] = self.min_[self.max_ == self.min_] + 1.0
        elif self.method == "zscore":
            self.mean_ = flat.mean(axis=0, keepdims=True)
            self.std_ = flat.std(axis=0, keepdims=True)
            self.std_[self.std_ == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the fitted normalization."""
        if self.method == "minmax":
            return (data - self.min_) / (self.max_ - self.min_)
        elif self.method == "zscore":
            return (data - self.mean_) / self.std_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Undo the normalization, restoring original scale."""
        if self.method == "minmax":
            return data * (self.max_ - self.min_) + self.min_
        elif self.method == "zscore":
            return data * self.std_ + self.mean_


def create_masks(X: np.ndarray, missing_rate: float,
                 strategy: str = "random") -> np.ndarray:
    """Generate a boolean mask indicating observed (True) vs missing (False) entries.

    Two strategies are supported:
      - 'random': independently mask each element with probability `missing_rate`.
      - 'continuous': mask a contiguous block of length `missing_rate * T_in` per
        feature per sample, simulating sensor dropouts.

    Args:
        X: (B, T_in, D) input windows.
        missing_rate: Fraction of entries to mask.
        strategy: 'random' or 'continuous'.

    Returns:
        Boolean array of same shape as X.
    """
    if missing_rate <= 0:
        return np.ones_like(X, dtype=bool)
    if strategy == "random":
        mask = np.random.binomial(1, 1 - missing_rate, size=X.shape).astype(bool)
    elif strategy == "continuous":
        mask = np.ones_like(X, dtype=bool)
        T_in, D = X.shape[1], X.shape[2]
        for b in range(X.shape[0]):
            for d in range(D):
                cont_len = max(1, int(T_in * missing_rate))
                start = np.random.randint(0, T_in - cont_len + 1)
                mask[b, start:start + cont_len, d] = False
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
    return mask


class TSSLCDataset(Dataset):
    """PyTorch Dataset for TSS-LCD with optional missing-data masking.

    Each sample is an (input_window, target_window) pair. The input may
    be partially masked to simulate missing observations during training.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        missing_rate: float = 0.25,
        masking_strategy: str = "random",
        zero_pad_missing: bool = True,
        complete_observation_baseline: bool = False,
        transform: Callable | None = None,
    ):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        # Disable masking entirely in complete-observation baseline mode
        self.missing_rate = missing_rate if not complete_observation_baseline else 0.0
        self.masking_strategy = masking_strategy
        self.zero_pad_missing = zero_pad_missing
        self.complete_observation_baseline = complete_observation_baseline
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()
        y = self.Y[idx].clone()
        if self.missing_rate > 0:
            mask = create_masks(
                x.unsqueeze(0).numpy(),
                self.missing_rate,
                self.masking_strategy,
            )[0]
            mask_t = torch.from_numpy(mask)
            # Zero out masked entries instead of removing them
            if self.zero_pad_missing:
                x = x * mask_t.float()
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def get_dataloaders(
    config: dict,
    normalizer: Normalizer | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Normalizer, int, int, int]:
    """Build train/val/test DataLoaders from a configuration dictionary.

    The pipeline is: load CSV → split → normalize → build windows →
    wrap in TSSLCDataset → create DataLoaders.

    Args:
        config: Full experiment configuration dict.
        normalizer: Optional pre-fitted normalizer (reused across stages).

    Returns:
        (train_loader, val_loader, test_loader, normalizer, L, F, T_out).
    """
    data_cfg = config["data"]
    window_cfg = config["windowing"]
    split_cfg = config["split"]
    preproc_cfg = config["preprocessing"]
    train_cfg = config["training"]

    # Load raw CSV data
    data = load_csv_numpy(
        data_cfg["dataset_path"],
        data_cfg["n_nodes"],
        data_cfg["n_bins_per_node"],
        cc2_only=data_cfg.get("cc2_only_smoke_test", False),
        selected_nodes=data_cfg.get("selected_nodes"),
        node_names=data_cfg.get("node_names"),
    )

    L = data_cfg["n_nodes"]
    F = data_cfg["n_bins_per_node"]
    if data_cfg.get("cc2_only_smoke_test", False):
        L = 1
    elif data_cfg.get("selected_nodes") is not None:
        L = len(data_cfg["selected_nodes"])

    T_in = window_cfg["input_sequence_length"]
    T_out = window_cfg["prediction_horizon"]
    train_stride = window_cfg.get("train_stride", 1)
    val_stride = window_cfg.get("val_stride", 1)
    test_stride = window_cfg.get("test_stride", 1)

    # Split into train/val/test sets
    if split_cfg.get("chronological_split", True):
        train_series, val_series, test_series = split_series_chronological(
            data, split_cfg["train_ratio"], split_cfg["val_ratio"],
        )
    else:
        T = data.shape[0]
        idx = np.random.permutation(T)
        n_train = int(T * split_cfg["train_ratio"])
        n_val = int(T * split_cfg["val_ratio"])
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        train_series = data[train_idx]
        val_series = data[val_idx]
        test_series = data[test_idx]

    # Fit normalizer on training data only (prevents data leakage)
    if normalizer is None:
        normalizer = Normalizer(method=preproc_cfg.get("normalization", "minmax"))
        normalizer.fit(train_series)
    train_norm = normalizer.transform(train_series)
    val_norm = normalizer.transform(val_series)
    test_norm = normalizer.transform(test_series)

    # Build windows per split with per-split stride
    X_train, Y_train = build_windows(train_norm, T_in, T_out, train_stride)
    X_val, Y_val = build_windows(val_norm, T_in, T_out, val_stride)
    X_test, Y_test = build_windows(test_norm, T_in, T_out, test_stride)

    missing_rate = preproc_cfg.get("missing_rate", 0.25)
    masking_strategy = preproc_cfg.get("masking_strategy", "random")
    zero_pad_missing = preproc_cfg.get("zero_pad_missing", True)
    complete_obs = preproc_cfg.get("complete_observation_baseline", False)

    train_dataset = TSSLCDataset(
        X_train, Y_train, missing_rate, masking_strategy,
        zero_pad_missing, complete_obs,
    )
    val_dataset = TSSLCDataset(
        X_val, Y_val, missing_rate, masking_strategy,
        zero_pad_missing, complete_obs,
    )
    test_dataset = TSSLCDataset(
        X_test, Y_test, missing_rate, masking_strategy,
        zero_pad_missing, complete_obs,
    )

    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, normalizer, L, F, T_out
