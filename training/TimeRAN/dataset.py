from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_csv(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2:
        raise ValueError(f"Expected 2D CSV, got shape {data.shape}")
    return data


class AERPAWDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        t_in: int,
        t_out: int,
        stride: int,
    ):
        self.t_in = t_in
        self.t_out = t_out
        self.stride = stride
        self.window_len = t_in + t_out

        starts = []
        for start in range(0, len(indices) - self.window_len + 1, stride):
            starts.append(start)
        self.starts = np.array(starts, dtype=np.int64)
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        global_start = self.indices[start]
        global_end = self.indices[start + self.window_len - 1] + 1

        window = self.data[global_start:global_end]
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
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    normalization: str = "revin_only",
):
    data = load_csv(csv_path)
    T, C = data.shape
    assert val_ratio >= 0 and train_ratio + val_ratio <= 1.0
    test_ratio = 1.0 - train_ratio - val_ratio

    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    n_test = T - n_train - n_val

    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, T)

    norm_stats = None
    if normalization == "train_zscore":
        train_data = data[train_idx]
        mean = np.mean(train_data, axis=0, keepdims=True)
        std = np.std(train_data, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        norm_stats = {"mean": mean, "std": std}

        train_data_norm = (train_data - mean) / std
        val_data_norm = (data[val_idx] - mean) / std
        test_data_norm = (data[test_idx] - mean) / std

        train_ds = AERPAWDataset(train_data_norm, train_idx, t_in, t_out, stride)
        val_ds = AERPAWDataset(val_data_norm, val_idx, t_in, t_out, stride) if n_val > 0 else None
        test_ds = AERPAWDataset(test_data_norm, test_idx, t_in, t_out, stride) if n_test > 0 else None
    else:
        train_ds = AERPAWDataset(data, train_idx, t_in, t_out, stride)
        val_ds = AERPAWDataset(data, val_idx, t_in, t_out, stride) if n_val > 0 else None
        test_ds = AERPAWDataset(data, test_idx, t_in, t_out, stride) if n_test > 0 else None

    return train_ds, val_ds, test_ds, norm_stats
