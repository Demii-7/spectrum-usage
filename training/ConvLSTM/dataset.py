import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    def __init__(self, data_3d, t_in, t_out, start_indices):
        self.data = torch.from_numpy(data_3d).float()
        self.t_in = t_in
        self.t_out = t_out
        self.indices = start_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i : i + self.t_in]
        y = self.data[i + self.t_in : i + self.t_in + self.t_out]
        x = x.unsqueeze(0).transpose(0, 1)
        y = y.unsqueeze(0).transpose(0, 1)
        return x, y


def load_csv(csv_path):
    return np.loadtxt(csv_path, delimiter=",").astype(np.float32)


def reshape_to_3d(arr, n_nodes, n_bins):
    return arr.reshape(-1, n_nodes, n_bins)


def compute_norm_stats(data_3d):
    mean = np.mean(data_3d, axis=0, keepdims=True)
    std = np.std(data_3d, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore(data_3d, mean, std):
    return ((data_3d - mean) / std).astype(np.float32)


def denormalize(data, mean, std):
    return data * std + mean


def split_indices(n_windows, train_ratio, val_ratio, chronological=True):
    if n_windows <= 0:
        return [], [], []
    if chronological:
        n_train = int(n_windows * train_ratio)
        n_val = int(n_windows * val_ratio)
        train_idx = list(range(0, n_train))
        val_idx = list(range(n_train, n_train + n_val))
        test_idx = list(range(n_train + n_val, n_windows))
    else:
        perm = np.random.RandomState(42).permutation(n_windows)
        n_train = int(n_windows * train_ratio)
        n_val = int(n_windows * val_ratio)
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train : n_train + n_val].tolist()
        test_idx = perm[n_train + n_val :].tolist()
    return train_idx, val_idx, test_idx


def create_datasets(csv_path, n_nodes, n_bins, t_in, t_out, stride,
                    train_ratio, val_ratio, chronological=True,
                    normalization="zscore", fit_on_train_only=True):
    raw = load_csv(csv_path)
    data_3d = reshape_to_3d(raw, n_nodes, n_bins)

    total_windows = (len(data_3d) - t_in - t_out) // stride + 1
    if total_windows <= 0:
        raise ValueError(f"Not enough data ({len(data_3d)} steps) for T_in={t_in} + T_out={t_out}")

    if fit_on_train_only:
        n_train = int(total_windows * train_ratio)
        train_end_idx = (n_train - 1) * stride + t_in + t_out
        train_segment = data_3d[:train_end_idx] if train_end_idx > 0 else data_3d[:1]
        mean, std = compute_norm_stats(train_segment)
    else:
        mean, std = compute_norm_stats(data_3d)

    if normalization == "zscore":
        data_norm = zscore(data_3d, mean, std)
    elif normalization == "minmax":
        dmin = data_3d.min(axis=0, keepdims=True)
        dmax = data_3d.max(axis=0, keepdims=True)
        data_norm = ((data_3d - dmin) / (dmax - dmin + 1e-8)).astype(np.float32)
        mean, std = dmin.astype(np.float32), (dmax - dmin + 1e-8).astype(np.float32)
    else:
        data_norm = data_3d.astype(np.float32)
        mean = np.zeros((1, n_nodes, n_bins), dtype=np.float32)
        std = np.ones((1, n_nodes, n_bins), dtype=np.float32)

    all_start = list(range(0, len(data_3d) - t_in - t_out + 1, stride))
    train_s, val_s, test_s = split_indices(len(all_start), train_ratio, val_ratio, chronological)

    train_ds = SpectrumDataset(data_norm, t_in, t_out, [all_start[i] for i in train_s]) if train_s else None
    val_ds = SpectrumDataset(data_norm, t_in, t_out, [all_start[i] for i in val_s]) if val_s else None
    test_ds = SpectrumDataset(data_norm, t_in, t_out, [all_start[i] for i in test_s]) if test_s else None

    stats = {"mean": mean, "std": std, "n_nodes": n_nodes, "n_bins": n_bins}
    return train_ds, val_ds, test_ds, stats
