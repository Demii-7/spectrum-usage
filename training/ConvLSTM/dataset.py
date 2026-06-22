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


def _make_windows(data_len, t_in, t_out, stride):
    return list(range(0, data_len - t_in - t_out + 1, max(stride, 1)))


def _split_time_ranges(total_len, train_ratio, val_ratio, chronological=True):
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
    train_stride = stride if train_stride is None else train_stride
    val_stride = stride if val_stride is None else val_stride
    test_stride = stride if test_stride is None else test_stride

    raw = load_csv(csv_path)
    data_3d = reshape_to_3d(raw, n_nodes, n_bins)
    T = len(data_3d)

    if fit_on_train_only:
        n_train_raw = int(T * train_ratio)
        train_segment_end = n_train_raw
        train_segment = data_3d[:train_segment_end] if train_segment_end > 0 else data_3d[:1]
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

    train_range, val_range, test_range = _split_time_ranges(
        T, train_ratio, val_ratio, chronological,
    )

    train_starts = _make_windows(len(train_range), t_in, t_out, train_stride)
    val_starts = _make_windows(len(val_range), t_in, t_out, val_stride)
    test_starts = _make_windows(len(test_range), t_in, t_out, test_stride)

    train_ds = SpectrumDataset(data_norm, t_in, t_out,
                               [train_range[s] for s in train_starts]) if train_starts else None
    val_ds = SpectrumDataset(data_norm, t_in, t_out,
                             [val_range[s] for s in val_starts]) if val_starts else None
    test_ds = SpectrumDataset(data_norm, t_in, t_out,
                              [test_range[s] for s in test_starts]) if test_starts else None

    stats = {"mean": mean, "std": std, "n_nodes": n_nodes, "n_bins": n_bins}
    return train_ds, val_ds, test_ds, stats
