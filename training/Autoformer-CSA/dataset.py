import numpy as np
import torch
from torch.utils.data import Dataset


class AERPAWDataset(Dataset):
    def __init__(self, data_2d, seq_len, label_len, pred_len, start_indices):
        self.data = torch.from_numpy(data_2d).float()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.indices = start_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        s_begin = i
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        return seq_x, seq_y


def load_csv(csv_path):
    return np.loadtxt(csv_path, delimiter=",").astype(np.float32)


def compute_norm_stats(data_2d):
    mean = np.mean(data_2d, axis=0, keepdims=True)
    std = np.std(data_2d, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore(data_2d, mean, std):
    return ((data_2d - mean) / std).astype(np.float32)


def _make_windows(data_len, seq_len, pred_len, stride):
    return list(range(0, data_len - seq_len - pred_len + 1, max(stride, 1)))


def _split_time_ranges(total_len, train_ratio, val_ratio, chronological=True):
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    if chronological:
        train_range = list(range(0, n_train))
        val_range = list(range(n_train, n_train + n_val))
        test_range = list(range(n_train + n_val, total_len))
    else:
        perm = np.random.RandomState(42).permutation(total_len)
        train_range = perm[:n_train].tolist()
        val_range = perm[n_train:n_train + n_val].tolist()
        test_range = perm[n_train + n_val:].tolist()
    return train_range, val_range, test_range


def create_datasets(csv_path, seq_len, label_len, pred_len,
                    train_stride=1, val_stride=None, test_stride=None,
                    train_ratio=0.8, val_ratio=0.1, chronological=True,
                    normalization="zscore", fit_on_train_only=True):
    val_stride = pred_len if val_stride is None else val_stride
    test_stride = pred_len if test_stride is None else test_stride

    raw = load_csv(csv_path)
    T = len(raw)

    if fit_on_train_only:
        n_train_raw = int(T * train_ratio)
        train_segment = raw[:n_train_raw] if n_train_raw > 0 else raw[:1]
        mean, std = compute_norm_stats(train_segment)
    else:
        mean, std = compute_norm_stats(raw)

    if normalization == "zscore":
        data_norm = zscore(raw, mean, std)
    elif normalization == "minmax":
        dmin = raw.min(axis=0, keepdims=True)
        dmax = raw.max(axis=0, keepdims=True)
        data_norm = ((raw - dmin) / (dmax - dmin + 1e-8)).astype(np.float32)
        mean, std = dmin.astype(np.float32), (dmax - dmin + 1e-8).astype(np.float32)
    else:
        data_norm = raw.astype(np.float32)
        mean = np.zeros((1, raw.shape[1]), dtype=np.float32)
        std = np.ones((1, raw.shape[1]), dtype=np.float32)

    train_range, val_range, test_range = _split_time_ranges(
        T, train_ratio, val_ratio, chronological,
    )

    train_starts = _make_windows(len(train_range), seq_len, pred_len, train_stride)
    val_starts = _make_windows(len(val_range), seq_len, pred_len, val_stride)
    test_starts = _make_windows(len(test_range), seq_len, pred_len, test_stride)

    train_ds = AERPAWDataset(
        data_norm, seq_len, label_len, pred_len,
        [train_range[s] for s in train_starts],
    ) if train_starts else None

    val_ds = AERPAWDataset(
        data_norm, seq_len, label_len, pred_len,
        [val_range[s] for s in val_starts],
    ) if val_starts else None

    test_ds = AERPAWDataset(
        data_norm, seq_len, label_len, pred_len,
        [test_range[s] for s in test_starts],
    ) if test_starts else None

    stats = {"mean": mean, "std": std}
    return train_ds, val_ds, test_ds, stats
