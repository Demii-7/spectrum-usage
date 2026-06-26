import numpy as np
import torch
from torch.utils.data import Dataset


def node_column_slice(node_names, bins_per_node=250):
    known = {"CC1": 0, "CC2": 1, "LW1": 2}
    cols = []
    for name in node_names:
        idx = known.get(name)
        if idx is None:
            raise ValueError(f"Unknown node {name}, known: {list(known.keys())}")
        cols.extend(range(idx * bins_per_node, (idx + 1) * bins_per_node))
    return cols


def random_mask(shape, missing_rate):
    return (torch.rand(shape) > missing_rate).float()


def block_mask(shape, missing_rate):
    mask = torch.ones(shape)
    T, H, W, F = shape
    block_len = max(1, int(W * missing_rate * 0.5))
    n_blocks = max(1, int(T * H * missing_rate))
    for _ in range(n_blocks):
        t = np.random.randint(0, T)
        h = np.random.randint(0, H)
        w_start = np.random.randint(0, W - block_len + 1)
        mask[t, h, w_start:w_start + block_len, :] = 0.0
    return mask


def frequency_mask(shape, missing_rate):
    T, H, W, F = shape
    n_masked = max(1, int(W * missing_rate))
    mask = torch.ones(shape)
    for t in range(T):
        freq_indices = np.random.choice(W, n_masked, replace=False)
        mask[t, :, freq_indices, :] = 0.0
    return mask


def node_mask(shape, missing_rate, n_nodes):
    T, H, W, F = shape
    mask = torch.ones(shape)
    n_masked = max(1, int(n_nodes * missing_rate))
    for t in range(T):
        node_indices = np.random.choice(n_nodes, n_masked, replace=False)
        mask[t, node_indices, :, :] = 0.0
    return mask


class AERPAWDataset(Dataset):
    def __init__(self, data, config, split="train"):
        self.config = config
        self.split = split
        self.T_in = config["windowing"]["input_sequence_length"]
        self.T_out = config["windowing"]["prediction_horizon"]
        self.missing_rate = config["preprocessing"]["missing_rate"]
        self.missing_strategy = config["preprocessing"].get("missing_strategy", "random")
        self.mask_targets = config["preprocessing"].get("mask_targets", False)

        stride_key = f"{split}_stride"
        stride = config["windowing"].get(stride_key)
        if stride is None:
            stride = config["windowing"].get("test_stride" if split == "test" else "val_stride")
        if stride is None:
            stride = 1 if split == "train" else self.T_out
        self.stride = stride

        maps = self._build_windows(data)
        self.maps = torch.from_numpy(maps).float()

    def _build_windows(self, data):
        T_total, H, W, F = data.shape
        total_len = self.T_in + self.T_out
        indices = []
        for start in range(0, T_total - total_len + 1, self.stride):
            indices.append(start)
        self.window_starts = indices
        windows = []
        for s in indices:
            windows.append(data[s:s + total_len])
        return np.stack(windows)

    def __len__(self):
        return len(self.window_starts)

    def _generate_mask(self, shape):
        if self.missing_strategy == "block":
            return block_mask(shape, self.missing_rate)
        elif self.missing_strategy == "frequency":
            return frequency_mask(shape, self.missing_rate)
        elif self.missing_strategy == "node":
            n_nodes = shape[1]
            return node_mask(shape, self.missing_rate, n_nodes)
        else:
            return random_mask(shape, self.missing_rate)

    def __getitem__(self, idx):
        window = self.maps[idx]
        X = window[:self.T_in].clone()
        Y = window[self.T_in:self.T_in + self.T_out].clone()

        mask_shape = (self.T_in,) + tuple(X.shape[1:])
        mask = self._generate_mask(mask_shape)

        X_masked = X * mask
        return X_masked, mask, Y


def load_and_split(config, cc2_only=False):
    csv_path = config["data"]["dataset_path"]
    nodes = config["data"]["selected_nodes"]
    bins = config["data"]["bins_per_node"]
    node_names = config["data"]["node_names"]
    max_rows = config["data"].get("max_rows")

    if cc2_only:
        nodes = ["CC2"]

    cols = node_column_slice(nodes, bins)
    raw = np.loadtxt(csv_path, delimiter=",")
    if raw.ndim == 1:
        raw = raw.reshape(-1, len(cols))
    if max_rows is not None:
        raw = raw[:max_rows]
    data_2d = raw[:, cols]

    T, _ = data_2d.shape
    H = len(nodes)
    W = bins
    data_map = data_2d.reshape(T, H, W, 1).astype(np.float32)

    ratios = config["split"]
    tr, vr, ter = ratios["train_ratio"], ratios["val_ratio"], ratios["test_ratio"]
    n_train = int(T * tr)
    n_val = int(T * vr)
    n_test = int(T * ter)

    train_data = data_map[:n_train]
    val_data = data_map[n_train:n_train + n_val]
    test_data = data_map[n_train + n_val:n_train + n_val + n_test]

    norm_cfg = config["preprocessing"]
    method = norm_cfg.get("normalization", "minmax")
    fit_on_train = norm_cfg.get("fit_on_train_only", True)
    target_range = norm_cfg.get("minmax_range", [-1, 1])

    if fit_on_train:
        dmin = train_data.min()
        dmax = train_data.max()
    else:
        dmin = data_map.min()
        dmax = data_map.max()

    from utils import normalize_minmax
    train_norm = normalize_minmax(train_data, dmin, dmax, target_range)
    val_norm = normalize_minmax(val_data, dmin, dmax, target_range)
    test_norm = normalize_minmax(test_data, dmin, dmax, target_range)

    stats = {
        "method": method,
        "dmin": float(dmin),
        "dmax": float(dmax),
        "range": target_range,
    }

    return train_norm, val_norm, test_norm, stats, node_names if not cc2_only else ["CC2"]
