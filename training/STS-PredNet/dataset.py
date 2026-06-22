import numpy as np
import torch
from torch.utils.data import Dataset


def load_csv(csv_path):
    return np.loadtxt(csv_path, delimiter=",").astype(np.float32)


def reshape_to_3d(arr, n_nodes, bins_per_node):
    return arr.reshape(-1, n_nodes, bins_per_node)


def compute_minmax_stats(data_3d):
    dmin = data_3d.min(axis=0, keepdims=True)
    dmax = data_3d.max(axis=0, keepdims=True)
    return dmin.astype(np.float32), dmax.astype(np.float32)


def minmax_neg1_pos1(data_3d, dmin, dmax):
    eps = 1e-8
    return (2.0 * (data_3d - dmin) / (dmax - dmin + eps) - 1.0).astype(np.float32)


def denormalize(data, dmin, dmax):
    eps = 1e-8
    return 0.5 * (data + 1.0) * (dmax - dmin + eps) + dmin


def zscore(data_3d, mean, std):
    return ((data_3d - mean) / (std + 1e-8)).astype(np.float32)


def split_indices(n_total, train_ratio, val_ratio, chronological=True):
    if n_total <= 0:
        return [], [], []
    if chronological:
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_idx = list(range(0, n_train))
        val_idx = list(range(n_train, n_train + n_val))
        test_idx = list(range(n_train + n_val, n_total))
    else:
        perm = np.random.RandomState(42).permutation(n_total)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train:n_train + n_val].tolist()
        test_idx = perm[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


class STSPredNetDataset(Dataset):
    def __init__(self, data_3d, target_indices,
                 use_closeness, use_period, use_trend,
                 lc, lp, lq, period_interval, trend_interval,
                 prediction_offset):
        self.data = torch.from_numpy(data_3d).float()
        self.target_indices = target_indices
        self.use_closeness = use_closeness
        self.use_period = use_period
        self.use_trend = use_trend
        self.lc = lc
        self.lp = lp
        self.lq = lq
        self.period_interval = period_interval
        self.trend_interval = trend_interval
        self.prediction_offset = prediction_offset

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):
        target_idx = self.target_indices[idx]
        t = target_idx - self.prediction_offset
        result = {}

        if self.use_closeness:
            c_start = t - self.lc + 1
            c_seq = self.data[c_start:t + 1]
            c_seq = c_seq.unsqueeze(1)
            result["closeness"] = c_seq

        if self.use_period:
            p_indices = [target_idx - i * self.period_interval
                         for i in range(self.lp, 0, -1)]
            p_seq = torch.stack([self.data[i] for i in p_indices], dim=0)
            p_seq = p_seq.unsqueeze(1)
            result["period"] = p_seq

        if self.use_trend:
            q_indices = [target_idx - i * self.trend_interval
                         for i in range(self.lq, 0, -1)]
            q_seq = torch.stack([self.data[i] for i in q_indices], dim=0)
            q_seq = q_seq.unsqueeze(1)
            result["trend"] = q_seq

        target = self.data[target_idx]
        result["target"] = target

        return result


def generate_target_indices(total_len, prediction_offset,
                            use_closeness, use_period, use_trend,
                            lc, lp, lq, period_interval, trend_interval):
    indices = []
    for target_idx in range(prediction_offset, total_len):
        t = target_idx - prediction_offset
        valid = True
        if use_closeness:
            if t - lc + 1 < 0:
                valid = False
        if use_period:
            if target_idx - lp * period_interval < 0:
                valid = False
        if use_trend:
            if target_idx - lq * trend_interval < 0:
                valid = False
        if valid:
            indices.append(target_idx)
    return indices


def collate_branch_samples(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "target":
            out[k] = torch.stack([b[k] for b in batch], dim=0).unsqueeze(1)
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def create_datasets(csv_path, config):
    dcfg = config["data"]
    pcfg = config["preprocessing"]
    scfg = config["splits"]
    bcfg = config["branches"]

    n_nodes = dcfg["n_nodes"]
    bins_per_node = dcfg["bins_per_node"]
    normalization = pcfg["normalization"]
    fit_on_train_only = pcfg["fit_on_train_only"]

    use_c = bcfg["use_closeness"]
    use_p = bcfg["use_period"]
    use_t = bcfg["use_trend"]
    lc = bcfg["lc"]
    lp = bcfg["lp"]
    lq = bcfg["lq"]
    period_interval = bcfg["period_interval"]
    trend_interval = bcfg["trend_interval"]
    prediction_offset = bcfg.get("prediction_offset", 1)

    raw = load_csv(csv_path)
    data_3d = reshape_to_3d(raw, n_nodes, bins_per_node)

    all_targets = generate_target_indices(
        len(data_3d), prediction_offset,
        use_c, use_p, use_t,
        lc, lp, lq, period_interval, trend_interval,
    )

    n_total = len(all_targets)
    train_idx_list, val_idx_list, test_idx_list = split_indices(
        n_total, scfg["train_ratio"], scfg["val_ratio"],
        scfg.get("chronological_split", True),
    )

    train_targets = [all_targets[i] for i in train_idx_list]
    val_targets = [all_targets[i] for i in val_idx_list]
    test_targets = [all_targets[i] for i in test_idx_list]

    if normalization == "minmax_neg1_pos1":
        if fit_on_train_only and train_targets:
            train_data_end = max(train_targets) + 1
            train_segment = data_3d[:train_data_end]
            dmin, dmax = compute_minmax_stats(train_segment)
        else:
            dmin, dmax = compute_minmax_stats(data_3d)
        data_norm = minmax_neg1_pos1(data_3d, dmin, dmax)
        stats = {"dmin": dmin, "dmax": dmax, "method": "minmax_neg1_pos1"}
    elif normalization == "zscore":
        if fit_on_train_only and train_targets:
            train_data_end = max(train_targets) + 1
            train_segment = data_3d[:train_data_end]
            mean, std = train_segment.mean(axis=0, keepdims=True), train_segment.std(axis=0, keepdims=True)
        else:
            mean, std = data_3d.mean(axis=0, keepdims=True), data_3d.std(axis=0, keepdims=True)
        data_norm = zscore(data_3d, mean, std)
        stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "method": "zscore"}
    else:
        data_norm = data_3d.astype(np.float32)
        stats = {"method": "none"}

    train_ds = STSPredNetDataset(data_norm, train_targets, use_c, use_p, use_t,
                                  lc, lp, lq, period_interval, trend_interval,
                                  prediction_offset) if train_targets else None
    val_ds = STSPredNetDataset(data_norm, val_targets, use_c, use_p, use_t,
                                lc, lp, lq, period_interval, trend_interval,
                                prediction_offset) if val_targets else None
    test_ds = STSPredNetDataset(data_norm, test_targets, use_c, use_p, use_t,
                                 lc, lp, lq, period_interval, trend_interval,
                                 prediction_offset) if test_targets else None

    return train_ds, val_ds, test_ds, stats
