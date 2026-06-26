import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import get_device, load_config, load_checkpoint, denormalize, normalize_minmax
from dataset import node_column_slice
from model import DSwinLSTM_I


class InferenceDataset(Dataset):
    def __init__(self, data, T_in, T_out, stride=1):
        self.T_in = T_in
        self.T_out = T_out
        total_len = T_in + T_out
        self.windows = []
        for start in range(0, len(data) - total_len + 1, stride):
            self.windows.append(data[start:start + T_in])
        self.windows = np.stack(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        X = self.windows[idx]
        dummy_mask = np.ones_like(X)
        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(dummy_mask).float(),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--config", default=None, help="Config YAML (overrides checkpoint)")
    args = parser.parse_args()

    device = get_device(config.get("device", {}).get("device", "auto"))
    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    stats = ckpt["norm_stats"]

    if args.config:
        config = load_config(args.config)

    T_in = config["windowing"]["input_sequence_length"]
    T_out = config["windowing"]["prediction_horizon"]
    nodes = config["data"]["selected_nodes"]
    bins = config["data"]["bins_per_node"]
    cc2_only = config["data"].get("cc2_only_smoke_test", False)
    if cc2_only:
        nodes = ["CC2"]

    cols = node_column_slice(nodes, bins)
    raw = np.loadtxt(args.input, delimiter=",")
    if raw.ndim == 1:
        raw = raw.reshape(-1, len(cols))
    data_2d = raw[:, cols]
    T, _ = data_2d.shape
    H = len(nodes)
    W = bins
    data_map = data_2d.reshape(T, H, W, 1).astype(np.float32)

    dmin = stats["dmin"]
    dmax = stats["dmax"]
    target_range = stats.get("range", [-1, 1])
    data_norm = normalize_minmax(data_map, dmin, dmax, target_range)

    dataset = InferenceDataset(data_norm, T_in, T_out, stride=1)
    loader = DataLoader(dataset, batch_size=config["training"].get("batch_size", 4), shuffle=False)

    model = DSwinLSTM_I(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_pred = []
    with torch.no_grad():
        for X, mask in loader:
            X = X.permute(0, 1, 4, 2, 3).contiguous().to(device)
            mask = mask.permute(0, 1, 2, 3, 4).contiguous().to(device)
            pred = model(X, mask)
            all_pred.append(pred.cpu().numpy())

    pred_norm = np.concatenate(all_pred, axis=0)
    pred_dbm = denormalize(pred_norm, stats)
    B, T_o, C, H, W = pred_dbm.shape
    pred_flat = pred_dbm.reshape(B, T_o, -1)

    np.savetxt(args.output, pred_flat[0], delimiter=",", fmt="%.6f")
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
