import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import get_device, load_config, load_checkpoint, denormalize, normalize_minmax
from dataset import node_column_slice, clean_map_nans
from model import DSwinLSTM_I


class InferenceDataset(Dataset):
    def __init__(self, data, T_in, T_out, stride=1):
        self.T_in = T_in
        self.T_out = T_out
        total_len = T_in + T_out
        self.windows = [data[start:start + T_in] for start in range(0, len(data) - total_len + 1, stride)]
        self.windows = np.stack(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        X = self.windows[idx]
        dummy_mask = np.ones_like(X)
        return torch.from_numpy(X).float(), torch.from_numpy(dummy_mask).float()


def load_inference_map(config, stats):
    raw = np.load(config["data"]["map_path"])[config["data"].get("map_key", "map_db")].astype(np.float32)
    preproc = config.get("preprocessing", {})
    cleaned, _ = clean_map_nans(
        raw,
        time_window_steps=int(preproc.get("nan_time_window_steps", 2)),
        frequency_window_steps=int(preproc.get("nan_frequency_window_steps", 2)),
    )
    if stats.get("method", "minmax") == "minmax":
        return normalize_minmax(cleaned, stats["dmin"], stats["dmax"], stats.get("range", [-1, 1]))
    mean = stats["mean"]
    std = stats["std"]
    return ((cleaned - mean) / (std + 1e-8)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--input", required=True, help="Path to input CSV or NPZ")
    parser.add_argument("--output", default=None, help="Output path")
    parser.add_argument("--config", default=None, help="Config YAML (overrides checkpoint)")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, "cpu")
    config = ckpt["config"]
    stats = ckpt["norm_stats"]
    if args.config:
        config = load_config(args.config)

    device = get_device(config.get("device", {}).get("device", "auto"))
    T_in = config["windowing"]["input_sequence_length"]
    T_out = config["windowing"]["prediction_horizon"]
    data_format = config["data"].get("format", "csv")

    if data_format == "interpolated_map":
        raw = np.load(args.input)[config["data"].get("map_key", "map_db")].astype(np.float32)
        preproc = config.get("preprocessing", {})
        data_map, _ = clean_map_nans(
            raw,
            time_window_steps=int(preproc.get("nan_time_window_steps", 2)),
            frequency_window_steps=int(preproc.get("nan_frequency_window_steps", 2)),
        )
        if stats.get("method", "minmax") == "minmax":
            data_norm = normalize_minmax(data_map, stats["dmin"], stats["dmax"], stats.get("range", [-1, 1]))
        else:
            data_norm = ((data_map - stats["mean"]) / (stats["std"] + 1e-8)).astype(np.float32)
        config["model"]["map_height"] = int(data_norm.shape[1])
        config["model"]["map_width"] = int(data_norm.shape[2])
        config["model"]["input_channels"] = int(data_norm.shape[3])
        output_path = args.output or "predictions.npz"
    else:
        nodes = config["data"]["selected_nodes"]
        bins = config["data"]["bins_per_node"]
        cc2_only = config["data"].get("cc2_only_smoke_test", False)
        if cc2_only:
            nodes = ["CC2"]
        cols = node_column_slice(nodes, bins)
        raw = np.loadtxt(args.input, delimiter=",")
        if raw.ndim == 1:
            raw = raw.reshape(-1, len(cols))
        data_2d = raw if raw.shape[1] == len(cols) else raw[:, cols]
        T, _ = data_2d.shape
        H = len(nodes)
        W = bins
        data_map = data_2d.reshape(T, H, W, 1).astype(np.float32)
        data_norm = normalize_minmax(data_map, stats["dmin"], stats["dmax"], stats.get("range", [-1, 1])) if stats.get("method", "minmax") == "minmax" else ((data_map - stats["mean"]) / (stats["std"] + 1e-8)).astype(np.float32)
        config["model"]["map_height"] = H
        config["model"]["map_width"] = W
        config["model"]["input_channels"] = 1
        output_path = args.output or "predictions.csv"

    dataset = InferenceDataset(data_norm, T_in, T_out, stride=1)
    loader = DataLoader(dataset, batch_size=config["training"].get("batch_size", 4), shuffle=False)

    model = DSwinLSTM_I(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    infer_start = time.perf_counter()
    all_pred = []
    with torch.no_grad():
        for X, mask in loader:
            X = X.permute(0, 1, 4, 2, 3).contiguous().to(device)
            mask = mask.to(device)
            pred = model(X, mask)
            all_pred.append(pred.cpu().numpy())
    total_infer = time.perf_counter() - infer_start

    pred_norm = np.concatenate(all_pred, axis=0)
    pred_denorm = denormalize(pred_norm, stats)
    if data_format == "interpolated_map":
        np.savez(output_path, predictions=pred_denorm)
    else:
        B, T_o, C, H, W = pred_denorm.shape
        np.savetxt(output_path, pred_denorm.reshape(B * T_o, -1), delimiter=",", fmt="%.6f")

    metadata_path = os.path.splitext(output_path)[0] + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "data_format": data_format,
            "prediction_horizon": T_out,
            "height": int(config["model"]["map_height"]),
            "width": int(config["model"]["map_width"]),
            "channels": int(config["model"]["input_channels"]),
        }, f, indent=2)
    print(f"Predictions saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Total inference time: {total_infer:.2f}s")


if __name__ == "__main__":
    main()
