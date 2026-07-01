"""Inference script for a trained STS-PredNet checkpoint.

Supports both CSV and interpolated-map inputs. The model remains a direct
single-step forecaster; multiple requested horizons are handled by rebuilding
inference samples at different ``prediction_offset`` values.
"""
import os
import json
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from stsprednet import STSPredNet
from utils import get_device, load_checkpoint, denormalize
from dataset import (
    load_csv,
    reshape_to_3d,
    minmax_neg1_pos1,
    STSPredNetDataset,
    collate_branch_samples,
    generate_target_indices,
    load_map_npz,
    impute_full_nan_timesteps,
    impute_nan_local_time,
    impute_nan_local_frequency,
    trim_trailing_nan_timesteps,
)


def build_dataset_for_horizon(data, bcfg, prediction_offset, add_channel_dim):
    """Build a branch dataset for one direct-prediction horizon."""
    target_indices = generate_target_indices(
        len(data),
        prediction_offset,
        bcfg["use_closeness"],
        bcfg["use_period"],
        bcfg["use_trend"],
        bcfg["lc"],
        bcfg["lp"],
        bcfg["lq"],
        bcfg["period_interval"],
        bcfg["trend_interval"],
    )
    if len(target_indices) == 0:
        return None
    return STSPredNetDataset(
        data,
        target_indices,
        bcfg["use_closeness"],
        bcfg["use_period"],
        bcfg["use_trend"],
        bcfg["lc"],
        bcfg["lp"],
        bcfg["lq"],
        bcfg["period_interval"],
        bcfg["trend_interval"],
        prediction_offset,
        add_channel_dim=add_channel_dim,
    )


def load_and_normalize_input(config, stats, input_path):
    """Load unseen input data and apply checkpoint normalization."""
    dcfg = config["data"]
    data_format = dcfg.get("format", "csv")
    method = stats.get("method", "minmax_neg1_pos1")

    if data_format == "interpolated_map":
        map_key = dcfg.get("map_key", "map_db")
        data = load_map_npz(input_path, map_key)
        ipcfg = config.get("preprocessing", {}).get("imputation", {})
        if ipcfg.get("enabled", True):
            time_window = int(ipcfg.get("window_steps", 2))
            freq_window = int(ipcfg.get("frequency_window_steps", time_window))
            data = impute_full_nan_timesteps(data, time_window)
            data = impute_nan_local_time(data, time_window)
            data = impute_nan_local_frequency(data, freq_window)
            data, _ = trim_trailing_nan_timesteps(data)
        if method == "minmax_neg1_pos1":
            dmin = stats["dmin"]
            dmax = stats["dmax"]
            data_norm = (2.0 * (data - dmin) / (dmax - dmin + 1e-8) - 1.0).astype(np.float32)
        elif method == "zscore":
            data_norm = ((data - stats["mean"]) / (stats["std"] + 1e-8)).astype(np.float32)
        else:
            data_norm = data.astype(np.float32)
        return data_norm, False

    raw = load_csv(input_path)
    data = reshape_to_3d(raw, dcfg["n_nodes"], dcfg["bins_per_node"])
    if method == "minmax_neg1_pos1":
        data_norm = minmax_neg1_pos1(data, stats["dmin"], stats["dmax"])
    elif method == "zscore":
        data_norm = ((data - stats["mean"]) / (stats["std"] + 1e-8)).astype(np.float32)
    else:
        data_norm = data.astype(np.float32)
    return data_norm, True


def save_predictions(output_path, horizon_outputs, data_format):
    """Save all inference outputs plus metadata."""
    metadata = {
        "data_format": data_format,
        "horizons": [],
        "row_order": "rows are grouped by requested horizon order, then sample index",
    }

    if data_format == "interpolated_map":
        npz_payload = {}
        for horizon, pred_dbm in horizon_outputs:
            key = f"t_plus_{horizon}"
            npz_payload[key] = pred_dbm
            metadata["horizons"].append({"horizon": horizon, "key": key, "num_samples": int(pred_dbm.shape[0])})
        np.savez(output_path, **npz_payload)
    else:
        rows = []
        for horizon, pred_dbm in horizon_outputs:
            rows.append(pred_dbm.reshape(pred_dbm.shape[0], -1))
            metadata["horizons"].append({"horizon": horizon, "num_samples": int(pred_dbm.shape[0])})
        np.savetxt(output_path, np.concatenate(rows, axis=0), delimiter=",", fmt="%.6f")

    metadata_path = os.path.splitext(output_path)[0] + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def prepare_model_config(config, stats):
    """Restore architecture fields needed to instantiate the trained model."""
    model_config = dict(config)
    model_config["model"] = dict(config["model"])
    if model_config["data"].get("format", "csv") == "interpolated_map":
        model_config["model"]["input_channels"] = int(stats["n_freq"])
        model_config["model"]["map_height"] = int(stats["grid_h"])
        model_config["model"]["map_width"] = int(stats["grid_w"])
    return model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to CSV or NPZ input matching training mode")
    parser.add_argument("--output", default=None)
    parser.add_argument("--horizons", type=int, nargs="+")
    args = parser.parse_args()

    device = get_device("auto")
    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    stats = ckpt["norm_stats"]
    data_format = config["data"].get("format", "csv")
    horizons = args.horizons or [config["branches"].get("prediction_offset", 1)]

    output_path = args.output
    if output_path is None:
        output_path = "predictions.npz" if data_format == "interpolated_map" else "predictions.csv"

    print(f"Loading input: {args.input}")
    data_norm, add_channel_dim = load_and_normalize_input(config, stats, args.input)

    bcfg = dict(config["branches"])
    if data_format == "interpolated_map":
        temporal = config["data"].get("temporal_overrides", {})
        if temporal:
            bcfg["lc"] = int(temporal.get("lc", bcfg["lc"]))
            bcfg["lp"] = int(temporal.get("lp", bcfg["lp"]))
            bcfg["period_interval"] = int(temporal.get("period_interval", bcfg["period_interval"]))

    model_config = prepare_model_config(config, stats)
    model = STSPredNet(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    horizon_outputs = []
    total_inference_time_seconds = 0.0
    for horizon in horizons:
        dataset = build_dataset_for_horizon(data_norm, bcfg, int(horizon), add_channel_dim=add_channel_dim)
        if dataset is None:
            raise ValueError(f"Input too short to build inference samples for horizon t+{horizon}.")

        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_branch_samples)
        all_pred = []
        horizon_start_time = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                closeness = batch.get("closeness")
                period = batch.get("period")
                trend = batch.get("trend")
                if closeness is not None:
                    closeness = closeness.to(device)
                if period is not None:
                    period = period.to(device)
                if trend is not None:
                    trend = trend.to(device)
                pred = model(closeness, period, trend)
                all_pred.append(pred.cpu())

        pred = torch.cat(all_pred, dim=0).numpy()
        horizon_inference_time = time.perf_counter() - horizon_start_time
        total_inference_time_seconds += horizon_inference_time
        pred_dbm = denormalize(pred, stats) if stats.get("method", "none") != "none" else pred
        horizon_outputs.append((int(horizon), pred_dbm))
        print(f"Horizon t+{horizon}: inference time {horizon_inference_time:.2f}s")

    metadata_path = save_predictions(output_path, horizon_outputs, data_format)
    print(f"Predictions saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Total inference time: {total_inference_time_seconds:.2f}s")


if __name__ == "__main__":
    main()
