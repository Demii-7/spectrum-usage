import os, sys, argparse, json, time
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_csv, load_map_npz, prepare_map_series
from utils import get_device, load_checkpoint, denormalize
from train import build_model, resolve_feature_dims, run_model


def save_metadata(output_path, data_format, num_streams, pred_len, n_features):
    metadata_path = os.path.splitext(output_path)[0] + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "data_format": data_format,
            "num_streams": num_streams,
            "prediction_horizon": pred_len,
            "n_features": n_features,
        }, f, indent=2)
    return metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV or NPZ")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config.get("device", {}).get("device", "auto"))
    print(f"Device: {device}")

    checkpoint = load_checkpoint(args.checkpoint, device)
    stats = checkpoint["norm_stats"]
    data_format = config["data"].get("format", "csv")
    windowing = config["windowing"]
    seq_len = windowing["seq_len"]
    label_len = windowing["label_len"]
    pred_len = windowing["pred_len"]

    if data_format == "interpolated_map":
        map_path = args.input
        raw = load_map_npz(map_path, config["data"].get("map_key", "map_db"))
        series, meta = prepare_map_series(map_path, config["data"].get("map_key", "map_db"))
        num_streams = series.shape[0]
        if series.shape[1] < seq_len:
            raise ValueError(f"Map input has {series.shape[1]} timesteps, need at least {seq_len}")
        data_norm = ((series - stats["mean"]) / (stats["std"] + 1e-8)).astype(np.float32)
        output_path = args.output or "predictions.npz"
    else:
        raw = load_csv(args.input)
        num_streams = 1
        if len(raw) < seq_len:
            raise ValueError(f"CSV input has {len(raw)} rows, need at least {seq_len}")
        data_norm = ((raw[None, :, :] - stats["mean"]) / (stats["std"] + 1e-8)).astype(np.float32)
        output_path = args.output or "predictions.csv"

    resolve_feature_dims(config, stats)
    model, model_cfg = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    infer_start = time.perf_counter()
    preds = []
    with torch.no_grad():
        for sample_idx in range(num_streams):
            series_t = torch.from_numpy(data_norm[sample_idx : sample_idx + 1]).float().to(device)
            x_enc = series_t[:, -seq_len:, :]
            dec_input = torch.zeros(1, label_len + pred_len, x_enc.shape[-1], device=device)
            dec_input[:, :label_len, :] = x_enc[:, -label_len:, :]
            output = run_model(model, model_cfg, x_enc, dec_input, device)
            preds.append(output.cpu())
    total_infer = time.perf_counter() - infer_start

    pred = torch.cat(preds, dim=0)
    mean_t = torch.from_numpy(stats["mean"]).float()
    std_t = torch.from_numpy(stats["std"]).float()
    pred_denorm = denormalize(pred, mean_t, std_t).numpy()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    if data_format == "interpolated_map":
        np.savez(output_path, predictions=pred_denorm)
    else:
        np.savetxt(output_path, pred_denorm.squeeze(0), delimiter=",", fmt="%.6f")
    metadata_path = save_metadata(output_path, data_format, num_streams, pred_len, stats["n_features"])
    print(f"Predictions saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Total inference time: {total_infer:.2f}s")


if __name__ == "__main__":
    main()
