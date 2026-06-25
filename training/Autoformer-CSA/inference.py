import os, sys, argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_csv
from utils import get_device, load_checkpoint, denormalize
from train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, default=None, help="Path for predictions CSV")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config.get("device", {}).get("device", "auto"))
    print(f"Device: {device}")

    checkpoint = load_checkpoint(args.checkpoint, device)
    norm_stats = checkpoint["norm_stats"]
    mean, std = norm_stats["mean"], norm_stats["std"]

    windowing = config["windowing"]
    seq_len = windowing["seq_len"]
    label_len = windowing["label_len"]
    pred_len = windowing["pred_len"]

    raw = load_csv(args.input)
    T = len(raw)
    if T < seq_len:
        print(f"Error: input has {T} rows, need at least {seq_len}")
        return

    data_norm = ((raw - mean) / (std + 1e-8)).astype(np.float32)
    data_t = torch.from_numpy(data_norm).float().unsqueeze(0).to(device)

    model, model_cfg = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x_enc = data_t[:, -seq_len:, :]
        dec_input = torch.zeros(1, label_len + pred_len, data_t.shape[-1], device=device)
        dec_input[:, :label_len, :] = x_enc[:, -label_len:, :]

        x_mark_enc = torch.zeros(1, seq_len, 4, device=device)
        x_mark_dec = torch.zeros(1, label_len + pred_len, 4, device=device)

        output = model(x_enc, x_mark_enc, dec_input, x_mark_dec)

    pred_norm = output.cpu().squeeze(0)
    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()
    pred_dbm = denormalize(pred_norm, mean_t, std_t).numpy()

    output_path = args.output or "predictions.csv"
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    np.savetxt(output_path, pred_dbm, delimiter=",", fmt="%.6f")
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
