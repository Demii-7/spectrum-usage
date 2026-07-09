from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import denormalize_array, load_single_site_csv
from model import VanillaLSTMForecaster
from utils import load_config, load_normalization_stats, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VanillaLSTM inference on a single-site CSV.")
    parser.add_argument("--checkpoint", default="training/VanillaLSTM/checkpoints/best_model.pt")
    parser.add_argument("--config", default=None, help="Optional config override.")
    parser.add_argument("--csv", required=True, help="Input single-site CSV path.")
    parser.add_argument("--output", default="training/VanillaLSTM/evaluation/inference_predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location="cpu", weights_only=False)
    config = load_config(args.config) if args.config else checkpoint["config"]

    raw = load_single_site_csv(config, csv_path=args.csv)
    input_length = int(config["windowing"]["input_sequence_length"])
    if raw.shape[0] < input_length:
        raise RuntimeError(f"Need at least {input_length} rows for inference, got {raw.shape[0]}.")

    stats = load_normalization_stats(Path(config["paths"]["checkpoints_dir"]) / "normalization_stats.pt")
    normalized = ((raw - stats["mean_dbm"]) / stats["std_dbm"]).astype(np.float32)
    model_input = torch.from_numpy(normalized[-input_length:][None, ...])

    model = VanillaLSTMForecaster(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_norm = model(model_input).cpu().numpy()[0]
    pred_dbm = denormalize_array(pred_norm, stats)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, pred_dbm.astype(np.float32), delimiter=",", fmt="%.6f")
    print(f"Saved {pred_dbm.shape[0]} forecast rows x {pred_dbm.shape[1]} bins to {output_path}")


if __name__ == "__main__":
    main()
