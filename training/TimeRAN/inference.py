"""Inference script for running a trained TimeRAN model on arbitrary CSV data.

Loads a checkpoint, runs a sliding-window forecast over the input CSV, and
saves the flattened predictions.  Supports on-the-fly z-score denormalization
if the checkpoint contains training-set statistics.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_csv
from utils import denormalize, get_device, load_checkpoint

from momentfm import MOMENTPipeline


# Mapping from user-facing size labels to HuggingFace Hub model identifiers.
VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


@torch.no_grad()
def predict(model, data: np.ndarray, t_in: int, t_out: int, stride: int, device: torch.device):
    """Run the model over a sliding window of the input data.

    Args:
        model: The forecasting model in eval mode.
        data: Input time-series of shape (T, C).
        t_in: Number of input time steps per window.
        t_out: Forecast horizon per window.
        stride: Step between consecutive window starts.
        device: Target torch device.

    Returns:
        Predictions array of shape (num_windows, C, t_out), or an empty array
        if the input is too short for even one window.
    """
    model.eval()
    T, C = data.shape
    predictions = []

    for start in range(0, T - t_in + 1, stride):
        window = data[start : start + t_in]
        # Add batch dimension and transpose to (1, C, t_in).
        x = torch.as_tensor(window.T, dtype=torch.float32).unsqueeze(0).to(device)
        input_mask = torch.ones(1, t_in, device=device)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(x_enc=x, input_mask=input_mask)
        else:
            out = model(x_enc=x, input_mask=input_mask)

        pred_np = out.forecast.cpu().numpy()
        predictions.append(pred_np)

    if not predictions:
        return np.array([]).reshape(0, 0, t_out)

    pred = np.concatenate(predictions, axis=0)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    device = get_device("auto")
    print(f"Device: {device}")

    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    norm_stats = ckpt.get("norm_stats")

    t_in = config["windowing"]["input_sequence_length"]
    t_out = config["windowing"]["prediction_horizon"]
    stride = config["windowing"]["stride"]
    variant = config["model"]["checkpoint_size"].lower()
    normalization = config["preprocessing"]["normalization"]

    data = load_csv(args.input)
    T, C = data.shape
    print(f"Input shape: {data.shape}")

    # Apply the same normalization that was used during training.
    if normalization == "train_zscore" and norm_stats:
        data = (data - norm_stats["mean"]) / norm_stats["std"]

    model_name = VARIANT_TO_MODEL[variant]
    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": t_out,
            "seq_len": t_in,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
    )
    model.init()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    pred = predict(model, data, t_in, t_out, stride, device)

    # Denormalise predictions back to the original scale for saved output.
    if normalization == "train_zscore" and norm_stats:
        pred = denormalize(pred, norm_stats["mean"], norm_stats["std"])

    # Flatten to (windows * horizon, channels) for CSV export.
    B, C, H = pred.shape
    flat_pred = pred.transpose(0, 2, 1).reshape(B * H, C)

    np.savetxt(args.output, flat_pred, delimiter=",", fmt="%.6f")
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
