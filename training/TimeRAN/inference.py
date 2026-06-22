import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_csv
from utils import denormalize, get_device, load_checkpoint

from momentfm import MOMENTPipeline


VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


@torch.no_grad()
def predict(model, data: np.ndarray, t_in: int, t_out: int, stride: int, device: torch.device):
    model.eval()
    T, C = data.shape
    predictions = []

    for start in range(0, T - t_in + 1, stride):
        window = data[start : start + t_in]
        x = torch.as_tensor(window.T, dtype=torch.float32).unsqueeze(0).to(device)
        input_mask = torch.ones(1, t_in, device=device)

        with torch.cuda.amp.autocast():
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

    if normalization == "train_zscore" and norm_stats:
        data = (data - norm_stats["mean"]) / norm_stats["std"]

    model_name = VARIANT_TO_MODEL[variant]
    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": t_out,
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

    if normalization == "train_zscore" and norm_stats:
        pred = denormalize(pred, norm_stats["mean"], norm_stats["std"])

    B, C, H = pred.shape
    flat_pred = pred.transpose(0, 2, 1).reshape(B * H, C)

    np.savetxt(args.output, flat_pred, delimiter=",", fmt="%.6f")
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
