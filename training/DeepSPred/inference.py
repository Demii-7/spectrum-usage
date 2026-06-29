"""
Standalone inference with a trained DeepSPred checkpoint.

Given a CSV file and a checkpoint, outputs predictions as a .npy file.

Usage:
    python training/DeepSPred/inference.py \
        --checkpoint training/DeepSPred/smoke_test/checkpoints/best_model.pt \
        --csv        training/data/cc2_smoke_test.csv \
        --out        predictions.npy
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from dataset import create_datasets
from model import SwinSTB3D
from utils import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv",        required=True)
    parser.add_argument("--out",        default="predictions.npy")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, torch.device("cpu"))
    config = ckpt["config"]
    device = get_device(config.get("device", "auto"))

    train_ds, val_ds, test_ds, _ = create_datasets(config, csv_path=args.csv)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    if ds is None or len(ds) == 0:
        print(f"No samples for split '{args.split}'.")
        return

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    model = SwinSTB3D(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).cpu().numpy())

    out = np.concatenate(preds, axis=0)   # (N, T_in, 3, H, W_orig)
    np.save(args.out, out)
    print(f"Predictions saved to {args.out}  shape={out.shape}")


if __name__ == "__main__":
    main()
