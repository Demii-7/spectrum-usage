"""
Training script for DeepSPred (3D-SwinSTB).

Usage:
    python training/DeepSPred/train.py --config training/DeepSPred/smoke_test/config.yaml
    python training/DeepSPred/train.py --config training/DeepSPred/config.yaml --epochs 50
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Allow running from repo root.
sys.path.insert(0, os.path.dirname(__file__))

from dataset import create_datasets
from model import SwinSTB3D
from utils import set_seed, get_device, compute_metrics, save_checkpoint


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_tgt = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item() * x.size(0)
        all_pred.append(pred.cpu())
        all_tgt.append(y.cpu())
    loss = total_loss / len(loader.dataset)
    pred_cat = torch.cat(all_pred)
    tgt_cat  = torch.cat(all_tgt)
    metrics  = compute_metrics(pred_cat, tgt_cat)
    metrics["loss"] = loss
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, help="Path to config.yaml")
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--csv",        default=None, help="Override CSV path")
    args = parser.parse_args()

    config = load_config(args.config)
    tcfg = config["training"]

    # CLI overrides.
    if args.batch_size: tcfg["batch_size"] = args.batch_size
    if args.epochs:     tcfg["epochs"]     = args.epochs
    if args.lr:         tcfg["learning_rate"] = args.lr

    set_seed(tcfg["seed"])
    device = get_device(config.get("device", "auto"))
    print(f"Device: {device}")

    # Data.
    train_ds, val_ds, _, norm_stats = create_datasets(config, csv_path=args.csv)
    if train_ds is None or len(train_ds) == 0:
        raise RuntimeError("No training samples. Check config (input_frames, csv path).")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds else 0}")

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"],
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=False) if val_ds else None

    # Model.
    model = SwinSTB3D(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tcfg["learning_rate"],
                                  weight_decay=tcfg.get("weight_decay", 0.05))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    ckpt_dir  = config["checkpoints"]["dir"]
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_left = tcfg.get("patience", 10)
    log = []

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_metrics = {}
        val_loss = float("nan")
        if val_loader:
            val_metrics = validate(model, val_loader, criterion, device)
            val_loss = val_metrics["loss"]
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_left = tcfg.get("patience", 10)
                save_checkpoint(best_path, model, optimizer, epoch,
                                norm_stats, config, val_metrics)
            else:
                patience_left -= 1

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{tcfg['epochs']}  "
              f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
              f"rmse={val_metrics.get('rmse', float('nan')):.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  {elapsed:.1f}s")

        log.append({"epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, **val_metrics})

        if patience_left <= 0:
            print("Early stopping.")
            break

    # Save final checkpoint and training log.
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    save_checkpoint(final_path, model, optimizer, epoch, norm_stats, config,
                    {"val_loss": best_val_loss})
    log_path = os.path.join(ckpt_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Done. Best val_loss={best_val_loss:.5f}  checkpoint: {best_path}")


if __name__ == "__main__":
    main()
