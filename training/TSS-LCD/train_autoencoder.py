from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import LatentSpaceEncoder, LatentSpaceDecoder
from utils import load_config, set_seed, get_device, save_checkpoint


def train_epoch(enc, dec, loader, optimizer, device, clip):
    enc.train()
    dec.train()
    total_loss = 0.0
    for x, y in loader:
        y = y.to(device)
        z = enc(y)
        y_hat = dec(z)
        loss = nn.functional.mse_loss(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()), clip,
            )
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def validate(enc, dec, loader, device):
    enc.eval()
    dec.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            z = enc(y)
            y_hat = dec(z)
            loss = nn.functional.mse_loss(y_hat, y)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config = load_config(args.config)
    set_seed(config.get("seed"))
    device = get_device(config["device"]["device"])

    train_loader, val_loader, _, normalizer, L, F, T_out = get_dataloaders(config)

    model_cfg = config["model"]
    train_cfg = config["training"]
    ckpt_dir = script_dir / train_cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    enc = LatentSpaceEncoder(
        T_out=T_out, L=L, F=F,
        latent_dim=model_cfg["latent_dim"],
        num_blocks=model_cfg.get("autoencoder_num_blocks", 3),
        init_channels=model_cfg.get("autoencoder_initial_channels", 32),
        kernel_size=model_cfg.get("autoencoder_kernel_size", 3),
        pool_kernel=model_cfg.get("autoencoder_pool_kernel", 2),
        pool_stride=model_cfg.get("autoencoder_pool_stride", 2),
        activation=model_cfg.get("autoencoder_activation", "relu"),
    ).to(device)
    dec = LatentSpaceDecoder(
        T_out=T_out, L=L, F=F,
        latent_dim=model_cfg["latent_dim"],
        num_blocks=model_cfg.get("autoencoder_num_blocks", 3),
        init_channels=model_cfg.get("autoencoder_initial_channels", 32),
        kernel_size=model_cfg.get("autoencoder_kernel_size", 3),
        activation=model_cfg.get("autoencoder_activation", "relu"),
    ).to(device)

    params = list(enc.parameters()) + list(dec.parameters())
    opt_name = train_cfg.get("optimizer", "adam")
    lr = train_cfg.get("autoencoder_learning_rate", 0.0001)
    wd = train_cfg.get("weight_decay", 0.0)
    if opt_name == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # Scheduler
    scheduler = None
    lr_sched_name = train_cfg.get("lr_scheduler", "none")
    epochs = train_cfg.get("autoencoder_epochs", 300)
    if lr_sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_sched_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=train_cfg.get("lr_scheduler_factor", 0.5),
            patience=train_cfg.get("lr_scheduler_patience", 5),
        )

    # Git hash
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    # Environment info
    env_info = {
        "device": str(device),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)

    clip = train_cfg.get("gradient_clip", 0.0)
    best_val = float("inf")
    training_log = []
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(enc, dec, train_loader, optimizer, device, clip)
        val_loss = validate(enc, dec, val_loader, device)
        epoch_time = time.perf_counter() - epoch_start

        samples_per_sec = len(train_loader.dataset) / epoch_time if epoch_time > 0 else 0
        batches_per_sec = len(train_loader) / epoch_time if epoch_time > 0 else 0

        print(f"[AE] Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}"
              f"  {epoch_time:.2f}s  {batches_per_sec:.1f}bat/s")

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "epoch_seconds": round(epoch_time, 2),
            "samples_per_second": round(samples_per_sec, 2),
            "batches_per_second": round(batches_per_sec, 2),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if torch.cuda.is_available():
            log_entry["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 2)
            log_entry["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 2)
        training_log.append(log_entry)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save checkpoint only when validation loss improves
        if val_loss < best_val:
            best_val = val_loss
            # Collect norm stats for checkpoint
            norm_stats = {
                "method": normalizer.method,
            }
            if hasattr(normalizer, "min_") and normalizer.min_ is not None:
                norm_stats["min_"] = normalizer.min_.tolist()
                norm_stats["max_"] = normalizer.max_.tolist()
            if hasattr(normalizer, "mean_") and normalizer.mean_ is not None:
                norm_stats["mean_"] = normalizer.mean_.tolist()
                norm_stats["std_"] = normalizer.std_.tolist()
            save_checkpoint(ckpt_dir / "best_autoencoder.pt", {
                "epoch": epoch,
                "enc_state_dict": enc.state_dict(),
                "dec_state_dict": dec.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
                "norm_stats": norm_stats,
                "git_hash": git_hash,
                "env_info": env_info,
                "total_params": total_params,
                "trainable_params": trainable_params,
            })

    total_train_time = time.perf_counter() - train_start
    summary = {
        "stage": "autoencoder",
        "total_training_seconds": round(total_train_time, 2),
        "average_epoch_seconds": round(total_train_time / epochs, 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "git_hash": git_hash,
        "env": env_info,
    }
    training_log.append({"summary": summary})

    # Save training log
    with open(ckpt_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Save config copy
    import shutil
    shutil.copy2(args.config, ckpt_dir / "config.yaml")

    print(f"[AE] Training complete. Best val loss: {best_val:.6f}")
    print(f"     Total training time: {total_train_time:.2f}s")


if __name__ == "__main__":
    main()
