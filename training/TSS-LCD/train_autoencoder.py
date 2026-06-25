from __future__ import annotations

import argparse
import os
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

    train_loader, val_loader, _, _, L, F, T_out = get_dataloaders(config)

    model_cfg = config["model"]
    train_cfg = config["training"]
    ckpt_dir = script_dir / train_cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    enc = LatentSpaceEncoder(
        T_out=T_out, L=L, F=F,
        latent_dim=model_cfg["latent_dim"],
        num_blocks=model_cfg.get("autoencoder_num_blocks", 3),
        init_channels=model_cfg.get("autoencoder_initial_channels", 32),
    ).to(device)
    dec = LatentSpaceDecoder(
        T_out=T_out, L=L, F=F,
        latent_dim=model_cfg["latent_dim"],
        num_blocks=model_cfg.get("autoencoder_num_blocks", 3),
        init_channels=model_cfg.get("autoencoder_initial_channels", 32),
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

    clip = train_cfg.get("gradient_clip", 0.0)
    best_val = float("inf")
    epochs = train_cfg.get("autoencoder_epochs", 300)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(enc, dec, train_loader, optimizer, device, clip)
        val_loss = validate(enc, dec, val_loader, device)
        print(f"[AE] Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best_autoencoder.pt", {
                "epoch": epoch,
                "enc_state_dict": enc.state_dict(),
                "dec_state_dict": dec.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            })

    print(f"[AE] Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
