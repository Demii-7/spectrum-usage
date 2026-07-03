"""
Stage 2 training: TSS Condition Constructor (TSS-CC).

Trains the transformer-based conditioner to predict the latent code
z = enc(y) from the input window x. The frozen autoencoder encoder
provides the latent target. The trained TSS-CC later conditions the
diffusion model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import LatentSpaceEncoder, TSSConditionConstructor
from utils import load_config, set_seed, get_device, save_checkpoint, load_checkpoint


def main():
    """Entry point: load frozen autoencoder, train TSS-CC, save best checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--autoencoder_checkpoint", type=str, required=True,
                        help="Path to trained autoencoder checkpoint")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config = load_config(args.config)
    set_seed(config.get("seed"))
    device = get_device(config["device"]["device"])
    train_loader, val_loader, _, _, L, F, T_out = get_dataloaders(config)

    train_cfg = config["training"]
    model_cfg = config["model"]
    ckpt_dir = script_dir / train_cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    objective = train_cfg.get("tss_condition_objective", "projection_to_latent")
    if objective == "repo_context_ae":
        raise NotImplementedError(
            "repo_context_ae is not implemented; use projection_to_latent"
        )

    # Load and freeze the trained autoencoder encoder
    ae_checkpoint = load_checkpoint(args.autoencoder_checkpoint, map_location=device)
    enc = LatentSpaceEncoder(
        T_out=T_out, L=L, F=F,
        latent_dim=model_cfg["latent_dim"],
        num_blocks=model_cfg.get("autoencoder_num_blocks", 3),
        init_channels=model_cfg.get("autoencoder_initial_channels", 32),
    ).to(device)
    enc.load_state_dict(ae_checkpoint["enc_state_dict"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    T_in = config["windowing"]["input_sequence_length"]
    tss_cc = TSSConditionConstructor(
        T_in=T_in, L=L, F=F,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_heads=model_cfg.get("attention_heads", 4),
        num_layers=model_cfg.get("num_attention_layers", 2),
        ffn_dim=model_cfg.get("ffn_dim", 1024),
        dropout=model_cfg.get("dropout", 0.1),
        latent_dim=model_cfg["latent_dim"],
        use_temporal=model_cfg.get("use_temporal_branch", True),
        use_spectral=model_cfg.get("use_spectral_branch", True),
        use_spatial=model_cfg.get("use_spatial_branch", True),
    ).to(device)

    opt_name = train_cfg.get("optimizer", "adam")
    lr = train_cfg.get("tss_learning_rate", 0.0001)
    wd = train_cfg.get("weight_decay", 0.0)
    if opt_name == "adam":
        optimizer = optim.Adam(tss_cc.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(tss_cc.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    clip = train_cfg.get("gradient_clip", 0.0)
    best_val = float("inf")
    epochs = train_cfg.get("tss_epochs", 200)

    for epoch in range(1, epochs + 1):
        tss_cc.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Use frozen encoder to produce latent target
            with torch.no_grad():
                z_target = enc(y)
            z_pred = tss_cc(x)
            loss = nn.functional.mse_loss(z_pred, z_target)
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(tss_cc.parameters(), clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        tss_cc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                z_target = enc(y)
                z_pred = tss_cc(x)
                loss = nn.functional.mse_loss(z_pred, z_target)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f"[TSS] Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best_tss_condition.pt", {
                "epoch": epoch,
                "tss_cc_state_dict": tss_cc.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            })

    print(f"[TSS] Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
