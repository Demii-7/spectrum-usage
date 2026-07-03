"""
Stage 3 training: Latent Conditional Diffusion.

Trains the diffusion model (EnhancedNoiseNet) to denoise latent vectors
conditioned on the TSS-CC output. The conditioner can be frozen
(standard) or jointly fine-tuned (joint_with_diffusion).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import LatentSpaceEncoder, TSSConditionConstructor, DiffusionModel
from utils import load_config, set_seed, get_device, save_checkpoint, load_checkpoint


def main():
    """Entry point: load frozen LSE + optionally frozen TSS-CC, train diffusion, save best."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--autoencoder_checkpoint", type=str, required=True,
                        help="Path to trained autoencoder checkpoint")
    parser.add_argument("--tss_checkpoint", type=str, default=None,
                        help="Path to trained TSS condition checkpoint (optional for joint)")
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
    T_in = config["windowing"]["input_sequence_length"]

    objective = train_cfg.get("tss_condition_objective", "projection_to_latent")

    # Load frozen LSE — provides latent targets z0 from spectrogram y
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

    # Load TSS-CC (optionally trainable for joint_with_diffusion objective)
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

    freeze_tss = True
    if args.tss_checkpoint is not None:
        ckpt = load_checkpoint(args.tss_checkpoint, map_location=device)
        tss_cc.load_state_dict(ckpt["tss_cc_state_dict"])
    if objective == "joint_with_diffusion":
        freeze_tss = False
    if freeze_tss:
        tss_cc.eval()
        for p in tss_cc.parameters():
            p.requires_grad = False
    else:
        tss_cc.train()

    # Diffusion model
    diffusion = DiffusionModel(
        latent_dim=model_cfg["latent_dim"],
        n_timestep=model_cfg.get("diffusion_steps", 1000),
        device=device,
        noise_schedule=model_cfg.get("noise_schedule", "cosine"),
        nen_encoder_channels=model_cfg.get("nen_encoder_channels", [64, 128]),
        nen_bottleneck_channels=model_cfg.get("nen_bottleneck_channels", 256),
        nen_decoder_channels=model_cfg.get("nen_decoder_channels", [128, 64]),
        nen_kernel_size=model_cfg.get("nen_kernel_size", 3),
        time_embed_dim=model_cfg.get("time_embed_dim", 32),
    ).to(device)

    params = list(diffusion.parameters())
    if not freeze_tss:
        params += list(tss_cc.parameters())

    opt_name = train_cfg.get("optimizer", "adam")
    lr = train_cfg.get("diffusion_learning_rate", 0.0001)
    wd = train_cfg.get("weight_decay", 0.0)
    if opt_name == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    clip = train_cfg.get("gradient_clip", 0.0)
    best_val = float("inf")
    epochs = train_cfg.get("diffusion_epochs", 1000)

    for epoch in range(1, epochs + 1):
        diffusion.train()
        if not freeze_tss:
            tss_cc.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                z0 = enc(y)
                # Detach condition when frozen to avoid gradient flow
                cond_z = tss_cc(x).detach() if freeze_tss else tss_cc(x)
            # Sample random timestep and noise, then apply forward diffusion
            t = torch.randint(0, diffusion.n_timestep, (x.size(0),), device=device)
            noise = torch.randn_like(z0)
            zt = diffusion.q_sample(z0, t, noise)
            noise_pred = diffusion(zt, cond_z, t)
            loss = nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        diffusion.eval()
        tss_cc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                z0 = enc(y)
                cond_z = tss_cc(x)
                t = torch.randint(0, diffusion.n_timestep, (x.size(0),), device=device)
                noise = torch.randn_like(z0)
                zt = diffusion.q_sample(z0, t, noise)
                noise_pred = diffusion(zt, cond_z, t)
                loss = nn.functional.mse_loss(noise_pred, noise)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f"[DIF] Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best_diffusion.pt", {
                "epoch": epoch,
                "diffusion_state_dict": diffusion.state_dict(),
                "tss_cc_state_dict": tss_cc.state_dict() if not freeze_tss else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            })

    print(f"[DIF] Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
