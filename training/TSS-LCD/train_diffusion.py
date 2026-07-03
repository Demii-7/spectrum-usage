from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import LatentSpaceEncoder, TSSConditionConstructor, DiffusionModel
from utils import load_config, set_seed, get_device, save_checkpoint, load_checkpoint


def main():
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

    ae_checkpoint = load_checkpoint(args.autoencoder_checkpoint, map_location=device)
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
    enc.load_state_dict(ae_checkpoint["enc_state_dict"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

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
        condition_proj_dim=model_cfg.get("condition_proj_dim", None),
        condition_strategy=model_cfg.get("condition_strategy", "concat"),
        nen_activation=model_cfg.get("nen_activation", "relu"),
        nen_normalization=model_cfg.get("nen_normalization", "batchnorm"),
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
    epochs = train_cfg.get("diffusion_epochs", 1000)

    # Scheduler
    scheduler = None
    lr_sched_name = train_cfg.get("lr_scheduler", "none")
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

    env_info = {
        "device": str(device),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)

    best_val = float("inf")
    training_log = []
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        diffusion.train()
        if not freeze_tss:
            tss_cc.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                z0 = enc(y)
                cond_z = tss_cc(x).detach() if freeze_tss else tss_cc(x)
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

        epoch_time = time.perf_counter() - epoch_start
        samples_per_sec = len(train_loader.dataset) / epoch_time if epoch_time > 0 else 0
        batches_per_sec = len(train_loader) / epoch_time if epoch_time > 0 else 0

        print(f"[DIF] Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}"
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

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best_diffusion.pt", {
                "epoch": epoch,
                "diffusion_state_dict": diffusion.state_dict(),
                "tss_cc_state_dict": tss_cc.state_dict() if not freeze_tss else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
                "git_hash": git_hash,
                "env_info": env_info,
                "total_params": total_params,
                "trainable_params": trainable_params,
            })

    total_train_time = time.perf_counter() - train_start
    summary = {
        "stage": "diffusion",
        "total_training_seconds": round(total_train_time, 2),
        "average_epoch_seconds": round(total_train_time / epochs, 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "git_hash": git_hash,
        "env": env_info,
    }
    training_log.append({"summary": summary})

    with open(ckpt_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    import shutil
    shutil.copy2(args.config, ckpt_dir / "config.yaml")

    print(f"[DIF] Training complete. Best val loss: {best_val:.6f}")
    print(f"     Total training time: {total_train_time:.2f}s")


if __name__ == "__main__":
    main()
