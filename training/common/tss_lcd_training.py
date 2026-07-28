"""
Dedicated three-stage trainer for TSS-LCD used by the shared pipeline.

TSS-LCD cannot use the generic train_model() because it has three sequential
training objectives with separate optimizers and validation losses.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from training.common.runtime import (
    device_for,
    epoch_log_row,
    timestamp_utc,
)
from training.common.windowing import build_window_loaders


def train_autoencoder_stage(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Stage 1: train encoder+decoder to reconstruct future windows."""

    lr = float(train_cfg.get("autoencoder_learning_rate", 0.0001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg.get("autoencoder_epochs", 300))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 5.0))
    early_stopping = bool(train_cfg.get("early_stopping", True))
    patience = int(train_cfg.get("early_stopping_patience", 30))

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    training_log: list[dict[str, Any]] = []

    best_encoder_state = None
    best_decoder_state = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        epoch_start_counter = time.perf_counter()

        encoder.train()
        decoder.train()
        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_x, batch_y in train_loader:
            batch_y = batch_y.to(device, non_blocking=True)

            z_target = encoder(batch_y)
            y_hat = decoder(z_target)
            loss = nn.functional.mse_loss(y_hat, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                clip_norm,
            )
            optimizer.step()

            batch_samples = batch_y.size(0)
            train_loss_sum += loss.item() * batch_samples
            train_sample_count += batch_samples

        avg_train_loss = train_loss_sum / train_sample_count

        encoder.eval()
        decoder.eval()
        val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_y = batch_y.to(device, non_blocking=True)
                z_target = encoder(batch_y)
                y_hat = decoder(z_target)
                loss = nn.functional.mse_loss(y_hat, batch_y)
                batch_samples = batch_y.size(0)
                val_loss_sum += loss.item() * batch_samples
                val_sample_count += batch_samples

        avg_val_loss = val_loss_sum / val_sample_count

        epoch_duration = time.perf_counter() - epoch_start_counter

        training_log.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=timestamp_utc(),
                epoch_duration_sec=epoch_duration,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_encoder_state = {
                name: value.detach().cpu().clone()
                for name, value in encoder.state_dict().items()
            }
            best_decoder_state = {
                name: value.detach().cpu().clone()
                for name, value in decoder.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            break

    if best_encoder_state is None or best_decoder_state is None:
        raise RuntimeError(
            "Autoencoder training did not produce a best checkpoint."
        )

    encoder.load_state_dict(best_encoder_state)
    decoder.load_state_dict(best_decoder_state)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": epoch,
        "log_frame": pd.DataFrame(training_log),
        "best_checkpoint": {
            "encoder": best_encoder_state,
            "decoder": best_decoder_state,
        },
    }


def train_condition_stage(
    *,
    encoder: nn.Module,
    condition_constructor: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Stage 2: train TSS-CC to predict latent z_target from lookback x."""

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    lr = float(train_cfg.get("tss_learning_rate", 0.0001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg.get("tss_epochs", 200))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 5.0))
    early_stopping = bool(train_cfg.get("early_stopping", True))
    patience = int(train_cfg.get("early_stopping_patience", 30))

    optimizer = torch.optim.Adam(
        condition_constructor.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    training_log: list[dict[str, Any]] = []

    best_state = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        epoch_start_counter = time.perf_counter()

        condition_constructor.train()
        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with torch.no_grad():
                z_target = encoder(batch_y)

            z_pred = condition_constructor(batch_x)
            loss = nn.functional.mse_loss(z_pred, z_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                condition_constructor.parameters(), clip_norm
            )
            optimizer.step()

            batch_samples = batch_y.size(0)
            train_loss_sum += loss.item() * batch_samples
            train_sample_count += batch_samples

        avg_train_loss = train_loss_sum / train_sample_count

        condition_constructor.eval()
        val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                z_target = encoder(batch_y)
                z_pred = condition_constructor(batch_x)
                loss = nn.functional.mse_loss(z_pred, z_target)
                batch_samples = batch_y.size(0)
                val_loss_sum += loss.item() * batch_samples
                val_sample_count += batch_samples

        avg_val_loss = val_loss_sum / val_sample_count

        epoch_duration = time.perf_counter() - epoch_start_counter

        training_log.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=timestamp_utc(),
                epoch_duration_sec=epoch_duration,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in condition_constructor.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "Condition-constructor training did not produce a best checkpoint."
        )

    condition_constructor.load_state_dict(best_state)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": epoch,
        "log_frame": pd.DataFrame(training_log),
    }


def train_diffusion_stage(
    *,
    encoder: nn.Module,
    condition_constructor: nn.Module,
    diffusion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Stage 3: train diffusion to predict noise added to the latent z0."""

    encoder.eval()
    condition_constructor.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in condition_constructor.parameters():
        p.requires_grad = False

    lr = float(train_cfg.get("diffusion_learning_rate", 0.0001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg.get("diffusion_epochs", 1000))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 5.0))
    early_stopping = bool(train_cfg.get("early_stopping", True))
    patience = int(train_cfg.get("early_stopping_patience", 30))

    optimizer = torch.optim.Adam(
        diffusion.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    training_log: list[dict[str, Any]] = []

    best_state = None
    n_timestep = diffusion.n_timestep

    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        epoch_start_counter = time.perf_counter()

        diffusion.train()
        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with torch.no_grad():
                z0 = encoder(batch_y)
                cond_z = condition_constructor(batch_x)

            noise = torch.randn_like(z0)
            t = torch.randint(
                0, n_timestep, (z0.size(0),), device=device, dtype=torch.long
            )
            z_t = diffusion.q_sample(z0, t, noise)
            noise_pred = diffusion(z_t, cond_z, t)
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.parameters(), clip_norm)
            optimizer.step()

            batch_samples = batch_y.size(0)
            train_loss_sum += loss.item() * batch_samples
            train_sample_count += batch_samples

        avg_train_loss = train_loss_sum / train_sample_count

        diffusion.eval()
        val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                z0 = encoder(batch_y)
                cond_z = condition_constructor(batch_x)
                noise = torch.randn_like(z0)
                t = torch.randint(
                    0, n_timestep, (z0.size(0),), device=device, dtype=torch.long
                )
                z_t = diffusion.q_sample(z0, t, noise)
                noise_pred = diffusion(z_t, cond_z, t)
                loss = nn.functional.mse_loss(noise_pred, noise)
                batch_samples = batch_y.size(0)
                val_loss_sum += loss.item() * batch_samples
                val_sample_count += batch_samples

        avg_val_loss = val_loss_sum / val_sample_count

        epoch_duration = time.perf_counter() - epoch_start_counter

        training_log.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=timestamp_utc(),
                epoch_duration_sec=epoch_duration,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in diffusion.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "Diffusion training did not produce a best checkpoint."
        )

    diffusion.load_state_dict(best_state)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": epoch,
        "log_frame": pd.DataFrame(training_log),
    }


def train_tss_lcd(
    *,
    model: nn.Module,
    train_data: np.ndarray,
    config: dict[str, Any],
    segments=(),
) -> tuple[nn.Module, dict[str, Any]]:
    """Run all three TSS-LCD training stages on an already-built composite model."""

    model_cfg = config["tss_lcd"]["model"]
    train_cfg = config["tss_lcd"]["train"]

    input_sequence_length = int(model_cfg["input_sequence_length"])
    prediction_horizon = int(model_cfg["prediction_horizon"])

    device = device_for(config)
    model = model.to(device)

    # Build one shared pair of loaders for all three stages
    train_loader, val_loader = build_window_loaders(
        data=train_data,
        lookback=input_sequence_length,
        rollout_horizon=prediction_horizon,
        batch_size=int(train_cfg.get("batch_size", 32)),
        val_fraction=float(train_cfg.get("val_fraction", 0.1)),
        train_stride=int(train_cfg.get("train_stride", 1)),
        val_stride=int(train_cfg.get("val_stride", 1)),
        segments=segments,
        data_loader_config=config.get("data_loader"),
    )

    # Stage 1 -- Autoencoder
    print(f"  Stage 1/3: Autoencoder")
    autoencoder_results = train_autoencoder_stage(
        encoder=model.encoder,
        decoder=model.decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        device=device,
    )
    model.encoder.load_state_dict(autoencoder_results["best_checkpoint"]["encoder"])
    model.decoder.load_state_dict(autoencoder_results["best_checkpoint"]["decoder"])

    # Stage 2 -- TSS Condition Constructor
    print(f"  Stage 2/3: TSS Condition Constructor")
    condition_results = train_condition_stage(
        encoder=model.encoder,
        condition_constructor=model.condition_constructor,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        device=device,
    )

    # Stage 3 -- Diffusion
    print(f"  Stage 3/3: Diffusion")
    diffusion_results = train_diffusion_stage(
        encoder=model.encoder,
        condition_constructor=model.condition_constructor,
        diffusion=model.diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        device=device,
    )

    training_results = {
        "model_name": "tss_lcd",
        "autoencoder": autoencoder_results,
        "condition_constructor": condition_results,
        "diffusion": diffusion_results,
    }

    return model, training_results
