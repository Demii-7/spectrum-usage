from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import (  # noqa: E402
    LatentSpaceEncoder,
    LatentSpaceDecoder,
    TSSConditionConstructor,
    DiffusionModel,
)
from dataset import create_masks  # noqa: E402
from training.common.config import load_config  # noqa: E402
from training.common.runtime import epoch_log_row, timestamp_utc  # noqa: E402
from training.common.results import prepare_output_dirs  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.data_loader import data_loader_kwargs  # noqa: E402
from training.common.windowing import make_window_starts  # noqa: E402
from training.common.training_events import TrainingCallback, emit_training_event  # noqa: E402
from training.common.metrics import denormalize  # noqa: E402

MODEL_NAME = "tss_lcd"


class TSSLCDWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, starts: np.ndarray,
                 t_in: int, t_out: int, mask_config: dict | None = None,
                 seed: int = 42):
        self.X = torch.from_numpy(
            np.stack([data[s:s + t_in] for s in starts], axis=0)
        ).float()
        self.Y = torch.from_numpy(
            np.stack([data[s + t_in:s + t_in + t_out] for s in starts], axis=0)
        ).float().unsqueeze(2)
        self.X = self.X.unsqueeze(2)  # Integrated vector data has one location.
        mask_config = mask_config or {}
        missing_rate = 0.0 if mask_config.get("complete_observation_baseline", False) else float(
            mask_config.get("missing_rate", 0.0)
        )
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError("missing_rate must be between 0 and 1")
        if missing_rate and not mask_config.get("zero_pad_missing", True):
            raise ValueError("TSS-LCD missing observations must use zero padding")
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            mask = create_masks(
                self.X.numpy(), missing_rate,
                str(mask_config.get("masking_strategy", "random")),
                mask_config.get("continuous_mask_length"),
                bool(mask_config.get("continuous_shared_gap", False)),
                bool(mask_config.get("continuous_multiple_gaps", False)),
            )
        finally:
            np.random.set_state(rng_state)
        self.observation_mask = torch.from_numpy(mask)
        self.X = self.X * self.observation_mask.float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_sections(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accept shared nested model/train config and the legacy flat section."""
    legacy = dict(config.get(MODEL_NAME) or {})
    flat = {key: value for key, value in legacy.items() if key not in {"model", "train", "training"}}
    model = dict(flat)
    model.update(legacy.get("model") or config.get("model") or {})
    train = dict(flat)
    train.update(legacy.get("train") or legacy.get("training") or config.get("train") or config.get("training") or {})
    return model, train


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_models(config: dict[str, Any], t_in: int, t_out: int,
                 n_bins: int, device: torch.device):
    tcfg, _ = config_sections(config)
    L = 1
    F = n_bins

    enc = LatentSpaceEncoder(
        T_out=t_out, L=L, F=F,
        latent_dim=tcfg["latent_dim"],
        num_blocks=tcfg.get("autoencoder_num_blocks", 3),
        init_channels=tcfg.get("autoencoder_initial_channels", 32),
        kernel_size=tcfg.get("autoencoder_kernel_size", 3),
        pool_kernel=tcfg.get("autoencoder_pool_kernel", 2),
        pool_stride=tcfg.get("autoencoder_pool_stride", 2),
        activation=tcfg.get("autoencoder_activation", "relu"),
    ).to(device)
    dec = LatentSpaceDecoder(
        T_out=t_out, L=L, F=F,
        latent_dim=tcfg["latent_dim"],
        num_blocks=tcfg.get("autoencoder_num_blocks", 3),
        init_channels=tcfg.get("autoencoder_initial_channels", 32),
        kernel_size=tcfg.get("autoencoder_kernel_size", 3),
        activation=tcfg.get("autoencoder_activation", "relu"),
    ).to(device)
    tss_cc = TSSConditionConstructor(
        T_in=t_in, L=L, F=F,
        hidden_dim=tcfg.get("hidden_dim", 256),
        num_heads=tcfg.get("attention_heads", 4),
        num_layers=tcfg.get("num_attention_layers", 2),
        ffn_dim=tcfg.get("ffn_dim", 1024),
        dropout=tcfg.get("dropout", 0.1),
        latent_dim=tcfg["latent_dim"],
        use_temporal=tcfg.get("use_temporal_branch", True),
        use_spectral=tcfg.get("use_spectral_branch", True),
        use_spatial=tcfg.get("use_spatial_branch", True),
    ).to(device)
    diffusion = DiffusionModel(
        latent_dim=tcfg["latent_dim"],
        n_timestep=tcfg.get("diffusion_steps", 1000),
        device=device,
        noise_schedule=tcfg.get("noise_schedule", "cosine"),
        nen_encoder_channels=tcfg.get("nen_encoder_channels", [64, 128]),
        nen_bottleneck_channels=tcfg.get("nen_bottleneck_channels", 256),
        nen_decoder_channels=tcfg.get("nen_decoder_channels", [128, 64]),
        nen_kernel_size=tcfg.get("nen_kernel_size", 3),
        time_embed_dim=tcfg.get("time_embed_dim", 32),
        condition_proj_dim=tcfg.get("condition_proj_dim"),
        condition_strategy=tcfg.get("condition_strategy", "concat"),
        nen_activation=tcfg.get("nen_activation", "relu"),
        nen_normalization=tcfg.get("nen_normalization", "batchnorm"),
    ).to(device)
    return enc, dec, tss_cc, diffusion


def build_dataloaders(train_matrix: np.ndarray, t_in: int, t_out: int,
                      batch_size: int, val_fraction: float = 0.1,
                      segments=(), data_loader_config=None,
                      validation_matrix: np.ndarray | None = None,
                      validation_segments=(), mask_config=None, seed: int = 42):
    all_starts = make_window_starts(
        len(train_matrix), t_in, t_out, 1, segments
    )
    if len(all_starts) < 10:
        raise ValueError(
            f"Not enough windows ({len(all_starts)}) "
            f"for t_in={t_in}, t_out={t_out}."
        )
    if validation_matrix is None:
        val_count = max(1, int(len(all_starts) * val_fraction))
        train_starts = all_starts[:-val_count]
        val_matrix = train_matrix
        val_starts = all_starts[-val_count:]
    else:
        train_starts = all_starts
        val_matrix = validation_matrix
        val_starts = make_window_starts(len(val_matrix), t_in, t_out, 1, validation_segments)
        if len(val_starts) == 0:
            raise ValueError("Explicit validation split contains no TSS-LCD windows")

    train_loader = DataLoader(
        TSSLCDWindowDataset(train_matrix, train_starts, t_in, t_out, mask_config, seed),
        batch_size=batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **data_loader_kwargs(data_loader_config),
    )
    val_loader = DataLoader(
        TSSLCDWindowDataset(val_matrix, val_starts, t_in, t_out, mask_config, seed + 1),
        batch_size=batch_size, shuffle=False,
        **data_loader_kwargs(data_loader_config),
    )
    return train_loader, val_loader


def make_test_loader(full_x: np.ndarray, test_start: int,
                     t_in: int, t_out: int, batch_size: int):
    starts = np.arange(test_start - t_in, len(full_x) - t_in - t_out + 1)
    if len(starts) == 0:
        return None, np.array([], dtype=np.int64), np.empty((0, t_out, full_x.shape[1]))
    ds = TSSLCDWindowDataset(full_x, starts, t_in, t_out)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, starts + t_in + t_out - 1, ds.Y.numpy()


def validation_forecast_metrics(dec, tss_cc, diffusion, val_loader, horizons: list[int],
                                normalization: dict[str, Any] | None, device: torch.device,
                                sampler_steps: int, sampler_seed: int) -> dict[str, float]:
    """Evaluate decoded validation forecasts in physical dB for Ray selection."""
    sums = {horizon: 0.0 for horizon in horizons}
    counts = {horizon: 0 for horizon in horizons}
    generator = torch.Generator(device=device).manual_seed(sampler_seed)
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            cond_z = tss_cc(x)
            y_hat = dec(diffusion.ddim_sample_loop(cond_z, sampler_steps, generator))
            pred = denormalize(y_hat.squeeze(2).cpu().numpy(), normalization)
            target = denormalize(y.squeeze(2).numpy(), normalization)
            for horizon in horizons:
                error = np.abs(pred[:, horizon - 1] - target[:, horizon - 1])
                sums[horizon] += float(error.sum())
                counts[horizon] += int(error.size)
    per_horizon = {
        horizon: sums[horizon] / max(counts[horizon], 1)
        for horizon in horizons
    }
    return {
        **{f"val_mae_db_t{horizon}": float(per_horizon[horizon]) for horizon in horizons},
        "val_mean_horizon_mae_db": float(np.mean(list(per_horizon.values()))),
    }


def train_autoencoder(enc, dec, train_loader, val_loader,
                      tcfg: dict, device: torch.device,
                      checkpoints: Path, out: Path, chunk_id: str,
                      callback: TrainingCallback | None = None,
                      epoch_offset: int = 0,
                      total_epochs: int | None = None):
    epochs = int(tcfg["autoencoder_epochs"])
    lr = float(tcfg["autoencoder_learning_rate"])
    clip_norm = float(tcfg.get("gradient_clip_norm", tcfg.get("gradient_clip", 5.0)))
    patience = int(tcfg.get("patience", tcfg.get("early_stopping_patience", 30)))

    params = list(enc.parameters()) + list(dec.parameters())
    optimizer = torch.optim.Adam(
        params, lr=lr, weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    epoch_times: list[float] = []
    log_rows: list[dict[str, Any]] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        t_epoch = time.perf_counter()
        enc.train()
        dec.train()
        train_loss = 0.0
        for _, y in train_loader:
            y = y.to(device)
            optimizer.zero_grad()
            z = enc(y)
            y_hat = dec(z)
            loss = criterion(y_hat, y)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(params, clip_norm)
            optimizer.step()
            train_loss += loss.item() * y.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        enc.eval()
        dec.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, y in val_loader:
                y = y.to(device)
                z = enc(y)
                y_hat = dec(z)
                val_loss += criterion(y_hat, y).item() * y.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        t_epoch = time.perf_counter() - t_epoch
        log_rows.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=timestamp_utc(),
                epoch_duration_sec=t_epoch,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        epoch_times.append(t_epoch)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta = avg_time * (epochs - epoch)
        print(f"{chunk_id} ae epoch {epoch:03d}/{epochs} "
              f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"time={t_epoch:.1f}s avg={avg_time:.1f}s eta={eta:.0f}s")

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_state = {
                "enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                "dec": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1

        emit_training_event(
            callback,
            model_name=MODEL_NAME,
            chunk_id=chunk_id,
            stage="autoencoder",
            epoch=epoch_offset + epoch,
            epochs=total_epochs or epochs,
            stage_epoch=epoch,
            stage_epochs=epochs,
            metrics={"train_loss": float(train_loss), "val_loss": float(val_loss)},
            selection_metric="val_loss",
            selection_mode="min",
            duration=t_epoch,
            prunable=False,
            is_best=is_best,
        )

        if not is_best:
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.perf_counter() - t_start
    print(f"{chunk_id} ae training done in {total_time:.1f}s ({total_time/60:.1f} min)")
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_ae_training_log.csv", index=False)
    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        dec.load_state_dict(best_state["dec"])
    return enc, dec


def train_tss_condition(enc, tss_cc, train_loader, val_loader,
                        tcfg: dict, device: torch.device,
                        checkpoints: Path, out: Path, chunk_id: str,
                        callback: TrainingCallback | None = None,
                        epoch_offset: int = 0,
                        total_epochs: int | None = None):
    epochs = int(tcfg["tss_epochs"])
    lr = float(tcfg["tss_learning_rate"])
    clip_norm = float(tcfg.get("gradient_clip_norm", tcfg.get("gradient_clip", 5.0)))
    patience = int(tcfg.get("patience", tcfg.get("early_stopping_patience", 30)))

    optimizer = torch.optim.Adam(
        tss_cc.parameters(), lr=lr,
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()
    enc.eval()

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    epoch_times: list[float] = []
    log_rows: list[dict[str, Any]] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        t_epoch = time.perf_counter()
        tss_cc.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z_target = enc(y)
            z_pred = tss_cc(x)
            loss = criterion(z_pred, z_target)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(tss_cc.parameters(), clip_norm)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        tss_cc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                z_target = enc(y)
                z_pred = tss_cc(x)
                val_loss += criterion(z_pred, z_target).item() * x.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        t_epoch = time.perf_counter() - t_epoch
        log_rows.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=timestamp_utc(),
                epoch_duration_sec=t_epoch,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        epoch_times.append(t_epoch)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta = avg_time * (epochs - epoch)
        print(f"{chunk_id} tss epoch {epoch:03d}/{epochs} "
              f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"time={t_epoch:.1f}s avg={avg_time:.1f}s eta={eta:.0f}s")

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in tss_cc.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        emit_training_event(
            callback,
            model_name=MODEL_NAME,
            chunk_id=chunk_id,
            stage="tss_condition",
            epoch=epoch_offset + epoch,
            epochs=total_epochs or epochs,
            stage_epoch=epoch,
            stage_epochs=epochs,
            metrics={"train_loss": float(train_loss), "val_loss": float(val_loss)},
            selection_metric="val_loss",
            selection_mode="min",
            duration=t_epoch,
            prunable=False,
            is_best=is_best,
        )

        if not is_best:
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.perf_counter() - t_start
    print(f"{chunk_id} tss training done in {total_time:.1f}s ({total_time/60:.1f} min)")
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_tss_training_log.csv", index=False)
    if best_state is not None:
        tss_cc.load_state_dict(best_state)
    return tss_cc


def train_diffusion(enc, dec, tss_cc, diffusion, train_loader, val_loader,
                    tcfg: dict, device: torch.device,
                    checkpoints: Path, out: Path, chunk_id: str,
                    callback: TrainingCallback | None = None,
                    epoch_offset: int = 0,
                    total_epochs: int | None = None, horizons: list[int] | None = None,
                    normalization: dict[str, Any] | None = None, sampler_steps: int = 50,
                    sampler_seed: int = 42):
    epochs = int(tcfg["diffusion_epochs"])
    lr = float(tcfg["diffusion_learning_rate"])
    clip_norm = float(tcfg.get("gradient_clip_norm", tcfg.get("gradient_clip", 5.0)))
    patience = int(tcfg.get("patience", tcfg.get("early_stopping_patience", 30)))

    optimizer = torch.optim.Adam(
        diffusion.parameters(), lr=lr,
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()
    enc.eval()
    dec.eval()
    tss_cc.eval()
    n_timestep = diffusion.n_timestep
    horizons = horizons or [1]

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    epoch_times: list[float] = []
    log_rows: list[dict[str, Any]] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        t_epoch = time.perf_counter()
        diffusion.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            optimizer.zero_grad()
            with torch.no_grad():
                z_target = enc(y)
                cond_z = tss_cc(x)
            t = torch.randint(0, n_timestep, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(z_target)
            z_t = diffusion.q_sample(z_target, t, noise)
            noise_pred = diffusion(z_t, cond_z, t)
            loss = criterion(noise_pred, noise)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(diffusion.parameters(), clip_norm)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        diffusion.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                B = x.size(0)
                z_target = enc(y)
                cond_z = tss_cc(x)
                t = torch.randint(0, n_timestep, (B,), device=device, dtype=torch.long)
                noise = torch.randn_like(z_target)
                z_t = diffusion.q_sample(z_target, t, noise)
                noise_pred = diffusion(z_t, cond_z, t)
                val_loss += criterion(noise_pred, noise).item() * y.size(0)
        val_loss /= max(len(val_loader.dataset), 1)
        forecast_metrics = validation_forecast_metrics(
            dec=dec, tss_cc=tss_cc, diffusion=diffusion, val_loader=val_loader,
            horizons=horizons, normalization=normalization, device=device,
            sampler_steps=sampler_steps, sampler_seed=sampler_seed,
        )

        t_epoch = time.perf_counter() - t_epoch
        log_row = epoch_log_row(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            epoch_start_time=epoch_start_time,
            epoch_end_time=timestamp_utc(),
            epoch_duration_sec=t_epoch,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        log_row.update(forecast_metrics)
        log_rows.append(log_row)
        epoch_times.append(t_epoch)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta = avg_time * (epochs - epoch)
        print(f"{chunk_id} diff epoch {epoch:03d}/{epochs} "
              f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"val_mae_db={forecast_metrics['val_mean_horizon_mae_db']:.4f} "
              f"time={t_epoch:.1f}s avg={avg_time:.1f}s eta={eta:.0f}s")

        selection_value = forecast_metrics["val_mean_horizon_mae_db"]
        is_best = np.isfinite(selection_value) and selection_value < best_loss
        if is_best:
            best_loss = selection_value
            best_state = {k: v.detach().cpu().clone()
                          for k, v in diffusion.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        emit_training_event(
            callback,
            model_name=MODEL_NAME,
            chunk_id=chunk_id,
            stage="diffusion",
            epoch=epoch_offset + epoch,
            epochs=total_epochs or epochs,
            stage_epoch=epoch,
            stage_epochs=epochs,
            metrics={"train_loss": float(train_loss), "val_loss": float(val_loss), **forecast_metrics},
            selection_metric="val_mean_horizon_mae_db",
            selection_mode="min",
            duration=t_epoch,
            prunable=True,
            is_best=is_best,
        )

        if not is_best:
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.perf_counter() - t_start
    print(f"{chunk_id} diff training done in {total_time:.1f}s ({total_time/60:.1f} min)")
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_diff_training_log.csv", index=False)
    if best_state is not None:
        diffusion.load_state_dict(best_state)
    return diffusion


def train_chunk(config: dict[str, Any], chunk, data, out: Path, checkpoints: Path,
                callback: TrainingCallback | None = None) -> Path:
    """Train all paper stages for one loaded chunk and write one final checkpoint."""
    model_cfg, train_cfg = config_sections(config)
    seed = int(train_cfg.get("seed", config.get("seed", 42)))
    set_deterministic_seed(seed)
    train_split = data.splits[data.train_split]
    validation_split = data.splits.get(data.validation_split)
    t_in = int(config["windowing"].get("lookback", config["windowing"].get("input_sequence_length")))
    horizons = config["windowing"].get("horizons")
    t_out = max(map(int, horizons)) if horizons else int(config["windowing"]["prediction_horizon"])
    batch_size = int(train_cfg.get("batch_size", model_cfg.get("batch_size", 32)))
    device = device_for()
    enc, dec, tss_cc, diffusion = build_models(config, t_in, t_out, train_split.model_input.shape[-1], device)
    train_loader, val_loader = build_dataloaders(
        train_split.model_input, t_in, t_out, batch_size,
        val_fraction=float(train_cfg.get("val_fraction", 0.1)), segments=train_split.segments,
        data_loader_config=config.get("data_loader"),
        validation_matrix=None if validation_split is None else validation_split.model_input,
        validation_segments=() if validation_split is None else validation_split.segments,
        mask_config={
            **dict(config.get("preprocessing") or {}),
            **dict((config.get(MODEL_NAME) or {}).get("preprocessing") or {}),
        }, seed=seed,
    )
    autoencoder_epochs = int(train_cfg["autoencoder_epochs"])
    tss_epochs = int(train_cfg["tss_epochs"])
    diffusion_epochs = int(train_cfg["diffusion_epochs"])
    total_epochs = autoencoder_epochs + tss_epochs + diffusion_epochs
    enc, dec = train_autoencoder(
        enc, dec, train_loader, val_loader, train_cfg, device, checkpoints, out,
        chunk.chunk_id, callback=callback, total_epochs=total_epochs,
    )
    tss_cc = train_tss_condition(
        enc, tss_cc, train_loader, val_loader, train_cfg, device, checkpoints, out,
        chunk.chunk_id, callback=callback, epoch_offset=autoencoder_epochs,
        total_epochs=total_epochs,
    )
    diffusion = train_diffusion(
        enc, dec, tss_cc, diffusion, train_loader, val_loader, train_cfg, device,
        checkpoints, out, chunk.chunk_id, callback=callback,
        epoch_offset=autoencoder_epochs + tss_epochs, total_epochs=total_epochs,
        horizons=[int(horizon) for horizon in (horizons or [t_out])],
        normalization=data.normalization,
        sampler_steps=int(model_cfg.get("evaluation_sampler_steps", 50)),
        sampler_seed=seed + int(model_cfg.get("evaluation_sampler_seed_offset", 10_000)),
    )
    path = checkpoints / f"{chunk.chunk_id}_tss_lcd.pt"
    torch.save({
        "model_name": MODEL_NAME,
        "component_states": {
            "encoder": enc.state_dict(), "decoder": dec.state_dict(),
            "tss_condition": tss_cc.state_dict(), "diffusion": diffusion.state_dict(),
        },
        "model_config": model_cfg,
        "train_config": train_cfg,
        "frequencies": np.asarray(data.frequencies, dtype=np.float32),
        "normalization": data.normalization,
        "preprocessing": {
            **dict(config.get("preprocessing") or {}),
            **dict((config.get(MODEL_NAME) or {}).get("preprocessing") or {}),
        },
        "mask_metadata": {
            **dict(config.get("preprocessing") or {}),
            **dict((config.get(MODEL_NAME) or {}).get("preprocessing") or {}),
            "padding_value": 0.0,
            "seed": seed,
        },
        "tensor_layout": "B,T,L,F", "t_in": t_in, "t_out": t_out,
        "evaluation_sampler": {
            "name": "ddim", "steps": int(model_cfg.get("evaluation_sampler_steps", 50)),
            "seed": seed + int(model_cfg.get("evaluation_sampler_seed_offset", 10_000)),
        },
    }, path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = "tss-lcd"
    if args.output_dir is not None:
        run_dir = args.output_dir
    else:
        exp_name = args.name or f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out, checkpoints = prepare_output_dirs(run_dir)

    for chunk in chunk_specs(config):
        print(f"Training TSS-LCD for {chunk.chunk_id} "
              f"({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        chunk_start = time.perf_counter()
        _, train_cfg = config_sections(config)
        data = load_chunk(
            config, chunk, val_fraction=float(train_cfg.get("val_fraction", 0.1))
        )
        train_chunk(config, chunk, data, out, checkpoints)

        print(f"  {chunk.chunk_id} total done in {time.perf_counter() - chunk_start:.1f}s")


if __name__ == "__main__":
    main()
