from __future__ import annotations

import argparse
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
from training.common.config import load_config  # noqa: E402
from training.common.integrated import epoch_log_row, prepare_output_dirs, timestamp_utc  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.windowing import make_window_starts  # noqa: E402

MODEL_NAME = "tss_lcd"


class TSSLCDWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, starts: np.ndarray,
                 t_in: int, t_out: int):
        self.X = torch.from_numpy(
            np.stack([data[s:s + t_in] for s in starts], axis=0)
        ).float()
        self.Y = torch.from_numpy(
            np.stack([data[s + t_in:s + t_in + t_out] for s in starts], axis=0)
        ).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_models(config: dict[str, Any], t_in: int, t_out: int,
                 n_bins: int, device: torch.device):
    tcfg = config["tss_lcd"]
    L = 1
    F = n_bins

    enc = LatentSpaceEncoder(
        T_out=t_out, L=L, F=F,
        latent_dim=tcfg["latent_dim"],
        num_blocks=tcfg.get("autoencoder_num_blocks", 3),
        init_channels=tcfg.get("autoencoder_initial_channels", 32),
    ).to(device)
    dec = LatentSpaceDecoder(
        T_out=t_out, L=L, F=F,
        latent_dim=tcfg["latent_dim"],
        num_blocks=tcfg.get("autoencoder_num_blocks", 3),
        init_channels=tcfg.get("autoencoder_initial_channels", 32),
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
    ).to(device)
    return enc, dec, tss_cc, diffusion


def build_dataloaders(train_matrix: np.ndarray, t_in: int, t_out: int,
                      batch_size: int, val_fraction: float = 0.1,
                      segments=()):
    all_starts = make_window_starts(
        len(train_matrix), t_in, t_out, 1, segments
    )
    if len(all_starts) < 10:
        raise ValueError(
            f"Not enough windows ({len(all_starts)}) "
            f"for t_in={t_in}, t_out={t_out}."
        )
    val_count = max(1, int(len(all_starts) * val_fraction))
    train_starts = all_starts[:-val_count]
    val_starts = all_starts[-val_count:]

    train_loader = DataLoader(
        TSSLCDWindowDataset(train_matrix, train_starts, t_in, t_out),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TSSLCDWindowDataset(train_matrix, val_starts, t_in, t_out),
        batch_size=batch_size, shuffle=False,
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


def train_autoencoder(enc, dec, train_loader, val_loader,
                      tcfg: dict, device: torch.device,
                      checkpoints: Path, out: Path, chunk_id: str):
    epochs = int(tcfg["autoencoder_epochs"])
    lr = float(tcfg["autoencoder_learning_rate"])
    clip_norm = float(tcfg.get("gradient_clip_norm", 5.0))
    patience = int(tcfg.get("patience", 30))

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

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                "enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                "dec": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.perf_counter() - t_start
    print(f"{chunk_id} ae training done in {total_time:.1f}s ({total_time/60:.1f} min)")
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_ae_training_log.csv", index=False)
    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        dec.load_state_dict(best_state["dec"])
    torch.save(
        {
            "state": best_state,
            "training_start_time": training_start_time,
            "training_end_time": timestamp_utc(),
            "training_duration_sec": total_time,
        },
        checkpoints / f"{chunk_id}_tss_lcd_autoencoder.pt",
    )
    return enc, dec


def train_tss_condition(enc, tss_cc, train_loader, val_loader,
                        tcfg: dict, device: torch.device,
                        checkpoints: Path, out: Path, chunk_id: str):
    epochs = int(tcfg["tss_epochs"])
    lr = float(tcfg["tss_learning_rate"])
    clip_norm = float(tcfg.get("gradient_clip_norm", 5.0))
    patience = int(tcfg.get("patience", 30))

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

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in tss_cc.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.perf_counter() - t_start
    print(f"{chunk_id} tss training done in {total_time:.1f}s ({total_time/60:.1f} min)")
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_tss_training_log.csv", index=False)
    if best_state is not None:
        tss_cc.load_state_dict(best_state)
    torch.save(
        {
            "tss_cc_state_dict": best_state,
            "training_start_time": training_start_time,
            "training_end_time": timestamp_utc(),
            "training_duration_sec": total_time,
        },
        checkpoints / f"{chunk_id}_tss_lcd_tss.pt",
    )
    return tss_cc


def train_diffusion(enc, tss_cc, diffusion, train_loader, val_loader,
                    tcfg: dict, device: torch.device,
                    checkpoints: Path, out: Path, chunk_id: str):
    epochs = int(tcfg["diffusion_epochs"])
    lr = float(tcfg["diffusion_learning_rate"])
    clip_norm = float(tcfg.get("gradient_clip_norm", 5.0))
    patience = int(tcfg.get("patience", 30))

    optimizer = torch.optim.Adam(
        diffusion.parameters(), lr=lr,
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()
    enc.eval()
    tss_cc.eval()
    n_timestep = diffusion.n_timestep

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
        print(f"{chunk_id} diff epoch {epoch:03d}/{epochs} "
              f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"time={t_epoch:.1f}s avg={avg_time:.1f}s eta={eta:.0f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in diffusion.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.perf_counter() - t_start
    print(f"{chunk_id} diff training done in {total_time:.1f}s ({total_time/60:.1f} min)")
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_diff_training_log.csv", index=False)
    if best_state is not None:
        diffusion.load_state_dict(best_state)
    torch.save({
        "diffusion_state_dict": best_state,
        "tss_cc_state_dict": tss_cc.state_dict(),
        "training_start_time": training_start_time,
        "training_end_time": timestamp_utc(),
        "training_duration_sec": total_time,
    }, checkpoints / f"{chunk_id}_tss_lcd_diffusion.pt")
    return diffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out, checkpoints = prepare_output_dirs(config, "TSS-LCD")
    if args.output_dir is not None:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        checkpoints = out / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)

    for chunk in chunk_specs(config):
        print(f"Training TSS-LCD for {chunk.chunk_id} "
              f"({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        chunk_start = time.perf_counter()
        data = load_chunk(config, chunk)
        train = data.splits[data.train_split].model_input
        train_raw = data.splits[data.train_split].raw_dbm
        n_bins = train.shape[1]
        t_in = int(config["windowing"]["lookback"])
        max_horizon = max(int(h) for h in config["windowing"]["horizons"])
        t_out = max_horizon
        batch_size = int(config["tss_lcd"]["batch_size"])
        tcfg = config["tss_lcd"]
        device = device_for()

        enc, dec, tss_cc, diffusion = build_models(config, t_in, t_out, n_bins, device)
        train_loader, val_loader = build_dataloaders(
            train, t_in, t_out, batch_size,
            segments=data.splits[data.train_split].segments,
        )

        print(f"  {chunk.chunk_id} training autoencoder...")
        enc, dec = train_autoencoder(enc, dec, train_loader, val_loader,
                                     tcfg, device, checkpoints, out, chunk.chunk_id)

        print(f"  {chunk.chunk_id} training TSS-CC...")
        tss_cc = train_tss_condition(enc, tss_cc, train_loader, val_loader,
                                     tcfg, device, checkpoints, out, chunk.chunk_id)

        print(f"  {chunk.chunk_id} training diffusion...")
        diffusion = train_diffusion(enc, tss_cc, diffusion, train_loader, val_loader,
                                    tcfg, device, checkpoints, out, chunk.chunk_id)

        print(f"  {chunk.chunk_id} total done in {time.perf_counter() - chunk_start:.1f}s")


if __name__ == "__main__":
    main()
