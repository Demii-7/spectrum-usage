from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import (
    LatentSpaceEncoder,
    LatentSpaceDecoder,
    TSSConditionConstructor,
    DiffusionModel,
)
from training.common.config import load_config
from training.common.runtime import timestamp_utc
from training.common.results import append_metric_rows, finalize_results, load_band_definitions, prepare_output_dirs
from training.common.data import chunk_specs, load_chunk
from training.common.metrics import absolute_and_squared_errors_dbm
from training.common.windowing import make_window_starts


MODEL_NAME = "tss_lcd"


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_models(config: dict[str, Any], t_in: int, t_out: int,
                 n_bins: int, device: torch.device,
                 enc, dec, tss_cc, diffusion):
    tcfg = config["tss_lcd"]
    L = 1
    F = n_bins

    if enc is None:
        enc = LatentSpaceEncoder(
            T_out=t_out, L=L, F=F,
            latent_dim=tcfg["latent_dim"],
            num_blocks=tcfg.get("autoencoder_num_blocks", 3),
            init_channels=tcfg.get("autoencoder_initial_channels", 32),
        ).to(device)
    if dec is None:
        dec = LatentSpaceDecoder(
            T_out=t_out, L=L, F=F,
            latent_dim=tcfg["latent_dim"],
            num_blocks=tcfg.get("autoencoder_num_blocks", 3),
            init_channels=tcfg.get("autoencoder_initial_channels", 32),
        ).to(device)
    if tss_cc is None:
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
    if diffusion is None:
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


def generate_full_predictions(tss_cc, diffusion, dec, device,
                              full_x: np.ndarray, target_origins: np.ndarray,
                              t_in: int, t_out: int,
                              batch_size: int) -> np.ndarray:
    starts = target_origins - t_out + 1 - t_in
    n = len(starts)
    if n == 0:
        return np.empty((0, t_out, full_x.shape[1]))
    all_preds = []
    for i in range(0, n, batch_size):
        batch_starts = starts[i:i + batch_size]
        x_batch = np.stack([full_x[s:s + t_in] for s in batch_starts], axis=0)
        x_t = torch.from_numpy(x_batch).float().to(device)
        with torch.no_grad():
            cond_z = tss_cc(x_t)
            z_sample = diffusion.p_sample_loop(cond_z)
            y_hat = dec(z_sample)
        all_preds.append(y_hat.cpu().numpy())
    return np.concatenate(all_preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands, out: Path,
                   ae_checkpoint: Path, tss_checkpoint: Path, diff_checkpoint: Path):
    tcfg = config["tss_lcd"]
    t_in = int(config["windowing"]["lookback"])
    max_horizon = max(int(h) for h in config["windowing"]["horizons"])
    t_out = max_horizon
    batch_size = int(tcfg["batch_size"])
    horizons = [int(h) for h in config["windowing"]["horizons"]]

    data = load_chunk(config, chunk)
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm
    n_bins = train.shape[1]
    device = device_for()

    # Load component checkpoints
    ae_ckpt = torch.load(ae_checkpoint, map_location="cpu", weights_only=False)
    tss_ckpt = torch.load(tss_checkpoint, map_location="cpu", weights_only=False)
    diff_ckpt = torch.load(diff_checkpoint, map_location="cpu", weights_only=False)

    enc, dec, tss_cc, diffusion = build_models(config, t_in, t_out, n_bins, device, None, None, None, None)
    if ae_ckpt["state"] is not None:
        enc.load_state_dict(ae_ckpt["state"]["enc"])
        dec.load_state_dict(ae_ckpt["state"]["dec"])
    tss_cc.load_state_dict(tss_ckpt["tss_cc_state_dict"])
    diffusion.load_state_dict(diff_ckpt["diffusion_state_dict"])

    enc.eval()
    dec.eval()
    tss_cc.eval()
    diffusion.eval()

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for split_name in test_splits:
        split = data.splits[split_name]
        split_x = split.model_input.astype(np.float32)
        split_raw = split.raw_dbm.astype(np.float32)
        starts = make_window_starts(
            n_timesteps=len(split_x),
            lookback=t_in,
            rollout_horizon=max_horizon,
            stride=1,
            segments=split.segments,
        )
        if len(starts) == 0:
            print(f"  No valid target rows for {chunk.chunk_id} {split_name}")
            continue

        target_origins = starts + t_in + max_horizon - 1
        y_hat = generate_full_predictions(
            tss_cc, diffusion, dec, device,
            split_x, target_origins, t_in, t_out, batch_size,
        )
        target = np.stack(
            [split_raw[o - max_horizon + 1:o + 1] for o in target_origins],
            axis=0,
        ).astype(np.float32)

        for horizon in horizons:
            pred_h = y_hat[:, horizon - 1, :]
            target_h = target[:, horizon - 1, :]
            _, abs_err, sq_err = absolute_and_squared_errors_dbm(
                pred_h, target_h, data.normalization,
            )
            append_metric_rows(
                aggregate_rows, frequency_rows, band_rows,
                chunk_id=chunk.chunk_id,
                start_mhz=chunk.start_mhz,
                end_mhz=chunk.end_mhz,
                split_name=split_name,
                horizon=horizon,
                model=MODEL_NAME,
                target_rows=target_origins + int(split.row_start),
                history_offset=int(split.row_start),
                freqs=data.frequencies,
                abs_err=abs_err,
                sq_err=sq_err,
                bands=bands,
            )

    return aggregate_rows, frequency_rows, band_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TSS-LCD checkpoint suite")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ae-checkpoint", type=Path, default=None, help="Autoencoder checkpoint (use {chunk_id})")
    parser.add_argument("--tss-checkpoint", type=Path, default=None, help="TSS-CC checkpoint (use {chunk_id})")
    parser.add_argument("--diff-checkpoint", type=Path, default=None, help="Diffusion checkpoint (use {chunk_id})")
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
    bands = load_band_definitions(config)

    total_start_time = timestamp_utc()
    total_start = time.perf_counter()
    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for chunk in chunk_specs(config):
        print(f"Evaluating TSS-LCD for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        ckpt_dir = out / "checkpoints"

        def _resolve(flag_val, suffix):
            if flag_val:
                return Path(str(flag_val).replace("{chunk_id}", chunk.chunk_id))
            return ckpt_dir / f"{chunk.chunk_id}_{suffix}"

        ae_path = _resolve(args.ae_checkpoint, "tss_lcd_autoencoder.pt")
        tss_path = _resolve(args.tss_checkpoint, "tss_lcd_tss.pt")
        diff_path = _resolve(args.diff_checkpoint, "tss_lcd_diffusion.pt")

        missing = [p for p in [ae_path, tss_path, diff_path] if not p.exists()]
        if missing:
            print(f"  Missing checkpoint(s): {[str(m) for m in missing]}, skipping {chunk.chunk_id}")
            continue

        a, f, b = evaluate_chunk(config, chunk, bands, out, ae_path, tss_path, diff_path)
        aggregate_rows.extend(a)
        frequency_rows.extend(f)
        band_rows.extend(b)

    total_run = time.perf_counter() - total_start
    finalize_results(
        out,
        "TSS-LCD",
        aggregate_rows,
        frequency_rows,
        band_rows,
        [f"Evaluation start time: {total_start_time}", f"Total run time seconds: {total_run:.2f}"],
    )
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
