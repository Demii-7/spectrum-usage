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
from train_integrated import config_sections, set_deterministic_seed


MODEL_NAME = "tss_lcd"


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_models(config: dict[str, Any], t_in: int, t_out: int,
                 n_bins: int, device: torch.device,
                 enc, dec, tss_cc, diffusion):
    tcfg, _ = config_sections(config)
    L = 1
    F = n_bins

    if enc is None:
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
    if dec is None:
        dec = LatentSpaceDecoder(
            T_out=t_out, L=L, F=F,
            latent_dim=tcfg["latent_dim"],
            num_blocks=tcfg.get("autoencoder_num_blocks", 3),
            init_channels=tcfg.get("autoencoder_initial_channels", 32),
            kernel_size=tcfg.get("autoencoder_kernel_size", 3),
            activation=tcfg.get("autoencoder_activation", "relu"),
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
            condition_proj_dim=tcfg.get("condition_proj_dim"),
            condition_strategy=tcfg.get("condition_strategy", "concat"),
            nen_activation=tcfg.get("nen_activation", "relu"),
            nen_normalization=tcfg.get("nen_normalization", "batchnorm"),
        ).to(device)
    return enc, dec, tss_cc, diffusion


def generate_full_predictions(tss_cc, diffusion, dec, device,
                               full_x: np.ndarray, target_origins: np.ndarray,
                               t_in: int, t_out: int,
                               batch_size: int, mask_config: dict | None = None,
                               sampler_steps: int = 50, sampler_seed: int = 42) -> np.ndarray:
    starts = target_origins - t_out + 1 - t_in
    n = len(starts)
    if n == 0:
        return np.empty((0, t_out, full_x.shape[1]))
    all_preds = []
    generator = torch.Generator(device=device).manual_seed(sampler_seed)
    for i in range(0, n, batch_size):
        batch_starts = starts[i:i + batch_size]
        x_batch = np.stack([full_x[s:s + t_in] for s in batch_starts], axis=0)
        mask_config = mask_config or {}
        missing_rate = 0.0 if mask_config.get("complete_observation_baseline", False) else float(mask_config.get("missing_rate", 0.0))
        if missing_rate:
            if not mask_config.get("zero_pad_missing", True):
                raise ValueError("TSS-LCD missing observations must use zero padding")
            from dataset import create_masks
            x_batch = x_batch * create_masks(
                x_batch, missing_rate, str(mask_config.get("masking_strategy", "random")),
                mask_config.get("continuous_mask_length"),
                bool(mask_config.get("continuous_shared_gap", False)),
                bool(mask_config.get("continuous_multiple_gaps", False)),
            )
        x_t = torch.from_numpy(x_batch).float().unsqueeze(2).to(device)
        with torch.no_grad():
            cond_z = tss_cc(x_t)
            z_sample = diffusion.ddim_sample_loop(cond_z, sampler_steps, generator)
            y_hat = dec(z_sample)
        all_preds.append(y_hat.squeeze(2).cpu().numpy())
    return np.concatenate(all_preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands, out: Path,
                   checkpoint_path: Path):
    model_cfg, train_cfg = config_sections(config)
    seed = int(train_cfg.get("seed", config.get("seed", 42)))
    set_deterministic_seed(seed)
    t_in = int(config["windowing"].get("lookback", config["windowing"].get("input_sequence_length")))
    configured_horizons = config["windowing"].get("horizons")
    max_horizon = max(map(int, configured_horizons)) if configured_horizons else int(config["windowing"]["prediction_horizon"])
    t_out = max_horizon
    batch_size = int(train_cfg.get("batch_size", model_cfg.get("batch_size", 32)))
    horizons = [int(h) for h in (configured_horizons or [max_horizon])]

    data = load_chunk(config, chunk, val_fraction=float(train_cfg.get("val_fraction", 0.1)))
    test_splits = config["data"].get("test_splits", [data.test_split])
    train = data.splits[data.train_split].model_input
    train_raw = data.splits[data.train_split].raw_dbm
    n_bins = train.shape[1]
    device = device_for()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError(f"Checkpoint model_name is not {MODEL_NAME!r}")
    checkpoint_frequencies = np.asarray(checkpoint.get("frequencies"), dtype=np.float32)
    if checkpoint_frequencies.shape != np.asarray(data.frequencies).shape or not np.allclose(
        checkpoint_frequencies, data.frequencies, rtol=0, atol=1e-6
    ):
        raise ValueError("Checkpoint frequencies do not match evaluation data")
    left, right = checkpoint.get("normalization"), data.normalization
    if (left is None) != (right is None) or (
        left is not None and any(
            not np.allclose(np.asarray(left[key]), np.asarray(right[key])) for key in ("mean_dbm", "std_dbm")
        )
    ):
        raise ValueError("Checkpoint normalization does not match evaluation data")

    checkpoint_config = dict(config)
    checkpoint_config.pop("model", None)
    checkpoint_config[MODEL_NAME] = dict(checkpoint["model_config"])
    enc, dec, tss_cc, diffusion = build_models(checkpoint_config, t_in, t_out, n_bins, device, None, None, None, None)
    states = checkpoint["component_states"]
    enc.load_state_dict(states["encoder"])
    dec.load_state_dict(states["decoder"])
    tss_cc.load_state_dict(states["tss_condition"])
    diffusion.load_state_dict(states["diffusion"])

    enc.eval()
    dec.eval()
    tss_cc.eval()
    diffusion.eval()
    sampler = checkpoint.get("evaluation_sampler", {
        "name": "ddim",
        "steps": int(model_cfg.get("evaluation_sampler_steps", 50)),
        "seed": seed + int(model_cfg.get("evaluation_sampler_seed_offset", 10_000)),
    })
    if sampler.get("name") != "ddim":
        raise ValueError(f"Unsupported TSS-LCD evaluation sampler: {sampler.get('name')!r}")

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
            stride=int(train_cfg.get("test_stride", 1)),
            segments=split.segments,
        )
        if len(starts) == 0:
            print(f"  No valid target rows for {chunk.chunk_id} {split_name}")
            continue

        target_origins = starts + t_in + max_horizon - 1
        y_hat = generate_full_predictions(
            tss_cc, diffusion, dec, device,
            split_x, target_origins, t_in, t_out, batch_size,
            checkpoint.get("preprocessing", config.get("preprocessing")),
            sampler_steps=int(sampler["steps"]), sampler_seed=int(sampler["seed"]),
        )
        target = np.stack(
            [split_raw[o - max_horizon + 1:o + 1] for o in target_origins],
            axis=0,
        ).astype(np.float32)

        for horizon in horizons:
            pred_h = y_hat[:, horizon - 1, :]
            target_h = target[:, horizon - 1, :]
            target_rows = starts + t_in + horizon - 1
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
                target_rows=target_rows + int(split.row_start),
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
    parser.add_argument("--checkpoint", type=Path, default=None, help="Final checkpoint (use {chunk_id})")
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

        checkpoint_path = (
            Path(str(args.checkpoint).replace("{chunk_id}", chunk.chunk_id))
            if args.checkpoint else ckpt_dir / f"{chunk.chunk_id}_tss_lcd.pt"
        )
        if not checkpoint_path.exists():
            print(f"  Missing checkpoint: {checkpoint_path}, skipping {chunk.chunk_id}")
            continue

        a, f, b = evaluate_chunk(config, chunk, bands, out, checkpoint_path)
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
