from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import get_dataloaders
from model import (
    LatentSpaceEncoder,
    LatentSpaceDecoder,
    TSSConditionConstructor,
    DiffusionModel,
)
from utils import (
    load_config,
    set_seed,
    get_device,
    load_checkpoint,
    compute_metrics,
    compute_metrics_per_horizon,
    compute_metrics_per_node,
    plot_spectrogram_comparison,
    plot_error_analysis,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to diffusion checkpoint (Stage 3)")
    parser.add_argument("--autoencoder_checkpoint", type=str, required=True)
    parser.add_argument("--tss_checkpoint", type=str, default=None,
                        help="TSS-CC checkpoint (not needed if saved in diffusion ckpt)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config = load_config(args.config)
    set_seed(config.get("seed"))
    device = get_device(config["device"]["device"])

    _, _, test_loader, normalizer, L, F, T_out = get_dataloaders(config)
    T_in = config["windowing"]["input_sequence_length"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    output_dir = script_dir / eval_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    n_bins_per_node = config["data"]["n_bins_per_node"]

    # Build models
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

    # Load weights
    ae_ckpt = load_checkpoint(args.autoencoder_checkpoint, map_location=device)
    enc.load_state_dict(ae_ckpt["enc_state_dict"])
    dec.load_state_dict(ae_ckpt["dec_state_dict"])

    diff_ckpt = load_checkpoint(args.checkpoint, map_location=device)
    diffusion.load_state_dict(diff_ckpt["diffusion_state_dict"])

    if args.tss_checkpoint is not None:
        tss_ckpt = load_checkpoint(args.tss_checkpoint, map_location=device)
        tss_cc.load_state_dict(tss_ckpt["tss_cc_state_dict"])
    elif diff_ckpt.get("tss_cc_state_dict") is not None:
        tss_cc.load_state_dict(diff_ckpt["tss_cc_state_dict"])

    enc.eval()
    dec.eval()
    tss_cc.eval()
    diffusion.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            cond_z = tss_cc(x)
            z_sample = diffusion.p_sample_loop(cond_z)
            y_hat = dec(z_sample)
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    pred = np.concatenate(all_preds, axis=0)
    target = np.concatenate(all_targets, axis=0)

    # Reshape to (B, T_out, D) for metric computation
    B_actual, T_actual, D_actual = pred.shape
    pred_flat = pred.reshape(B_actual * T_actual, D_actual)
    target_flat = target.reshape(B_actual * T_actual, D_actual)

    # Inverse normalize
    pred_dbm = normalizer.inverse_transform(pred_flat).reshape(B_actual, T_actual, D_actual)
    target_dbm = normalizer.inverse_transform(target_flat).reshape(B_actual, T_actual, D_actual)

    # Overall metrics
    overall = compute_metrics(pred_dbm.reshape(-1), target_dbm.reshape(-1))
    print("=== Overall Metrics (dBm) ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")

    with open(output_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)

    # Per-horizon metrics
    horizon_metrics = compute_metrics_per_horizon(pred_dbm, target_dbm)
    print("\n=== Per-Horizon Metrics (dBm) ===")
    rows = []
    for h in eval_cfg.get("eval_horizons", [1, 5, 10]):
        if h in horizon_metrics:
            m = horizon_metrics[h]
            print(f"  Horizon {h:2d}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")
            rows.append({"horizon": h, **m})
    with open(output_dir / "per_horizon_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Per-node metrics
    node_metrics = compute_metrics_per_node(pred_dbm, target_dbm, L)
    print("\n=== Per-Node Metrics (dBm) ===")
    for l in range(L):
        m = node_metrics[l]
        print(f"  Node {l}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    # Reshape to (B, T_out, L, F) for plotting (matching ConvLSTM convention)
    pred_4d = pred_dbm.reshape(B_actual, T_actual, L, -1)
    target_4d = target_dbm.reshape(B_actual, T_actual, L, -1)

    node_names = config["data"].get("node_names", [f"Node_{i}" for i in range(L)])
    errors = pred_4d - target_4d

    # Plot spectrogram for each node (first test sample, all horizons)
    for n, name in enumerate(node_names):
        plot_path = output_dir / f"spectrogram_{name}.png"
        plot_spectrogram_comparison(
            target_4d[0], pred_4d[0],
            n, name, config["windowing"]["input_sequence_length"], plot_path,
        )
        print(f"Spectrogram saved to {plot_path}")

    # Error analysis plot
    error_plot_path = output_dir / "error_analysis.png"
    plot_error_analysis(errors[0], node_names, error_plot_path)
    print(f"Error analysis saved to {error_plot_path}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
