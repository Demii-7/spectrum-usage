from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import get_dataloaders, Normalizer
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
    compute_metrics_per_frequency,
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

    # Build normalizer from autoencoder checkpoint if available
    ae_ckpt = load_checkpoint(args.autoencoder_checkpoint, map_location=device)
    norm_stats = ae_ckpt.get("norm_stats")
    if norm_stats is not None:
        normalizer = Normalizer(method=norm_stats.get("method", "minmax"))
        if normalizer.method == "minmax":
            normalizer.min_ = np.array(norm_stats["min_"]).reshape(1, -1)
            normalizer.max_ = np.array(norm_stats["max_"]).reshape(1, -1)
        elif normalizer.method == "zscore":
            normalizer.mean_ = np.array(norm_stats["mean_"]).reshape(1, -1)
            normalizer.std_ = np.array(norm_stats["std_"]).reshape(1, -1)
        _, _, test_loader, _, L, F, T_out = get_dataloaders(config, normalizer=normalizer)
    else:
        _, _, test_loader, normalizer, L, F, T_out = get_dataloaders(config)

    T_in = config["windowing"]["input_sequence_length"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    output_dir = script_dir / eval_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    n_bins_per_node = config["data"]["n_bins_per_node"]

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
    dec = LatentSpaceDecoder(
        T_out=T_out, L=L, F=F,
        latent_dim=model_cfg["latent_dim"],
        num_blocks=model_cfg.get("autoencoder_num_blocks", 3),
        init_channels=model_cfg.get("autoencoder_initial_channels", 32),
        kernel_size=model_cfg.get("autoencoder_kernel_size", 3),
        activation=model_cfg.get("autoencoder_activation", "relu"),
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
        condition_proj_dim=model_cfg.get("condition_proj_dim", None),
        condition_strategy=model_cfg.get("condition_strategy", "concat"),
        nen_activation=model_cfg.get("nen_activation", "relu"),
        nen_normalization=model_cfg.get("nen_normalization", "batchnorm"),
    ).to(device)

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

    eval_start = time.perf_counter()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            cond_z = tss_cc(x)
            z_sample = diffusion.p_sample_loop(cond_z)
            y_hat = dec(z_sample)
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    total_eval_time = time.perf_counter() - eval_start

    pred = np.concatenate(all_preds, axis=0)
    target = np.concatenate(all_targets, axis=0)

    B_actual, T_actual, D_actual = pred.shape

    # Inverse normalize
    pred_flat = pred.reshape(B_actual * T_actual, D_actual)
    target_flat = target.reshape(B_actual * T_actual, D_actual)
    pred_dbm = normalizer.inverse_transform(pred_flat).reshape(B_actual, T_actual, D_actual)
    target_dbm = normalizer.inverse_transform(target_flat).reshape(B_actual, T_actual, D_actual)

    # Overall metrics
    overall = compute_metrics(pred_dbm.reshape(-1), target_dbm.reshape(-1))
    print("=== Overall Metrics (dBm) ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")

    # Per-horizon metrics
    horizon_metrics = compute_metrics_per_horizon(pred_dbm, target_dbm)
    print("\n=== Per-Horizon Metrics (dBm) ===")
    for h in eval_cfg.get("eval_horizons", [1, 5, 10]):
        if h in horizon_metrics:
            m = horizon_metrics[h]
            print(f"  Horizon {h:2d}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    # Per-node metrics
    node_metrics = compute_metrics_per_node(pred_dbm, target_dbm, L)
    print("\n=== Per-Node Metrics (dBm) ===")
    for l in range(L):
        m = node_metrics[l]
        print(f"  Node {l}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    # Per-frequency metrics
    freq_metrics = compute_metrics_per_frequency(pred_dbm, target_dbm)

    # Compact per-frequency summary (RMSE per bin)
    rmse_per_bin = {str(i): freq_metrics[i]["rmse"] for i in range(D_actual)}

    # Single unified metrics.json
    metrics = {
        "overall": overall,
        "per_horizon": {str(h): horizon_metrics[h] for h in sorted(horizon_metrics.keys())},
        "per_node": {str(l): node_metrics[l] for l in range(L)},
        "per_frequency_rmse": rmse_per_bin,
        "timing": {
            "inference_seconds": round(total_eval_time, 4),
            "total_evaluation_seconds": round(total_eval_time, 4),
            "mean_batch_inference_seconds": round(total_eval_time / len(test_loader), 4) if len(test_loader) > 0 else 0,
            "mean_window_inference_seconds": round(total_eval_time / len(test_loader.dataset), 6) if len(test_loader.dataset) > 0 else 0,
        },
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Also save individual files for backward compat
    with open(output_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    with open(output_dir / "per_horizon_metrics.json", "w") as f:
        rows = [{"horizon": h, **horizon_metrics[h]} for h in eval_cfg.get("eval_horizons", [1, 5, 10]) if h in horizon_metrics]
        json.dump(rows, f, indent=2)

    # Save predictions.csv and ground_truth.csv (dBm, flattened)
    np.savetxt(output_dir / "predictions.csv", pred_dbm.reshape(-1, D_actual), delimiter=",", fmt="%.6f")
    np.savetxt(output_dir / "ground_truth.csv", target_dbm.reshape(-1, D_actual), delimiter=",", fmt="%.6f")
    print(f"\nPredictions saved to {output_dir / 'predictions.csv'}")
    print(f"Ground truth saved to {output_dir / 'ground_truth.csv'}")

    # Config copy
    import shutil
    shutil.copy2(args.config, output_dir / "config.yaml")

    # Reshape to (B, T_out, L, F) for plotting
    pred_4d = pred_dbm.reshape(B_actual, T_actual, L, -1)
    target_4d = target_dbm.reshape(B_actual, T_actual, L, -1)
    node_names = config["data"].get("node_names", [f"Node_{i}" for i in range(L)])
    errors = pred_4d - target_4d

    for n, name in enumerate(node_names):
        plot_path = output_dir / f"spectrogram_{name}.png"
        plot_spectrogram_comparison(
            target_4d[0], pred_4d[0],
            n, name, config["windowing"]["input_sequence_length"], plot_path,
        )
        print(f"Spectrogram saved to {plot_path}")

    error_plot_path = output_dir / "error_analysis.png"
    plot_error_analysis(errors[0], node_names, error_plot_path)
    print(f"Error analysis saved to {error_plot_path}")

    print(f"\nInference time: {total_eval_time:.2f}s")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
