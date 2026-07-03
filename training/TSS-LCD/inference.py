from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import Normalizer, load_csv_numpy, build_windows
from model import (
    LatentSpaceEncoder,
    LatentSpaceDecoder,
    TSSConditionConstructor,
    DiffusionModel,
)
from utils import load_config, set_seed, get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Diffusion checkpoint (Stage 3)")
    parser.add_argument("--autoencoder_checkpoint", type=str, required=True)
    parser.add_argument("--tss_checkpoint", type=str, default=None)
    parser.add_argument("--input_csv", type=str, default=None,
                        help="Optional: different CSV file for inference")
    parser.add_argument("--output", type=str, default="predictions.npy",
                        help="Output file for predictions (.npy)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed"))
    device = get_device(config["device"]["device"])

    data_cfg = config["data"]
    model_cfg = config["model"]
    window_cfg = config["windowing"]
    preproc_cfg = config["preprocessing"]
    T_in = window_cfg["input_sequence_length"]
    T_out = window_cfg["prediction_horizon"]
    L = data_cfg["n_nodes"]
    F = data_cfg["n_bins_per_node"]

    csv_path = args.input_csv or data_cfg["dataset_path"]
    data = load_csv_numpy(
        csv_path, data_cfg["n_nodes"], data_cfg["n_bins_per_node"],
        cc2_only=data_cfg.get("cc2_only_smoke_test", False),
        selected_nodes=data_cfg.get("selected_nodes"),
        node_names=data_cfg.get("node_names"),
    )
    if data_cfg.get("cc2_only_smoke_test", False):
        L = 1
        F = data_cfg["n_bins_per_node"]

    normalizer = Normalizer(method=preproc_cfg.get("normalization", "minmax"))
    normalizer.fit(data)
    data_norm = normalizer.transform(data)

    test_stride = window_cfg.get("test_stride", 1)
    X, _ = build_windows(data_norm, T_in, T_out, test_stride)

    if len(X) == 0:
        print("No complete windows available for the given T_in/T_out.")
        return

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

    X_tensor = torch.from_numpy(X).float().to(device)
    all_preds = []

    infer_start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(X_tensor), 32):
            batch = X_tensor[i:i + 32]
            cond_z = tss_cc(batch)
            z_sample = diffusion.p_sample_loop(cond_z)
            y_hat = dec(z_sample)
            all_preds.append(y_hat.cpu().numpy())

    total_infer_time = time.perf_counter() - infer_start

    pred_norm = np.concatenate(all_preds, axis=0)
    B_actual, T_actual, D_actual = pred_norm.shape
    pred_flat = pred_norm.reshape(B_actual * T_actual, D_actual)
    pred_dbm = normalizer.inverse_transform(pred_flat).reshape(pred_norm.shape)

    np.save(args.output, pred_dbm)

    # Save metadata
    metadata = {
        "output_file": args.output,
        "shape": list(pred_dbm.shape),
        "num_windows": B_actual,
        "T_out": T_actual,
        "D": D_actual,
        "inference_time_seconds": round(total_infer_time, 4),
        "mean_window_inference_seconds": round(total_infer_time / B_actual, 6) if B_actual > 0 else 0,
    }
    meta_path = Path(args.output).with_suffix(".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Predictions saved to {args.output}  shape={pred_dbm.shape}")
    print(f"Metadata saved to {meta_path}")
    print(f"Total inference time: {total_infer_time:.2f}s")


if __name__ == "__main__":
    main()
