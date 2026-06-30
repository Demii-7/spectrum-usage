"""
Training script for the ConvLSTM spectrum-prediction model.

Implements the full training loop:
- YAML config loading with optional CLI overrides
- Dataset creation (train/val/test) with normalization
- Per-epoch training with optional teacher forcing, input noise, and gradient clipping
- Validation after every epoch with ReduceLROnPlateau scheduling
- Early stopping based on validation loss
- Checkpointing (best and last model) and test-set evaluation on completion
"""

import os
import sys
import json
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import create_datasets, create_interpolated_map_datasets
from model import ConvLSTMPredictor
from utils import (
    set_seed, get_device, compute_metrics,
    compute_metrics_per_node, save_checkpoint,
)


def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def add_gaussian_noise(x, std):
    """
    Add Gaussian noise to the input tensor as a simple regularization.

    Noise is only applied during training (called within train_epoch) and helps
    the model become robust to small perturbations in input spectra.
    """
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio, noise_std, clip_norm):
    """
    Run one epoch of training.

    For each batch:
    1. Optionally add Gaussian noise to inputs.
    2. Forward pass with teacher forcing (if enabled in config).
    3. Compute loss, backpropagate, optionally clip gradients, and update weights.

    Returns:
        Average loss over the epoch (weighted by batch size).
    """
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = add_gaussian_noise(x, noise_std)  # Regularize with input noise.
        optimizer.zero_grad()
        pred = model(x, y_teacher=y, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(pred, y)
        loss.backward()
        if clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)  # Prevent gradient explosion.
        optimizer.step()
        total_loss += loss.item() * x.size(0)  # Accumulate sum for weighted average.
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluate the model on a validation or test set.

    Returns:
        metrics: dict with "loss", "rmse", "mae", "r2".
        pred_cat: Concatenated predictions, shape (N, t_out, C, H, W).
        target_cat: Corresponding ground truth, same shape.
    """
    model.eval()
    total_loss = 0
    all_pred, all_target = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            all_pred.append(pred)
            all_target.append(y)
    pred_cat = torch.cat(all_pred, dim=0)
    target_cat = torch.cat(all_target, dim=0)
    metrics = compute_metrics(pred_cat, target_cat)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, pred_cat, target_cat


def main():
    """
    Entry point for training the ConvLSTM model.

    Parses CLI args, loads/overrides config, creates datasets and dataloaders,
    builds the model, runs the training loop with validation, early stopping,
    LR scheduling, checkpointing, and final test-set evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--pred-horizon", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    # CLI overrides take precedence over values in the config file.
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.input_len:
        config["windowing"]["input_sequence_length"] = args.input_len
    if args.pred_horizon:
        config["windowing"]["prediction_horizon"] = args.pred_horizon

    set_seed()
    device = get_device(config["device"]["device"])
    print(f"Device: {device}")

    dcfg = config["data"]
    wcfg = config["windowing"]
    scfg = config["split"]
    tcfg = config["training"]
    data_format = dcfg.get("format", "csv")

    print("Loading data...")
    if data_format == "interpolated_map":
        map_path = dcfg["map_path"]
        if not os.path.exists(map_path):
            map_path = os.path.join(os.path.dirname(__file__), "..", "..", map_path)
        train_ds, val_ds, test_ds, stats = create_interpolated_map_datasets(
            map_path=map_path,
            map_key=dcfg.get("map_key", "map_db"),
            t_in=wcfg["input_sequence_length"],
            t_out=wcfg["prediction_horizon"],
            stride=wcfg.get("stride", 1),
            train_stride=wcfg.get("train_stride"),
            val_stride=wcfg.get("val_stride"),
            test_stride=wcfg.get("test_stride"),
            train_ratio=scfg["train_ratio"],
            val_ratio=scfg["val_ratio"],
            chronological=scfg["chronological_split"],
            normalization=config["preprocessing"]["normalization"],
            fit_on_train_only=config["preprocessing"]["fit_on_train_only"],
            imputation_cfg=config["preprocessing"].get("imputation"),
        )
    else:
        csv_path = dcfg["dataset_path"]
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "..", csv_path)
        train_ds, val_ds, test_ds, stats = create_datasets(
            csv_path=csv_path,
            n_nodes=dcfg["n_nodes"],
            n_bins=dcfg["n_bins_per_node"],
            t_in=wcfg["input_sequence_length"],
            t_out=wcfg["prediction_horizon"],
            stride=wcfg.get("stride", 1),
            train_stride=wcfg.get("train_stride"),
            val_stride=wcfg.get("val_stride"),
            test_stride=wcfg.get("test_stride"),
            train_ratio=scfg["train_ratio"],
            val_ratio=scfg["val_ratio"],
            chronological=scfg["chronological_split"],
            normalization=config["preprocessing"]["normalization"],
            fit_on_train_only=config["preprocessing"]["fit_on_train_only"],
        )

    # Shuffle training data for stochasticity; drop_last avoids ragged batches
    # (due to BN layers requiring consistent batch sizes across steps).
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True) if train_ds else None
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=tcfg["batch_size"], shuffle=False) if test_ds else None

    print(f"Train: {len(train_ds)} windows" if train_ds else "No training set")
    print(f"Val:   {len(val_ds)} windows" if val_ds else "No validation set")
    print(f"Test:  {len(test_ds)} windows" if test_ds else "No test set")

    model = ConvLSTMPredictor(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.MSELoss() if tcfg.get("loss", "mse") == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tcfg["learning_rate"],
        betas=(tcfg["beta1"], tcfg["beta2"]),
        eps=tcfg["epsilon"],
        weight_decay=tcfg["weight_decay"],
    )
    # Reduce LR by 50% when validation loss plateaus to help fine-tune convergence.
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=tcfg.get("lr_patience", 10)) \
        if tcfg.get("lr_scheduler") == "reduce_on_plateau" else None

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "training_log.json")
    log_data = {"train_loss": [], "val_metrics": []}

    teacher_forcing = tcfg["teacher_forcing_ratio"]
    noise_std = tcfg.get("noise_std", 0.0)
    clip_norm = tcfg.get("gradient_clip_norm", 0.0)
    patience = tcfg.get("early_stopping_patience", 20)

    for epoch in range(1, tcfg["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            teacher_forcing, noise_std, clip_norm,
        ) if train_loader else 0

        val_metrics = {"loss": float("inf")}
        if val_loader:
            val_metrics, _, _ = validate(model, val_loader, criterion, device)

        # Persist training progress after every epoch for crash recovery.
        log_data["train_loss"].append(train_loss)
        log_data["val_metrics"].append(val_metrics)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{tcfg['epochs']} | LR: {lr:.2e} | Train Loss: {train_loss:.6f} | Val Loss: {val_metrics.get('loss', 0):.6f} | Val RMSE: {val_metrics.get('rmse', 0):.4f}")

        if scheduler:
            scheduler.step(val_metrics.get("loss", float("inf")))

        # Save checkpoint whenever validation loss improves; track early stopping.
        if val_metrics.get("loss", float("inf")) < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                os.path.join(ckpt_dir, "best_model.pt"),
                model, optimizer, epoch, stats, config, val_metrics,
            )
            torch.save(stats, os.path.join(ckpt_dir, "normalization_stats.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    save_checkpoint(
        os.path.join(ckpt_dir, "last_model.pt"),
        model, optimizer, epoch, stats, config, val_metrics,
    )

    print(f"\nDone. Best epoch: {best_epoch}. Checkpoints in {ckpt_dir}/")


if __name__ == "__main__":
    main()
