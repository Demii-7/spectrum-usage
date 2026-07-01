"""Training script for STS-PredNet spectrum prediction model.

Loads configuration, creates datasets and dataloaders, trains the
STSPredNet model with early stopping, and saves checkpoints selected by
validation loss only. Final test evaluation is handled by ``evaluate.py``.
"""
import os
import sys
import json
import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_datasets, create_interpolated_map_datasets, collate_branch_samples
from stsprednet import STSPredNet
from utils import set_seed, get_device, compute_metrics, save_checkpoint


def load_config(config_path):
    """Load YAML configuration file from disk."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, criterion, device, clip_norm):
    """Run one training epoch over all batches.

    Args:
        model: The STSPredNet model.
        loader: DataLoader supplying training batches.
        optimizer: Optimizer for gradient updates.
        criterion: Loss function (e.g. MSELoss).
        device: torch device for computation.
        clip_norm: Max gradient norm for clipping (0 disables).

    Returns:
        Average epoch loss normalized by dataset size.
    """
    model.train()
    total_loss = 0
    for batch in loader:
        closeness = batch.get("closeness", None)
        period = batch.get("period", None)
        trend = batch.get("trend", None)
        target = batch["target"].to(device)

        if closeness is not None:
            closeness = closeness.to(device)
        if period is not None:
            period = period.to(device)
        if trend is not None:
            trend = trend.to(device)

        optimizer.zero_grad()
        pred = model(closeness, period, trend)
        loss = criterion(pred, target)
        loss.backward()
        if clip_norm > 0:
            # Prevent exploding gradients by scaling down if norm exceeds threshold
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * target.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """Evaluate the model on a validation or test loader.

    Args:
        model: Trained STSPredNet model.
        loader: DataLoader for evaluation.
        criterion: Loss function.
        device: torch device.

    Returns:
        Tuple of (metrics dict, concatenated predictions, concatenated targets).
    """
    model.eval()
    total_loss = 0
    all_pred, all_target = [], []
    with torch.no_grad():
        for batch in loader:
            closeness = batch.get("closeness", None)
            period = batch.get("period", None)
            trend = batch.get("trend", None)
            target = batch["target"].to(device)

            if closeness is not None:
                closeness = closeness.to(device)
            if period is not None:
                period = period.to(device)
            if trend is not None:
                trend = trend.to(device)

            pred = model(closeness, period, trend)
            loss = criterion(pred, target)
            total_loss += loss.item() * target.size(0)
            all_pred.append(pred)
            all_target.append(target)

    pred_cat = torch.cat(all_pred, dim=0)
    target_cat = torch.cat(all_target, dim=0)
    metrics = compute_metrics(pred_cat, target_cat)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, pred_cat, target_cat


def main():
    """Entry point: parse args, load config, train, validate, and test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    # Override config values with CLI arguments if provided
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    seed = config["training"].get("seed", 42)
    set_seed(seed)
    device = get_device(config["device"]["device"])
    print(f"Device: {device}")

    dcfg = config["data"]
    tcfg = config["training"]
    data_format = dcfg.get("format", "csv")

    if data_format == "interpolated_map":
        map_path = dcfg["map_path"]
        if not os.path.exists(map_path):
            map_path = os.path.join(os.path.dirname(__file__), "..", "..", map_path)
        print("Loading interpolated map data...")
        train_ds, val_ds, test_ds, stats = create_interpolated_map_datasets(map_path, config)
        F, H, W = stats.get("n_freq", 200), stats.get("grid_h", 50), stats.get("grid_w", 50)
        config["model"]["input_channels"] = F
        config["model"]["map_height"] = H
        config["model"]["map_width"] = W
        config["model"]["kernel_size"] = dcfg.get("kernel_size_map", [3, 3])
        bcfg = config["branches"]
        temporal = dcfg.get("temporal_overrides", {})
        if temporal:
            bcfg["lc"] = int(temporal.get("lc", bcfg["lc"]))
            bcfg["lp"] = int(temporal.get("lp", bcfg["lp"]))
            bcfg["period_interval"] = int(temporal.get("period_interval", bcfg["period_interval"]))
            print(f"  Temporal overrides: lc={bcfg['lc']}, lp={bcfg['lp']}, period_interval={bcfg['period_interval']}")
    else:
        csv_path = dcfg["dataset_path"]
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "..", csv_path)
        print("Loading CSV data...")
        train_ds, val_ds, test_ds, stats = create_datasets(csv_path, config)

    # Build dataloaders; skip if a split has no samples
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        drop_last=True, collate_fn=collate_branch_samples,
    ) if train_ds else None
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        collate_fn=collate_branch_samples,
    ) if val_ds else None
    print(f"Train: {len(train_ds)} samples" if train_ds else "No training set")
    print(f"Val:   {len(val_ds)} samples" if val_ds else "No validation set")
    print(f"Test:  {len(test_ds)} samples" if test_ds else "No test set")

    model = STSPredNet(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.MSELoss() if tcfg.get("loss", "mse") == "mse" else nn.L1Loss()
    optimizer_name = str(tcfg.get("optimizer", "adam")).lower()
    optimizer_kwargs = {
        "lr": tcfg["learning_rate"],
        "betas": (tcfg.get("beta1", 0.9), tcfg.get("beta2", 0.999)),
        "eps": tcfg.get("epsilon", 1e-8),
        "weight_decay": tcfg.get("weight_decay", 0.0),
    }
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    elif optimizer_name == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {tcfg.get('optimizer')}")

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    patience = tcfg.get("patience", 30)
    clip_norm = tcfg.get("gradient_clip_norm", 5.0)
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "training_log.json")
    log_data = {
        "train_loss": [],
        "val_metrics": [],
        "epoch_time_seconds": [],
        "summary": {},
    }

    training_start_time = time.perf_counter()
    for epoch in range(1, tcfg["epochs"] + 1):
        epoch_start_time = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, clip_norm,
        ) if train_loader else 0

        val_metrics = {"loss": float("inf")}
        if val_loader:
            val_metrics, _, _ = validate(model, val_loader, criterion, device)

        epoch_time_seconds = time.perf_counter() - epoch_start_time
        log_data["train_loss"].append(train_loss)
        log_data["val_metrics"].append(val_metrics)
        log_data["epoch_time_seconds"].append(epoch_time_seconds)
        log_data["summary"] = {
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "elapsed_training_time_seconds": time.perf_counter() - training_start_time,
            "mean_epoch_time_seconds": sum(log_data["epoch_time_seconds"]) / len(log_data["epoch_time_seconds"]),
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']} | LR: {lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_metrics.get('loss', 0):.6f} | "
            f"Val RMSE: {val_metrics.get('rmse', 0):.4f} | Epoch Time: {epoch_time_seconds:.2f}s"
        )

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
            # Stop training if validation loss has not improved for `patience` epochs
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    save_checkpoint(
        os.path.join(ckpt_dir, "last_model.pt"),
        model, optimizer, epoch, stats, config, val_metrics,
    )

    total_training_time = time.perf_counter() - training_start_time
    mean_epoch_time = sum(log_data["epoch_time_seconds"]) / len(log_data["epoch_time_seconds"]) if log_data["epoch_time_seconds"] else 0.0
    log_data["summary"] = {
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_training_time_seconds": total_training_time,
        "mean_epoch_time_seconds": mean_epoch_time,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(
        f"\nDone. Best epoch: {best_epoch}. Total training time: {total_training_time:.2f}s. "
        f"Mean epoch time: {mean_epoch_time:.2f}s. Checkpoints in {ckpt_dir}/"
    )


if __name__ == "__main__":
    main()
