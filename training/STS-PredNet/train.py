import os
import sys
import json
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_datasets, collate_branch_samples
from stsprednet import STSPredNet
from utils import (
    set_seed, get_device, compute_metrics, compute_metrics_per_node,
    compute_metrics_per_frequency, save_checkpoint,
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, criterion, device, clip_norm):
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
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * target.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
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

    csv_path = dcfg["dataset_path"]
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", csv_path)

    print("Loading data...")
    train_ds, val_ds, test_ds, stats = create_datasets(csv_path, config)

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        drop_last=True, collate_fn=collate_branch_samples,
    ) if train_ds else None
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        collate_fn=collate_branch_samples,
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds, batch_size=tcfg["batch_size"], shuffle=False,
        collate_fn=collate_branch_samples,
    ) if test_ds else None

    print(f"Train: {len(train_ds)} samples" if train_ds else "No training set")
    print(f"Val:   {len(val_ds)} samples" if val_ds else "No validation set")
    print(f"Test:  {len(test_ds)} samples" if test_ds else "No test set")

    model = STSPredNet(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.MSELoss() if tcfg.get("loss", "mse") == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tcfg["learning_rate"],
        betas=(tcfg.get("beta1", 0.9), tcfg.get("beta2", 0.999)),
        eps=tcfg.get("epsilon", 1e-8),
        weight_decay=tcfg.get("weight_decay", 0.0),
    )

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    patience = tcfg.get("patience", 30)
    clip_norm = tcfg.get("gradient_clip_norm", 5.0)
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "training_log.json")
    log_data = {"train_loss": [], "val_metrics": []}

    for epoch in range(1, tcfg["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, clip_norm,
        ) if train_loader else 0

        val_metrics = {"loss": float("inf")}
        if val_loader:
            val_metrics, _, _ = validate(model, val_loader, criterion, device)

        log_data["train_loss"].append(train_loss)
        log_data["val_metrics"].append(val_metrics)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']} | LR: {lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_metrics.get('loss', 0):.6f} | "
            f"Val RMSE: {val_metrics.get('rmse', 0):.4f}"
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
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    save_checkpoint(
        os.path.join(ckpt_dir, "last_model.pt"),
        model, optimizer, epoch, stats, config, val_metrics,
    )

    if test_loader:
        print("\n=== Test Set Evaluation ===")
        test_metrics, pred, target = validate(model, test_loader, criterion, device)
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"Test MAE:  {test_metrics['mae']:.4f}")
        print(f"Test R²:   {test_metrics['r2']:.4f}")

        node_names = dcfg.get("node_names", None)
        per_node = compute_metrics_per_node(pred, target, node_names)
        for k, v in per_node.items():
            print(f"  {k}: {v:.4f}")

        per_freq = compute_metrics_per_frequency(pred, target)
        freq_rmse = [v for k, v in per_freq.items() if "rmse" in k]
        if freq_rmse:
            print(f"  Per-frequency RMSE: min={min(freq_rmse):.4f} max={max(freq_rmse):.4f}")

        save_checkpoint(
            os.path.join(ckpt_dir, "best_model.pt"),
            model, optimizer, best_epoch, stats, config, test_metrics,
        )

    print(f"\nDone. Best epoch: {best_epoch}. Checkpoints in {ckpt_dir}/")


if __name__ == "__main__":
    main()
