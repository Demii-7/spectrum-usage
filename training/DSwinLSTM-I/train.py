import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import set_seed, get_device, load_config, save_checkpoint, init_logger
from dataset import SpectrumMapDataset, load_and_split
from model import DSwinLSTM_I


def make_loss(name):
    loss_name = str(name).lower()
    if loss_name == "mse":
        return lambda pred, target: torch.mean((pred - target) ** 2)
    if loss_name == "rmse":
        return lambda pred, target: torch.sqrt(torch.mean((pred - target) ** 2) + 1e-12)
    if loss_name == "mae":
        return lambda pred, target: torch.mean(torch.abs(pred - target))
    raise ValueError(f"Unsupported loss: {name}")


def build_optimizer(name, model, lr):
    opt_name = str(name).lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if opt_name == "nadam":
        return torch.optim.NAdam(model.parameters(), lr=lr)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(config, optimizer):
    sched_name = str(config["training"].get("lr_scheduler", "none")).lower()
    if sched_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config["training"].get("lr_factor", 0.5)),
            patience=int(config["training"].get("lr_patience", 5)),
        )
    if sched_name == "none":
        return None
    raise ValueError(f"Unsupported lr_scheduler: {config['training'].get('lr_scheduler')}")


def train_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0.0
    teacher_forcing_ratio = float(config["training"].get("teacher_forcing_ratio", 1.0))
    for X, mask, Y in loader:
        X = X.permute(0, 1, 4, 2, 3).contiguous().to(device)
        mask = mask.to(device)
        Y = Y.permute(0, 1, 4, 2, 3).contiguous().to(device)

        optimizer.zero_grad()
        pred = model(X, mask, y_teacher=Y, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(pred, Y)
        loss.backward()
        grad_clip = config["training"].get("gradient_clip")
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    if loader is None or len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    total_loss = 0.0
    for X, mask, Y in loader:
        X = X.permute(0, 1, 4, 2, 3).contiguous().to(device)
        mask = mask.to(device)
        Y = Y.permute(0, 1, 4, 2, 3).contiguous().to(device)
        pred = model(X, mask)
        loss = criterion(pred, Y)
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--cc2-only", action="store_true", help="Use CC2-only smoke mode")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)
    cc2_only = args.cc2_only or config["data"].get("cc2_only_smoke_test", False)

    device = get_device(config.get("device", {}).get("device", "auto"))
    set_seed(config["training"].get("seed", 42))

    train_norm, val_norm, test_norm, stats, node_names = load_and_split(config, cc2_only=cc2_only)
    config["model"]["map_height"] = int(stats["grid_height"])
    config["model"]["map_width"] = int(stats["grid_width"])
    config["model"]["input_channels"] = int(stats["n_freq_bins"])

    train_dataset = SpectrumMapDataset(train_norm, config, split="train")
    val_dataset = SpectrumMapDataset(val_norm, config, split="val")
    test_dataset = SpectrumMapDataset(test_norm, config, split="test")

    batch_size = config["training"].get("batch_size", 4)
    num_workers = config["training"].get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(config["training"].get("seed", 42)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if len(val_dataset) > 0 else None

    model = DSwinLSTM_I(config).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    criterion = make_loss(config["training"].get("loss", "mse"))
    optimizer = build_optimizer(config["training"].get("optimizer", "adam"), model, config["training"]["learning_rate"])
    scheduler = build_scheduler(config, optimizer)

    epochs = config["training"].get("epochs", 400)
    early_stopping = config["training"].get("early_stopping", False)
    patience = config["training"].get("patience", 30)

    ckpt_dir = os.path.join(script_dir, config["paths"]["checkpoints_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(stats, os.path.join(ckpt_dir, "normalization_stats.pt"))

    logger = init_logger(ckpt_dir, name="train")
    logger.info(f"Training DSwinLSTM-I on {device} | epochs={epochs} batch={batch_size} lr={config['training']['learning_rate']}")
    logger.info(f"Data: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    logger.info(f"Map shape: {config['model']['map_height']}x{config['model']['map_width']} C={config['model']['input_channels']} T_in={config['windowing']['input_sequence_length']} T_out={config['windowing']['prediction_horizon']}")

    best_val_loss = float("inf")
    no_improve = 0
    epoch_times = []
    training_log = {"epochs": [], "summary": {}}
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        val_loss = validate(model, val_loader, criterion, device) if val_loader is not None else train_loss
        if scheduler is not None:
            scheduler.step(val_loss)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        save_checkpoint(os.path.join(ckpt_dir, "last_model.pt"), model, optimizer, epoch, stats, config, {"val_loss": val_loss, "train_loss": train_loss})

        log = f"Epoch {epoch:03d}/{epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}  Epoch Time: {epoch_time:.2f}s"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(os.path.join(ckpt_dir, "best_model.pt"), model, optimizer, epoch, stats, config, {"val_loss": val_loss, "train_loss": train_loss})
            log += "  (saved best)"
        else:
            no_improve += 1

        training_log["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_seconds": epoch_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        logger.info(log)

        if early_stopping and no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    total_training_time = time.perf_counter() - train_start
    mean_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    training_log["summary"] = {
        "epochs_completed": len(training_log["epochs"]),
        "best_val_loss": best_val_loss,
        "elapsed_training_time_seconds": total_training_time,
        "mean_epoch_time_seconds": mean_epoch_time,
    }
    with open(os.path.join(ckpt_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
    logger.info(f"Total training time: {total_training_time:.2f}s | Mean epoch time: {mean_epoch_time:.2f}s")

    print("\nTraining complete. Files saved:")
    print(f"  {ckpt_dir}/best_model.pt")
    print(f"  {ckpt_dir}/last_model.pt")
    print(f"  {ckpt_dir}/normalization_stats.pt")
    print(f"  {ckpt_dir}/training_log.json")


if __name__ == "__main__":
    main()
