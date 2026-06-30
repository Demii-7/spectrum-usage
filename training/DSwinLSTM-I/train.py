import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import set_seed, get_device, load_config, save_checkpoint, load_checkpoint, init_logger
from dataset import AERPAWDataset, load_and_split
from model import DSwinLSTM_I


def train_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0
    for X, mask, Y in loader:
        X = X.permute(0, 1, 4, 2, 3).contiguous().to(device)
        mask = mask.to(device)
        Y = Y.permute(0, 1, 4, 2, 3).contiguous().to(device)

        optimizer.zero_grad()
        pred = model(X, mask)
        loss = criterion(pred, Y)
        loss.backward()

        if config["training"].get("gradient_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])

        optimizer.step()
        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
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

    train_dataset = AERPAWDataset(train_norm, config, split="train")
    val_dataset = AERPAWDataset(val_norm, config, split="val")
    test_dataset = AERPAWDataset(test_norm, config, split="test")

    batch_size = config["training"].get("batch_size", 4)
    num_workers = config["training"].get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(config["training"].get("seed", 42)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = DSwinLSTM_I(config).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    epochs = config["training"].get("epochs", 400)
    early_stopping = config["training"].get("early_stopping", False)
    patience = config["training"].get("patience", 30)

    ckpt_dir = os.path.join(script_dir, config["paths"]["checkpoints_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)

    stats_path = os.path.join(ckpt_dir, "normalization_stats.pt")
    torch.save(stats, stats_path)

    logger = init_logger(ckpt_dir, name="train")

    best_val_loss = float("inf")
    no_improve = 0
    train_losses = []

    logger.info(f"Training DSwinLSTM-I on {device} | epochs={epochs} batch={batch_size} lr={config['training']['learning_rate']}")
    logger.info(f"Data: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    logger.info(f"Map shape: {config['model']['map_height']}x{config['model']['map_width']} T_in={config['windowing']['input_sequence_length']} T_out={config['windowing']['prediction_horizon']}")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)

        log = f"Epoch {epoch:03d}/{epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}"

        save_checkpoint(
            os.path.join(ckpt_dir, "last_model.pt"),
            model, optimizer, epoch, stats, config, {"val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(
                os.path.join(ckpt_dir, "best_model.pt"),
                model, optimizer, epoch, stats, config, {"val_loss": val_loss})
            log += "  (saved best)"
        else:
            no_improve += 1

        logger.info(log)

        if early_stopping and no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

    print("\nTraining complete. Files saved:")
    print(f"  {ckpt_dir}/best_model.pt")
    print(f"  {ckpt_dir}/last_model.pt")
    print(f"  {ckpt_dir}/normalization_stats.pt")


if __name__ == "__main__":
    main()
