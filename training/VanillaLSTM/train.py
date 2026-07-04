from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import create_datasets
from model import VanillaLSTMForecaster
from utils import count_parameters, get_device, load_config, save_checkpoint, save_normalization_stats, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VanillaLSTM spectrum forecaster.")
    parser.add_argument("--config", default="training/VanillaLSTM/config.yaml", help="Path to config YAML.")
    parser.add_argument("--csv", default=None, help="Optional CSV path override.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--lr", type=float, default=None, help="Optional learning-rate override.")
    return parser.parse_args()


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer_name = str(config["training"].get("optimizer", "adam")).lower()
    learning_rate = float(config["training"]["learning_rate"])
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; only 'adam' is implemented.")
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            batch_size = inputs.size(0)
            total_loss += float(criterion(predictions, targets).item()) * batch_size
            total_examples += batch_size
    return total_loss / max(total_examples, 1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        config["training"]["learning_rate"] = float(args.lr)

    set_seed(int(config["training"]["seed"]))
    device = get_device(str(config["device"]["device"]))
    bundle = create_datasets(config, csv_path=args.csv)

    train_dataset = bundle["datasets"]["train"]
    val_dataset = bundle["datasets"]["val"]
    test_dataset = bundle["datasets"]["test"]
    if len(train_dataset) == 0:
        raise RuntimeError("Training split produced zero windows. Check CSV size and window settings.")

    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    _ = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VanillaLSTMForecaster(config).to(device)
    parameter_count = count_parameters(model)
    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, config)

    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_dir / "best_model.pt"
    last_model_path = checkpoints_dir / "last_model.pt"
    normalization_path = checkpoints_dir / "normalization_stats.pt"
    training_log_path = checkpoints_dir / "training_log.json"

    stats = bundle["normalization_stats"]
    save_normalization_stats(normalization_path, stats.mean_dbm, stats.std_dbm)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_limit = int(config["training"].get("patience", 15))
    patience_counter = 0
    use_early_stopping = bool(config["training"].get("early_stopping", True))
    gradient_clip = float(config["training"].get("gradient_clip", 0.0))
    epochs = int(config["training"]["epochs"])
    training_log: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_examples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            batch_size_now = inputs.size(0)
            total_train_loss += float(loss.item()) * batch_size_now
            total_examples += batch_size_now

        train_loss = total_train_loss / max(total_examples, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device) if len(val_dataset) > 0 else float("nan")

        improved = len(val_dataset) == 0 or val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(best_model_path, model, optimizer, epoch, config, best_val_loss)
        else:
            patience_counter += 1

        save_checkpoint(last_model_path, model, optimizer, epoch, config, best_val_loss)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        training_log.append(log_entry)
        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val={best_val_loss:.6f}"
        )

        if use_early_stopping and len(val_dataset) > 0 and patience_counter >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    payload = {
        "config": config,
        "window_counts": bundle["window_counts"],
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "log": training_log,
    }
    with training_log_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    final_train_loss = training_log[-1]["train_loss"] if training_log else float("nan")
    final_val_loss = training_log[-1]["val_loss"] if training_log else float("nan")
    print(f"Train windows: {bundle['window_counts']['train']}")
    print(f"Val windows: {bundle['window_counts']['val']}")
    print(f"Test windows: {bundle['window_counts']['test']}")
    print(f"Parameters: {parameter_count}")
    print(f"Final train loss: {final_train_loss:.6f}")
    print(f"Final val loss: {final_val_loss:.6f}")
    print(f"Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
