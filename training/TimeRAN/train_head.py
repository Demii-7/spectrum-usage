"""Training script for the TimeRAN forecasting head on top of a MOMENT backbone.

Supports three fine-tuning modes:
  - **linear_probing**: freeze the entire MOMENT encoder, train only the head.
  - **full_finetuning**: unfreeze everything and train end-to-end.
  - **lora**: attach LoRA adapters to the encoder's attention & feed-forward
    projection layers and train them alongside the head.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_datasets
from utils import (
    compute_metrics,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    get_peft_model = None

from momentfm import MOMENTPipeline


# Mapping from user-facing size labels to HuggingFace Hub model identifiers.
VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


def build_model(config: dict, device: torch.device):
    """Construct the MOMENT forecasting model with optional TimeRAN weight loading.

    Loads a MOMENTPipeline and optionally replaces its weights with a pre-trained
    TimeRAN checkpoint (excluding the head layer so it can be re-initialised).
    Applies LoRA adapters if *training_mode* is ``"lora"``.

    Args:
        config: Full training configuration dictionary.
        device: Target torch device.

    Returns:
        A ``torch.nn.Module`` ready for training or evaluation.
    """
    variant = config["model"]["checkpoint_size"].lower()
    if variant not in VARIANT_TO_MODEL:
        raise ValueError(f"Unknown checkpoint_size: {variant}")

    model_name = VARIANT_TO_MODEL[variant]
    horizon = config["windowing"]["prediction_horizon"]
    t_in = config["windowing"]["input_sequence_length"]
    mode = config.get("training_mode", "linear_probing")

    # Freeze encoder & embedder only in linear probing mode.
    freeze_encoder = mode == "linear_probing"
    freeze_embedder = mode == "linear_probing"

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": horizon,
            "seq_len": t_in,
            "freeze_encoder": freeze_encoder,
            "freeze_embedder": freeze_embedder,
            "freeze_head": False,
        },
    )
    model.init()

    # Load pre-trained TimeRAN encoder weights, discarding the head so it starts fresh.
    ckpt_path = Path(__file__).parent / "checkpoints" / variant / f"TimeRAN_{variant}.pth"
    if ckpt_path.exists():
        print(f"Loading TimeRAN checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Strip DataParallel wrapping prefix if present.
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        for k in ["head.linear.weight", "head.linear.bias"]:
            state_dict.pop(k, None)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"TimeRAN checkpoint not found at {ckpt_path}, using raw MOMENT weights")

    # Optionally wrap the encoder with LoRA adapters for parameter-efficient fine-tuning.
    if mode == "lora" and get_peft_model is not None:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
            task_type="FEATURE_EXTRACTION",
        )
        model.encoder = get_peft_model(model.encoder, lora_config)
        model.encoder.print_trainable_parameters()

    model = model.to(device)
    return model


def train_epoch(
    model, dataloader, criterion, optimizer, scheduler, scaler, device
):
    """Run one training epoch over the dataloader.

    Uses automatic mixed precision (AMP) on CUDA for faster training and
    gradient clipping to stabilise training.

    Args:
        model: The forecasting model.
        dataloader: Training data loader.
        criterion: Loss function (e.g. MSELoss).
        optimizer: Weight optimizer.
        scheduler: Learning-rate scheduler (stepped per batch).
        scaler: AMP gradient scaler (``None`` on CPU).
        device: Target torch device.

    Returns:
        Mean training loss for the epoch.
    """
    model.train()
    losses = []
    pbar = tqdm(dataloader, desc="Train")
    for timeseries, forecast in pbar:
        timeseries = timeseries.to(device)
        forecast = forecast.to(device)
        # All time steps are observed (no padding mask needed).
        input_mask = torch.ones(timeseries.shape[0], timeseries.shape[-1], device=device)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(x_enc=timeseries, input_mask=input_mask)
                loss = criterion(out.forecast, forecast)
        else:
            out = model(x_enc=timeseries, input_mask=input_mask)
            loss = criterion(out.forecast, forecast)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if scheduler:
            scheduler.step()

        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})

    return float(np.mean(losses))


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Evaluate the model on a validation/test dataloader without updating weights.

    Returns both aggregate metrics and the concatenated predictions/targets
    for downstream analysis.

    Args:
        model: The forecasting model.
        dataloader: Validation or test data loader.
        criterion: Loss function.
        device: Target torch device.

    Returns:
        Tuple of (metrics_dict, predictions_tensor, targets_tensor).
    """
    model.eval()
    losses = []
    all_pred, all_target = [], []
    for timeseries, forecast in dataloader:
        timeseries = timeseries.to(device)
        forecast = forecast.to(device)
        input_mask = torch.ones(timeseries.shape[0], timeseries.shape[-1], device=device)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(x_enc=timeseries, input_mask=input_mask)
                loss = criterion(out.forecast, forecast)
        else:
            out = model(x_enc=timeseries, input_mask=input_mask)
            loss = criterion(out.forecast, forecast)

        losses.append(loss.item())
        all_pred.append(out.forecast)
        all_target.append(forecast)

    pred_cat = torch.cat(all_pred, dim=0)
    target_cat = torch.cat(all_target, dim=0)
    metrics = compute_metrics(pred_cat.cpu().numpy(), target_cat.cpu().numpy())
    metrics["loss"] = float(np.mean(losses))
    return metrics, pred_cat, target_cat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", default=None, choices=["linear_probing", "full_finetuning", "lora"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    # Load config, CLI overrides take precedence over YAML values.
    config_path = args.config or str(Path(__file__).parent / "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.mode:
        config["training_mode"] = args.mode
    elif "training_mode" not in config:
        config["training_mode"] = "linear_probing"

    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    set_seed(config["training"]["seed"])
    device = get_device(config["device"]["device"])
    print(f"Device: {device}")

    dcfg = config["data"]
    wcfg = config["windowing"]
    scfg = config["split"]

    # Resolve CSV path relative to the repository root if not found as-is.
    csv_path = dcfg["dataset_path"]
    if not Path(csv_path).exists():
        csv_path = str(Path(__file__).resolve().parent.parent.parent / csv_path)

    train_ds, val_ds, test_ds, norm_stats = create_datasets(
        csv_path=csv_path,
        t_in=wcfg["input_sequence_length"],
        t_out=wcfg["prediction_horizon"],
        stride=wcfg.get("stride", 1),
        train_stride=wcfg.get("train_stride"),
        val_stride=wcfg.get("val_stride"),
        test_stride=wcfg.get("test_stride"),
        train_ratio=scfg["train_ratio"],
        val_ratio=scfg["val_ratio"],
        normalization=config["preprocessing"]["normalization"],
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True) if train_ds else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None

    print(f"Train windows: {len(train_ds) if train_ds else 0}, Val: {len(val_ds) if val_ds else 0}, Test: {len(test_ds) if test_ds else 0}")

    model = build_model(config, device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable: {trainable:,}, Frozen: {total - trainable:,}")

    criterion = torch.nn.MSELoss().to(device)
    lr = config["training"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = config["training"]["epochs"]
    max_lr = config["training"].get("max_learning_rate", lr)
    total_steps = len(train_loader) * epochs if train_loader else 0
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3) if total_steps > 0 else None
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ckpt_dir = Path(args.checkpoint_dir or Path(__file__).parent / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.json"
    log_data = {"train_loss": [], "val_metrics": []}

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device) if train_loader else 0.0
        print(f"Epoch {epoch+1}/{epochs}: Train MSE: {train_loss:.6f}")

        val_metrics = {"loss": float("inf"), "rmse": 0.0, "mae": 0.0, "r2": 0.0}
        if val_loader:
            val_metrics, _, _ = validate(model, val_loader, criterion, device)

        log_data["train_loss"].append(train_loss)
        log_data["val_metrics"].append(val_metrics)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"  Val Loss: {val_metrics.get('loss', 0):.6f} | Val RMSE: {val_metrics.get('rmse', 0):.4f}")

        if test_loader:
            test_metrics, _, _ = validate(model, test_loader, criterion, device)
            print(f"  Test MSE: {test_metrics.get('loss', 0):.6f} | Test RMSE: {test_metrics.get('rmse', 0):.4f}")

        # Save checkpoint whenever validation loss improves.
        if val_metrics.get("loss", float("inf")) < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            save_checkpoint(
                str(ckpt_dir / "best_model.pt"),
                model, optimizer, best_epoch, train_loss, val_metrics, config, norm_stats,
            )
            if norm_stats:
                torch.save(norm_stats, ckpt_dir / "normalization_stats.pt")

    # Final test evaluation after all epochs.
    if test_loader:
        print("\n=== Test Set Evaluation ===")
        test_metrics, pred, target = validate(model, test_loader, criterion, device)
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"Test MAE:  {test_metrics['mae']:.4f}")
        print(f"Test R\u00b2:   {test_metrics['r2']:.4f}")

        save_checkpoint(
            str(ckpt_dir / "best_model.pt"),
            model, optimizer, best_epoch, train_loss, test_metrics, config, norm_stats,
        )

    save_checkpoint(
        str(ckpt_dir / "last_model.pt"),
        model, optimizer, epochs, train_loss,
        val_metrics if val_loader else {"loss": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0},
        config, norm_stats,
    )

    print(f"\nDone. Best epoch: {best_epoch}. Checkpoints in {ckpt_dir}/")


if __name__ == "__main__":
    main()
