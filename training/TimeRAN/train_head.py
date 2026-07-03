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
import time
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
    load_info = {"checkpoint_path": str(ckpt_path), "missing_keys": [], "unexpected_keys": [], "used_raw_moment": True}
    if config["model"].get("use_timeran_checkpoint", True) and ckpt_path.exists():
        print(f"Loading TimeRAN checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Strip DataParallel wrapping prefix if present.
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        for k in ["head.linear.weight", "head.linear.bias"]:
            state_dict.pop(k, None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        load_info["missing_keys"] = list(missing)
        load_info["unexpected_keys"] = list(unexpected)
        load_info["used_raw_moment"] = False
        if missing:
            print(f"Missing keys from checkpoint load: {missing}")
        if unexpected:
            print(f"Unexpected keys from checkpoint load: {unexpected}")
    else:
        if config["model"].get("use_timeran_checkpoint", True):
            print(f"TimeRAN checkpoint not found at {ckpt_path}, using raw MOMENT weights")
        else:
            print("Configured to use raw MOMENT weights (TimeRAN checkpoint loading disabled)")

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
    return model, load_info


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
    batch_times = []
    pbar = tqdm(dataloader, desc="Train")
    for timeseries, forecast in pbar:
        batch_start = time.perf_counter()
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
        batch_times.append(time.perf_counter() - batch_start)
        pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})

    mean_batch_time = float(np.mean(batch_times)) if batch_times else 0.0
    batches_per_second = float(1.0 / mean_batch_time) if mean_batch_time > 0 else 0.0
    return float(np.mean(losses)), batches_per_second


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
    inference_time = 0.0
    for timeseries, forecast in dataloader:
        batch_start = time.perf_counter()
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
        inference_time += time.perf_counter() - batch_start

    pred_cat = torch.cat(all_pred, dim=0)
    target_cat = torch.cat(all_target, dim=0)
    metrics = compute_metrics(pred_cat.cpu().numpy(), target_cat.cpu().numpy())
    metrics["loss"] = float(np.mean(losses))
    metrics["inference_time_seconds"] = inference_time
    metrics["mean_batch_inference_time_seconds"] = inference_time / len(dataloader) if len(dataloader) > 0 else 0.0
    return metrics, pred_cat, target_cat


def build_optimizer(config: dict, model: torch.nn.Module):
    name = str(config["training"].get("optimizer", "adam")).lower()
    lr = config["training"]["learning_rate"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if name == "nadam":
        return torch.optim.NAdam(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {config['training'].get('optimizer')}")


def build_scheduler(config: dict, optimizer, train_loader_len: int):
    sched_name = str(config["training"].get("scheduler", "onecycle")).lower()
    epochs = config["training"]["epochs"]
    if sched_name == "onecycle":
        total_steps = train_loader_len * epochs if train_loader_len > 0 else 0
        return OneCycleLR(optimizer, max_lr=config["training"].get("max_learning_rate", config["training"]["learning_rate"]), total_steps=total_steps, pct_start=0.3) if total_steps > 0 else None
    if sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config["training"].get("scheduler_factor", 0.5)),
            patience=int(config["training"].get("scheduler_patience", 5)),
        )
    if sched_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {config['training'].get('scheduler')}")


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

    print(f"Train windows: {len(train_ds) if train_ds else 0}, Val: {len(val_ds) if val_ds else 0}, Test: {len(test_ds) if test_ds else 0}")

    model, load_info = build_model(config, device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable: {trainable:,}, Frozen: {total - trainable:,}")

    criterion = torch.nn.MSELoss().to(device)
    optimizer = build_optimizer(config, model)

    epochs = config["training"]["epochs"]
    scheduler = build_scheduler(config, optimizer, len(train_loader) if train_loader else 0)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ckpt_dir = Path(args.checkpoint_dir or Path(__file__).parent / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.json"
    log_data = {"epochs": [], "summary": {}, "checkpoint_load": load_info}
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    best_val_loss = float("inf")
    best_epoch = 0
    train_start = time.perf_counter()

    peak_gpu_memory_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_loss, batches_per_second = train_epoch(model, train_loader, criterion, optimizer, scheduler if isinstance(scheduler, OneCycleLR) else None, scaler, device) if train_loader else (0.0, 0.0)

        val_metrics = {"loss": float("inf"), "rmse": 0.0, "mae": 0.0, "r2": 0.0}
        if val_loader:
            val_metrics, _, _ = validate(model, val_loader, criterion, device)

        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            scheduler.step(val_metrics.get("loss", float("inf")))

        epoch_seconds = time.perf_counter() - epoch_start
        if device.type == "cuda":
            peak_gpu_memory_mb = max(peak_gpu_memory_mb, torch.cuda.max_memory_allocated(device) / (1024 * 1024))

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss", float("inf")),
            "val_rmse": val_metrics.get("rmse", 0.0),
            "epoch_seconds": epoch_seconds,
            "batches_per_second": batches_per_second,
            "device": str(device),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if device.type == "cuda":
            epoch_log["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        log_data["epochs"].append(epoch_log)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        print(
            f"Epoch {epoch+1}/{epochs}: Train MSE: {train_loss:.6f} | "
            f"Val Loss: {val_metrics.get('loss', 0):.6f} | Val RMSE: {val_metrics.get('rmse', 0):.4f} | "
            f"Epoch Time: {epoch_seconds:.2f}s | Batches/s: {batches_per_second:.2f}"
        )

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

    save_checkpoint(
        str(ckpt_dir / "last_model.pt"),
        model, optimizer, epochs, train_loss,
        val_metrics if val_loader else {"loss": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0},
        config, norm_stats,
    )

    total_training_time = time.perf_counter() - train_start
    mean_epoch_time = float(np.mean([entry["epoch_seconds"] for entry in log_data["epochs"]])) if log_data["epochs"] else 0.0
    log_data["summary"] = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_training_time_seconds": total_training_time,
        "average_epoch_time_seconds": mean_epoch_time,
        "device": str(device),
        "peak_gpu_memory_mb": peak_gpu_memory_mb if device.type == "cuda" else None,
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "dataset_path": csv_path,
        "normalization": config["preprocessing"]["normalization"],
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(
        f"\nDone. Best epoch: {best_epoch}. Total training time: {total_training_time:.2f}s. "
        f"Average epoch time: {mean_epoch_time:.2f}s. Checkpoints in {ckpt_dir}/"
    )


if __name__ == "__main__":
    main()
