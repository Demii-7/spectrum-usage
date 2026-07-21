from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from momentfm import MOMENTPipeline  # noqa: E402
from training.common.config import load_config  # noqa: E402
from training.common.data_loader import data_loader_kwargs  # noqa: E402
from training.common.runtime import epoch_log_row, timestamp_utc  # noqa: E402
from training.common.results import prepare_output_dirs  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.windowing import make_window_starts  # noqa: E402

MODEL_NAME = "timeran"

VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


class TimeRANDataset(Dataset):
    def __init__(self, data: np.ndarray, starts: np.ndarray, t_in: int, t_out: int):
        self.data = torch.from_numpy(data).float()
        self.starts = starts
        self.t_in = t_in
        self.t_out = t_out

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        window = self.data[start : start + self.t_in + self.t_out]
        x = window[:self.t_in].T
        y = window[self.t_in:].T
        return x, y


def device_for() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: dict[str, Any], device: torch.device, t_in: int, t_out: int):
    tcfg = config["timeran"]
    variant = tcfg["checkpoint_size"]
    model_name = VARIANT_TO_MODEL.get(variant)
    if model_name is None:
        raise ValueError(f"Unknown checkpoint_size: {variant}")

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": t_out,
            "seq_len": t_in,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
    )
    model.init()
    model = model.to(device)
    return model


def train_one_model(config: dict[str, Any], train_input: np.ndarray,
                    segments, checkpoints: Path, out: Path, chunk_id: str):
    tcfg = config["timeran"]
    lookback = int(config["windowing"]["lookback"])
    max_horizon = max(int(h) for h in config["windowing"]["horizons"])
    batch_size = int(tcfg["batch_size"])
    epochs = int(tcfg["epochs"])
    lr = float(tcfg["learning_rate"])
    max_lr = float(tcfg.get("max_learning_rate", lr))
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    clip_norm = float(tcfg.get("gradient_clip_norm", 5.0))

    t_in = lookback
    t_out = max_horizon
    window_len = t_in + t_out

    all_starts = make_window_starts(
        len(train_input), t_in, t_out, 1, segments
    )
    if len(all_starts) < 10:
        raise ValueError(
            f"Not enough training windows ({len(all_starts)}) "
            f"for t_in={t_in}, t_out={t_out}."
        )

    val_count = max(1, int(len(all_starts) * 0.1))
    train_starts = all_starts[:-val_count]
    val_starts = all_starts[-val_count:]

    train_ds = TimeRANDataset(train_input, train_starts, t_in, t_out)
    val_ds = TimeRANDataset(train_input, val_starts, t_in, t_out)

    loader_kwargs = data_loader_kwargs(config.get("data_loader"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    device = device_for()
    model = build_model(config, device, t_in, t_out)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {chunk_id} params: {total:,} total, {trainable:,} trainable")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    best_loss = float("inf")
    best_state = None
    log_rows = []
    epoch_times: list[float] = []
    training_start_time = timestamp_utc()
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = timestamp_utc()
        t_epoch = time.perf_counter()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            input_mask = torch.ones(x.shape[0], x.shape[-1], device=device)

            optimizer.zero_grad()
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    model_output = model(x_enc=x, input_mask=input_mask)
                    loss = criterion(model_output.forecast, y)
            else:
                model_output = model(x_enc=x, input_mask=input_mask)
                loss = criterion(model_output.forecast, y)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                input_mask = torch.ones(x.shape[0], x.shape[-1], device=device)
                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        model_output = model(x_enc=x, input_mask=input_mask)
                else:
                    model_output = model(x_enc=x, input_mask=input_mask)
                val_loss += criterion(model_output.forecast, y).item() * x.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        t_epoch = time.perf_counter() - t_epoch
        epoch_end_time = timestamp_utc()
        epoch_times.append(t_epoch)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta = avg_time * (epochs - epoch)
        log_rows.append(
            epoch_log_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_start_time=epoch_start_time,
                epoch_end_time=epoch_end_time,
                epoch_duration_sec=t_epoch,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        print(f"  {chunk_id} epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={t_epoch:.1f}s avg={avg_time:.1f}s eta={eta:.0f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    total_time = time.perf_counter() - t_start
    print(f"  {chunk_id} training done in {total_time:.1f}s ({total_time/60:.1f} min)")

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_training_log.csv", index=False)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config,
            "training_start_time": training_start_time,
            "training_end_time": timestamp_utc(),
            "training_duration_sec": total_time,
        },
        checkpoints / f"{chunk_id}_timeran.pt",
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = MODEL_NAME
    if args.output_dir is not None:
        run_dir = args.output_dir
    else:
        exp_name = args.name or f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out, checkpoints = prepare_output_dirs(run_dir)

    for chunk in chunk_specs(config):
        print(f"Training TimeRAN for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        data = load_chunk(config, chunk)
        train = data.splits[data.train_split].model_input
        train_one_model(
            config, train, data.splits[data.train_split].segments,
            checkpoints, out, chunk.chunk_id,
        )


if __name__ == "__main__":
    main()
