from __future__ import annotations

import argparse
from pathlib import Path
import sys
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
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.metrics import absolute_and_squared_errors_dbm  # noqa: E402
from training.common.results import append_metric_rows, load_band_definitions, output_dir  # noqa: E402
from training.common.windowing import target_rows_for  # noqa: E402

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
                    out: Path, chunk_id: str):
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

    all_starts = np.arange(0, len(train_input) - window_len + 1)
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            input_mask = torch.ones(x.shape[0], x.shape[-1], device=device)

            optimizer.zero_grad()
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = model(x_enc=x, input_mask=input_mask)
                    loss = criterion(out.forecast, y)
            else:
                out = model(x_enc=x, input_mask=input_mask)
                loss = criterion(out.forecast, y)
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
                        out = model(x_enc=x, input_mask=input_mask)
                else:
                    out = model(x_enc=x, input_mask=input_mask)
                val_loss += criterion(out.forecast, y).item() * x.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"  {chunk_id} epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(out / f"{chunk_id}_training_log.csv", index=False)
    torch.save(
        {"model_state_dict": best_state, "config": config},
        out / "models" / f"{chunk_id}_timeran.pt",
    )
    return model


def predict_timeran(model: nn.Module, device: torch.device,
                    full_x: np.ndarray, target_rows: np.ndarray,
                    horizon: int, t_in: int, batch_size: int) -> np.ndarray:
    origins = target_rows - horizon
    inputs = np.stack(
        [full_x[o - t_in + 1 : o + 1] for o in origins],
        axis=0,
    ).astype(np.float32)
    inputs = inputs.transpose(0, 2, 1)

    loader = DataLoader(torch.from_numpy(inputs).float(), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            input_mask = torch.ones(x.shape[0], x.shape[-1], device=device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = model(x_enc=x, input_mask=input_mask)
            else:
                out = model(x_enc=x, input_mask=input_mask)
            preds.append(out.forecast[:, :, horizon - 1].cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_chunk(config: dict[str, Any], chunk, bands: pd.DataFrame, out: Path):
    tcfg = config["timeran"]
    batch_size = int(tcfg["batch_size"])
    lookback = int(config["windowing"]["lookback"])
    min_history = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    test_splits = config["data"].get("test_splits", ["CC2_test"])

    data = load_chunk(config, chunk)
    train = data.splits["CC2_train"].model_input
    train_raw = data.splits["CC2_train"].raw_dbm

    model = train_one_model(config, train, out, chunk.chunk_id)
    device = next(model.parameters()).device
    t_in = lookback
    max_horizon = max(horizons)

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        min_needed = max(horizon + t_in - 1, min_history)
        for split_name in test_splits:
            split = data.splits[split_name]
            full_x = np.vstack([train, split.model_input]).astype(np.float32)
            full_raw = np.vstack([train_raw, split.raw_dbm]).astype(np.float32)
            history_offset = len(train)

            target_rows = target_rows_for(
                len(split.raw_dbm), history_offset, horizon,
                t_in, min_needed,
            )
            if len(target_rows) == 0:
                continue

            pred = predict_timeran(
                model, device, full_x, target_rows,
                horizon, t_in, batch_size,
            )
            target = full_raw[target_rows]
            _, abs_err, sq_err = absolute_and_squared_errors_dbm(
                pred, target, data.normalization,
            )
            append_metric_rows(
                aggregate_rows, frequency_rows, band_rows,
                chunk_id=chunk.chunk_id,
                start_mhz=chunk.start_mhz,
                end_mhz=chunk.end_mhz,
                split_name=split_name,
                horizon=horizon,
                model=MODEL_NAME,
                target_rows=target_rows,
                history_offset=history_offset,
                freqs=data.shared_frequencies,
                abs_err=abs_err,
                sq_err=sq_err,
                bands=bands,
            )

    return aggregate_rows, frequency_rows, band_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out = args.output_dir or output_dir(config, "TimeRAN")
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    bands = load_band_definitions(config)

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for chunk in chunk_specs(config):
        print(f"Training TimeRAN for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        a, f, b = evaluate_chunk(config, chunk, bands, out)
        aggregate_rows.extend(a)
        frequency_rows.extend(f)
        band_rows.extend(b)

    pd.DataFrame(aggregate_rows).to_csv(out / "aggregate_metrics.csv", index=False)
    pd.DataFrame(frequency_rows).to_csv(out / "per_frequency_metrics.csv", index=False)
    pd.DataFrame(band_rows).to_csv(out / "per_band_metrics.csv", index=False)
    print(f"Wrote {len(aggregate_rows)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
