from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from training.common.results import output_dir


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prepare_output_dirs(config: dict[str, Any], model_name: str) -> tuple[Path, Path]:
    out = output_dir(config, model_name)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    return out, checkpoints


def finalize_results(
    out: Path,
    model_name: str,
    aggregate_rows: list[dict[str, Any]],
    frequency_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    extra_lines: Iterable[str] = (),
) -> None:
    aggregate = pd.DataFrame(aggregate_rows)
    frequency = pd.DataFrame(frequency_rows)
    band = pd.DataFrame(band_rows)
    aggregate.to_csv(out / "aggregate_metrics.csv", index=False)
    frequency.to_csv(out / "per_frequency_metrics.csv", index=False)
    band.to_csv(out / "per_band_metrics.csv", index=False)

    lines = [f"Model: {model_name}"]
    if aggregate.empty:
        lines.append("No aggregate metrics produced.")
    else:
        lines.append(f"Aggregate rows: {len(aggregate)}")
        lines.append(f"Per-frequency rows: {len(frequency)}")
        lines.append(f"Per-band rows: {len(band)}")
        best = aggregate.sort_values("mae_db").iloc[0]
        lines.append(
            "Best aggregate MAE: "
            f"{best['mae_db']:.4f} dB on {best['chunk_id']} {best['split']} h={int(best['horizon'])}"
        )
    lines.extend(extra_lines)
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def epoch_log_row(
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    epoch_start_time: str,
    epoch_end_time: str,
    epoch_duration_sec: float,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch_start_time": epoch_start_time,
        "epoch_end_time": epoch_end_time,
        "epoch_duration_sec": epoch_duration_sec,
    }
    if learning_rate is not None:
        row["learning_rate"] = learning_rate
    return row
