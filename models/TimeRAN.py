"""
TimeRAN forecasting model (and its MOMENT controls).

TimeRAN (Panitsas and Tassiulas, 2026) is MOMENT with continued masked-reconstruction pretraining on RAN telemetry. Forecasting
always trains a new linear head on the flattened patch embeddings; the paper's two regimes are linear probing (head only) and full
fine-tuning. There is no zero-shot forecasting.

Integrated pipeline format:
    input:  (batch, lookback, features)
    output: (batch, prediction_horizon, features)

Every feature is an independent channel (MOMENT reshapes channels into the batch), so the model is channel-independent.

Model settings (``config["model"]``):
    backbone_init       "timeran" (MOMENT weights, then the TimeRAN checkpoint), "moment" (MOMENT weights only) or "random"
                        (same architecture, freshly initialized). Older configs: ``use_timeran_checkpoint`` true/false.
    backbone_trainable  "head" (encoder and patch embedding frozen: linear probing) or "all" (full fine-tuning). Older configs:
                        ``freeze_encoder`` / ``freeze_embedder``.
    context_length      the window the backbone sees (default 512, as in pretraining). Shorter inputs are left-padded with their
                        oldest value and the padding is masked out of the normalization and of attention, which is how MOMENT handles
                        short series. 0 builds the backbone for the raw lookback instead (the old behaviour: the patching then drops
                        the newest ``lookback % 8`` steps, so lookbacks must be a multiple of 8).
    timeran_checkpoint_path  local path of ``TimeRAN_<size>.pth`` (required for "timeran"; loading fails if keys do not match).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}
PATCH_LENGTH = 8


def _backbone_init(model_cfg: dict[str, Any]) -> str:
    if "backbone_init" in model_cfg:
        value = str(model_cfg["backbone_init"]).lower()
    else:
        value = "timeran" if bool(model_cfg.get("use_timeran_checkpoint", True)) else "moment"
    if value not in {"timeran", "moment", "random"}:
        raise ValueError(f"backbone_init must be timeran, moment or random, got {value!r}")
    return value


def _backbone_trainable(model_cfg: dict[str, Any]) -> str:
    if "backbone_trainable" in model_cfg:
        value = str(model_cfg["backbone_trainable"]).lower()
    else:
        frozen = bool(model_cfg.get("freeze_encoder", True)) or bool(model_cfg.get("freeze_embedder", True))
        value = "head" if frozen else "all"
    if value not in {"head", "all"}:
        raise ValueError(f"backbone_trainable must be head or all, got {value!r}")
    return value


def load_timeran_weights(moment: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a TimeRAN state dict into a MOMENT model; every backbone parameter must be covered. Returns a small report."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"TimeRAN checkpoint not found at {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"TimeRAN checkpoint must contain a state dict, got {type(state).__name__} from {path}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if not key.startswith("head.")}     # the head depends on this run's horizon
    backbone = {name for name, _ in moment.named_parameters() if not name.startswith("head.")}
    missing = sorted(backbone - set(state))
    if missing:
        raise ValueError(f"TimeRAN checkpoint {path.name} does not cover {len(missing)} of {len(backbone)} backbone parameters "
                         f"(first: {missing[:3]}); refusing to run with partly random weights")
    result = moment.load_state_dict(state, strict=False)
    return {"timeran_checkpoint": str(path), "loaded_tensors": len(state), "unexpected": list(result.unexpected_keys)[:10]}


class TimeRANForecaster(nn.Module):
    """MOMENT-based forecaster: TimeRAN, MOMENT or randomly initialized backbone; head-only or full training."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        from momentfm import MOMENTPipeline

        model_cfg = config["model"]
        self.input_sequence_length = int(model_cfg["input_sequence_length"])
        self.prediction_horizon = int(model_cfg["prediction_horizon"])
        self.input_size = int(model_cfg["input_size"])
        self.context_length = int(model_cfg.get("context_length", 512))
        if self.context_length and self.context_length < self.input_sequence_length:
            raise ValueError(f"context_length {self.context_length} is shorter than the lookback {self.input_sequence_length}")
        if not self.context_length and self.input_sequence_length % PATCH_LENGTH:
            raise ValueError(f"without padding the lookback must be a multiple of {PATCH_LENGTH} (MOMENT drops the remainder, "
                             f"which are the newest steps); got {self.input_sequence_length}")
        size = str(model_cfg.get("checkpoint_size", "small")).lower()
        if size not in VARIANT_TO_MODEL:
            raise ValueError(f"Unknown TimeRAN checkpoint_size {size!r}; expected one of {sorted(VARIANT_TO_MODEL)}")
        self.backbone_init = _backbone_init(model_cfg)
        self.backbone_trainable = _backbone_trainable(model_cfg)
        self.moment = MOMENTPipeline.from_pretrained(
            VARIANT_TO_MODEL[size],
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": self.prediction_horizon,
                "seq_len": self.context_length or self.input_sequence_length,
                "head_dropout": float(model_cfg.get("head_dropout", 0.1)),
                "freeze_encoder": False, "freeze_embedder": False, "freeze_head": False,
            },
        )
        self.moment.init()                                               # replaces the reconstruction head with the forecasting head
        self.load_report: dict[str, Any] = {"backbone_init": self.backbone_init, "backbone_trainable": self.backbone_trainable,
                                             "moment": VARIANT_TO_MODEL[size]}
        if self.backbone_init == "timeran":
            if not model_cfg.get("timeran_checkpoint_path"):
                raise ValueError("backbone_init 'timeran' needs model.timeran_checkpoint_path")
            self.load_report.update(load_timeran_weights(self.moment, model_cfg["timeran_checkpoint_path"]))
        elif self.backbone_init == "random":
            encoder = self.moment.encoder
            encoder.apply(encoder._init_weights)                         # T5's own initializer
            self.moment.patch_embedding.value_embedding.reset_parameters()
            with torch.no_grad():
                self.moment.patch_embedding.mask_embedding.zero_()
        if self.backbone_trainable == "head":
            for name, parameter in self.moment.named_parameters():
                parameter.requires_grad = name.startswith("head.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TimeRAN expects input shaped (batch, lookback, features), got {tuple(x.shape)}.")
        batch_size, lookback, input_size = x.shape
        if lookback != self.input_sequence_length or input_size != self.input_size:
            raise ValueError(f"TimeRAN was configured for lookback {self.input_sequence_length} and {self.input_size} features, "
                             f"got {lookback} and {input_size}.")
        series = x.transpose(1, 2).contiguous()                          # (batch, channels, lookback)
        mask = torch.ones(batch_size, lookback, dtype=series.dtype, device=series.device)
        pad = (self.context_length - lookback) if self.context_length else 0
        if pad:
            series = torch.cat([series[..., :1].expand(-1, -1, pad), series], dim=-1)
            mask = torch.cat([torch.zeros(batch_size, pad, dtype=series.dtype, device=series.device), mask], dim=-1)
        forecast = self.moment(x_enc=series, input_mask=mask).forecast
        forecast = forecast.transpose(1, 2).contiguous()
        expected = (batch_size, self.prediction_horizon, self.input_size)
        if tuple(forecast.shape) != expected:
            raise ValueError(f"TimeRAN forecast shape {tuple(forecast.shape)} does not match {expected}.")
        return forecast
