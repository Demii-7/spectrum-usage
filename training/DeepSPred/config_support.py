"""DeepSPred configuration translation for standalone and shared runners."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def resolve_deepspred_config(config: dict[str, Any], n_bins: int) -> dict[str, Any]:
    """Return the model-ready config, accepting nested and legacy settings."""
    source = config.get("deepspred", config)
    model = deepcopy(source.get("model") or config.get("model") or {})
    train = deepcopy(source.get("train") or source.get("training") or {})

    def setting(name: str, default: Any) -> Any:
        return train.get(name, source.get(name, default))

    frames = source.get("frames") or config.get("frames") or {}
    preprocessing = source.get("preprocessing") or config.get("preprocessing") or {}
    windowing = source.get("windowing") or config.get("windowing") or {}
    minutes = int(frames.get("minutes_per_frame", model.get("minutes_per_frame", source.get("minutes_per_frame", 60))))
    w_pad = int(frames.get("w_pad", model.get("w_pad", source.get("w_pad", 256))))
    input_frames = int(windowing.get("input_frames", model.get("input_frames", source.get("input_frames", 1))))
    output_frames = int(windowing.get("output_frames", model.get("output_frames", source.get("output_frames", 1))))
    stride = int(windowing.get("stride", setting("train_stride", source.get("stride", 1))))

    if minutes <= 0 or input_frames <= 0 or output_frames <= 0 or stride <= 0:
        raise ValueError("DeepSPred frame and window dimensions must be positive")
    if w_pad < n_bins:
        raise ValueError(f"DeepSPred w_pad ({w_pad}) must be at least input width ({n_bins})")
    if not model:
        raise ValueError("DeepSPred model configuration is required")
    patch_width = int(model.get("patch_size", [1, 1, 1])[-1])
    width_multiple = patch_width * 4  # patch embedding followed by two spatial merges
    if w_pad % width_multiple:
        raise ValueError(f"DeepSPred w_pad ({w_pad}) must be divisible by {width_multiple}")

    resolved_train = deepcopy(train)
    for name, default in (
        ("batch_size", 2), ("epochs", 6), ("learning_rate", 0.001),
        ("weight_decay", 0.05), ("early_stopping_epochs", 4),
    ):
        resolved_train[name] = setting(name, default)
    resolved_train["device"] = setting("device", config.get("training", {}).get("device", "auto"))

    return {
        "preprocessing": {
            "colormap": str(preprocessing.get("colormap", source.get("colormap", "jet"))),
            "normalization": str(preprocessing.get("normalization", source.get("normalization", "minmax"))),
        },
        "frames": {"minutes_per_frame": minutes, "w_pad": w_pad, "w_orig": int(n_bins)},
        "windowing": {
            "input_frames": input_frames, "output_frames": output_frames, "stride": stride,
        },
        "model": model,
        "train": resolved_train,
    }
