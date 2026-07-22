import math
from typing import Any


def max_abs_prediction_by_horizon(predictions: Any) -> list[float]:
    """Return batch-wide maximum absolute predictions for each horizon."""
    if predictions.ndim < 2:
        raise ValueError("predictions must include batch and horizon dimensions")
    return [
        float(predictions[:, horizon].abs().max().item())
        for horizon in range(predictions.shape[1])
    ]


def prediction_guard_status(
    max_abs_by_horizon: list[float], threshold: float
) -> tuple[bool, str]:
    """Return whether an epoch may participate in model selection."""
    if not all(math.isfinite(value) for value in max_abs_by_horizon):
        return False, "nonfinite_predictions"
    if max(max_abs_by_horizon, default=0.0) > threshold:
        return False, "prediction_magnitude_exceeded"
    return True, "valid"


def prediction_diagnostic_log(max_abs_by_horizon: list[float]) -> dict[str, float]:
    return {
        f"val_max_abs_prediction_t{horizon}": value
        for horizon, value in enumerate(max_abs_by_horizon, 1)
    }


def prediction_guard_threshold(train_config: dict[str, Any]) -> float:
    if "validation_prediction_magnitude_threshold" not in train_config:
        raise ValueError(
            "ConvLSTM training requires validation_prediction_magnitude_threshold "
            "in normalized prediction units"
        )
    threshold = float(train_config["validation_prediction_magnitude_threshold"])
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("validation_prediction_magnitude_threshold must be finite and > 0")
    return threshold
