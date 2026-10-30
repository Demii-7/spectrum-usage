"""Cost and hardware metadata recorded for every training run.

Collection is best-effort: a failure here must never fail a training run, so
problems are recorded as ``cost_error`` instead of raised.
"""

from __future__ import annotations

import json
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.common.forecasting import forecast

INFERENCE_BATCH = 32
INFERENCE_WARMUP = 3
INFERENCE_REPEATS = 10


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "cost_parameters_total": sum(p.numel() for p in model.parameters()),
        "cost_parameters_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }


def _ray_assigned_gpu() -> float | None:
    try:
        import ray

        if ray.is_initialized():
            return float(ray.get_runtime_context().get_assigned_resources().get("GPU", 0.0))
    except Exception:
        pass
    return None


def hardware_context() -> dict[str, Any]:
    assigned = _ray_assigned_gpu()
    context: dict[str, Any] = {
        "cost_hostname": socket.gethostname(),
        "cost_torch_version": torch.__version__,
        "cost_cuda_version": torch.version.cuda or "none",
        "cost_gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if assigned is not None:
        context["cost_ray_gpu_assigned"] = assigned
        context["cost_gpu_shared"] = 0.0 < assigned < 1.0
    return context


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**20


def inference_timing(
    model: nn.Module,
    example_x: torch.Tensor,
    prediction_horizon: int,
    rollout_horizon: int,
    device: torch.device,
) -> dict[str, float]:
    """Median wall time of the shared forecast call on one small batch."""
    was_training = model.training
    model.eval()
    x = example_x[:INFERENCE_BATCH].to(device)
    timings = []
    try:
        with torch.no_grad():
            for step in range(INFERENCE_WARMUP + INFERENCE_REPEATS):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                forecast(model=model, x=x, prediction_horizon=prediction_horizon,
                         rollout_horizon=rollout_horizon, targets=None)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if step >= INFERENCE_WARMUP:
                    timings.append(time.perf_counter() - start)
    finally:
        model.train(was_training)
    median = statistics.median(timings)
    return {
        "cost_inference_batch_size": int(len(x)),
        "cost_inference_ms_per_batch": median * 1e3,
        "cost_inference_ms_per_sample": median * 1e3 / len(x),
    }


def collect_run_costs(
    model: nn.Module,
    example_x: torch.Tensor | None,
    prediction_horizon: int,
    rollout_horizon: int,
    device: torch.device,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Any]:
    costs: dict[str, Any] = {}
    try:
        costs.update(parameter_counts(model))
        peak = peak_memory_mb()
        if peak is not None:
            costs["cost_peak_gpu_memory_mb"] = peak
        costs.update(hardware_context())
        costs["cost_model_config_json"] = json.dumps(model_config, sort_keys=True, default=str)
        costs["cost_train_config_json"] = json.dumps(train_config, sort_keys=True, default=str)
        if example_x is not None:
            costs.update(inference_timing(
                model, example_x, prediction_horizon, rollout_horizon, device
            ))
    except Exception as error:  # cost metadata must never fail a training run
        costs["cost_error"] = f"{type(error).__name__}: {error}"
    return costs


def write_run_costs(directory: Path, chunk_id: str, costs: dict[str, Any]) -> Path:
    path = Path(directory) / f"{chunk_id}_run_costs.json"
    path.write_text(json.dumps(costs, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return path


def read_run_costs(directory: Path) -> dict[str, Any]:
    """Load the run's cost record, or an empty dict if none was written."""
    files = sorted(Path(directory).rglob("*_run_costs.json"))
    return json.loads(files[0].read_text(encoding="utf-8")) if files else {}
