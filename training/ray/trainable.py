"""In-process Ray Train adapter for integrated training callbacks."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

from .candidate import candidate_id
from .config import inject_parameters
from .optional import require_ray
from .registry import FALLBACK_OBJECTIVE, PRIMARY_OBJECTIVE


ReportFunction = Callable[..., None]


class TuneReporter:
    """Log every event and report only prunable forecasting epochs to ASHA."""

    def __init__(self, report: ReportFunction | None = None,
                 event_logger: Callable[[Mapping[str, Any]], None] | None = None):
        self._report = report
        self._event_logger = event_logger
        self.events: list[dict[str, Any]] = []
        self.last_metrics: dict[str, Any] | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.report_count = 0
        self.used_fallback = False

    def _ray_report(self, metrics: Mapping[str, Any], checkpoint: Any = None) -> None:
        if self._report is not None:
            self._report(dict(metrics), checkpoint=checkpoint)
            return
        require_ray()
        from ray import tune

        tune.report(dict(metrics), checkpoint=checkpoint)

    def __call__(self, event: Mapping[str, Any]) -> None:
        stored = dict(event)
        stored["metrics"] = dict(event.get("metrics") or {})
        self.events.append(stored)
        if self._event_logger is not None:
            self._event_logger(stored)
        if not bool(event.get("prunable", False)):
            return
        if not bool(event.get("selection_eligible", True)):
            return

        values = stored["metrics"]
        if PRIMARY_OBJECTIVE in values:
            objective_name = PRIMARY_OBJECTIVE
        elif FALLBACK_OBJECTIVE in values:
            objective_name = FALLBACK_OBJECTIVE
            self.used_fallback = True
        else:
            raise KeyError(
                f"Prunable event has neither {PRIMARY_OBJECTIVE!r} nor {FALLBACK_OBJECTIVE!r}"
            )
        self.report_count += 1
        metrics = {
            str(name): float(value)
            for name, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        metrics.update({
            "objective": float(values[objective_name]),
            "objective_name": objective_name,
            "training_iteration": self.report_count,
            "forecast_epoch": int(event["epoch"]),
            "stage": str(event.get("stage", "train")),
        })
        self.last_metrics = metrics
        if bool(stored.get("is_best", False)) or (
            self.best_metrics is None
            or metrics["objective"] < self.best_metrics["objective"]
        ):
            self.best_metrics = dict(metrics)
        self._ray_report(metrics)

    def complete(self, checkpoint: Any) -> None:
        """Attach the finished integrated checkpoint to the final Ray result."""
        if self.last_metrics is None or self.best_metrics is None:
            raise RuntimeError("Integrated training completed without a prunable forecasting event")
        selected_metrics = dict(self.best_metrics)
        selected_metrics.update({
            "completed": True,
            "selected_training_iteration": int(self.best_metrics["training_iteration"]),
            "training_iteration": self.report_count + 1,
            "forecast_epoch": int(self.best_metrics["forecast_epoch"]),
        })
        selected_metrics.update({
            f"final_{name}": value
            for name, value in self.last_metrics.items()
            if name != "training_iteration"
        })
        selected_metrics["final_training_iteration"] = int(
            self.last_metrics["training_iteration"]
        )
        self._ray_report(selected_metrics, checkpoint=checkpoint)

    @property
    def selected_metrics(self) -> dict[str, Any] | None:
        return self.best_metrics or self.last_metrics

    @property
    def warnings(self) -> list[str]:
        if not self.used_fallback:
            return []
        return [
            f"{PRIMARY_OBJECTIVE} was not supplied by the trainer; ASHA used {FALLBACK_OBJECTIVE}. "
            "This objective is not directly comparable in physical dB across model families."
        ]


def _ray_checkpoint(run_directory: Path):
    require_ray()
    from ray.tune import Checkpoint

    return Checkpoint.from_directory(str(run_directory))


def run_integrated_trial(
    base_config: Mapping[str, Any],
    model_name: str,
    parameters: Mapping[str, Any],
    seed: int,
    trial_directory: Path,
    *,
    trainer: Callable[..., None] | None = None,
    reporter: TuneReporter | None = None,
    checkpoint_factory: Callable[[Path], Any] = _ray_checkpoint,
) -> dict[str, Any]:
    """Run one ASHA trial directly through the integrated Python API."""
    import yaml

    config = inject_parameters(base_config, model_name, parameters, seed=seed)
    trial_directory.mkdir(parents=True, exist_ok=True)
    (trial_directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    run_directory = trial_directory / "integrated"
    adapter = reporter or TuneReporter()
    event_log = trial_directory / "training_events.jsonl"

    def callback(event: Mapping[str, Any]) -> None:
        with event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event), sort_keys=True) + "\n")
        adapter(event)

    if trainer is None:
        from training.common.train_integrated import train_one_model

        trainer = train_one_model
    trainer(config, model_name, run_directory, callback=callback)

    checkpoints = sorted((run_directory / "checkpoints").glob("*.pt"))
    if not checkpoints:
        raise RuntimeError("Integrated training produced no .pt checkpoint")
    adapter.complete(checkpoint_factory(run_directory))
    result = {
        "candidate_id": candidate_id(model_name, parameters),
        "trial_id": candidate_id(model_name, parameters, seed),
        "model": model_name,
        "seed": int(seed),
        "objective": float(adapter.selected_metrics["objective"]),
        "objective_name": adapter.selected_metrics["objective_name"],
        "warnings": adapter.warnings,
        "event_log": str(event_log),
        "prunable_epochs_reported": adapter.report_count,
        "checkpoints": [str(path) for path in checkpoints],
        "checkpoint_format": "integrated:model_state_dict+normalization+frequencies+training_results",
    }
    (trial_directory / "result_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_non_asha_subprocess(
    config_path: Path, output_directory: Path, *, python: str = sys.executable
) -> None:
    """Optional isolated runner for diagnostics only; it cannot support ASHA."""
    import subprocess

    subprocess.run(
        [python, "-m", "training.common.train_integrated", "--config", str(config_path),
         "--output-dir", str(output_directory)],
        check=True,
    )
