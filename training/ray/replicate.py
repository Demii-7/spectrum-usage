"""Run a fixed HPO winner across five reproducibility seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import yaml

from .config import inject_parameters
from .optional import require_ray
from .registry import get_model_spec
from .storage import MinIOConfig
from .trainable import TuneReporter, run_integrated_trial


SEEDS = (40, 41, 42, 43, 44)


def _is_test_split(value: object) -> bool:
    """Accept the loader's canonical ``<reference>_test`` split name."""
    name = str(value)
    return name == "test" or name.endswith("_test")


def _source_validation_metrics(source_metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep numeric validation metrics without coercing objective metadata."""
    output: dict[str, Any] = {}
    for key, value in source_metrics.items():
        if key == "objective_name":
            output[key] = str(value)
        elif key.startswith("val_") or key in {"objective", "forecast_epoch"}:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                output[key] = float(value)
    return output


def _s3_filesystem(endpoint: str | None):
    import pyarrow.fs as pafs

    parsed = urlparse(endpoint or os.environ.get("AWS_ENDPOINT_URL", ""))
    host = parsed.netloc or parsed.path
    return pafs.S3FileSystem(
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_override=host or None,
        scheme=parsed.scheme or "http",
        region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def resolve_winner(search_path: str, model_name: str, filesystem) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    require_ray()
    from ray import tune

    analysis = tune.ExperimentAnalysis(search_path, storage_filesystem=filesystem)
    results = tune.ResultGrid(analysis)
    spec = get_model_spec(model_name, require_hpo=True)
    eligible = [
        result for result in results
        if result.metrics.get("completed") is True
        and result.metrics.get("objective_name") == spec.objective
        and result.metrics.get("objective") is not None
        and np.isfinite(float(result.metrics["objective"]))
        and result.checkpoint is not None
    ]
    if not eligible:
        raise RuntimeError(f"No completed physical-dB checkpoint found in {search_path}")
    winner = min(eligible, key=lambda result: (float(result.metrics["objective"]), result.path))
    parameters = dict(winner.config)
    architecture = parameters.get("architecture")
    if isinstance(architecture, str):
        token = architecture.removeprefix("historical:")
        if token in spec.anchors:
            parameters["architecture"] = dict(spec.anchors[token]["architecture"])
        elif token in spec.historical:
            parameters["architecture"] = dict(spec.historical[token]["architecture"])
        else:
            raise ValueError(f"Unknown persisted architecture token {architecture!r}")
    parameters.pop("seed", None)
    parameters.pop("capacity", None)
    return parameters, winner.checkpoint, dict(winner.metrics)


def _test_metrics(config: dict[str, Any], model_name: str, run_directory: Path) -> dict[str, float]:
    from training.common.evaluation_integrated import evaluate_one_model

    checkpoint = next((run_directory / "integrated" / "checkpoints").glob("*.pt"), None)
    if checkpoint is None:
        raise RuntimeError(f"No checkpoint found for {model_name} in {run_directory}")
    evaluation_dir = run_directory / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    evaluate_one_model(config, model_name, evaluation_dir, checkpoint, skip_plots=True)
    import pandas as pd

    metrics = pd.read_csv(evaluation_dir / "aggregate_metrics.csv")
    metrics = metrics[metrics["split"].map(_is_test_split)]
    if metrics.empty:
        raise RuntimeError(f"No test metrics produced for {model_name}")
    output: dict[str, float] = {}
    for horizon, frame in metrics.groupby("horizon"):
        output[f"test_mae_db_t{int(horizon)}"] = float(frame["mae_db"].mean())
        output[f"test_rmse_db_t{int(horizon)}"] = float(frame["rmse_db"].mean())
    output["test_mean_horizon_mae_db"] = float(metrics.groupby("horizon")["mae_db"].mean().mean())
    return output


def _campaign_trainable(base_config, model_name, parameters, source_checkpoint, source_metrics):
    require_ray()
    from ray import tune

    def trainable(values):
        seed = int(values["seed"])
        trial_directory = Path(tune.get_context().get_trial_dir())
        if seed == 42:
            trial_directory.mkdir(parents=True, exist_ok=True)
            source_checkpoint.to_directory(str(trial_directory / "integrated"))
            validation = _source_validation_metrics(source_metrics)
            test = _test_metrics(inject_parameters(base_config, model_name, parameters, seed=seed), model_name, trial_directory)
            tune.report({**validation, **test, "seed": seed, "completed": True},
                        checkpoint=tune.Checkpoint.from_directory(str(trial_directory / "integrated")))
            return

        reporter = TuneReporter()
        config = inject_parameters(base_config, model_name, parameters, seed=seed)
        run_integrated_trial(config, model_name, parameters, seed, trial_directory, reporter=reporter)
        test = _test_metrics(config, model_name, trial_directory)
        tune.report({**(reporter.selected_metrics or {}), **test, "seed": seed, "completed": True},
                    checkpoint=tune.Checkpoint.from_directory(str(trial_directory / "integrated")))

    return trainable


def launch(args: argparse.Namespace) -> None:
    require_ray()
    from ray import tune
    from ray.tune import RunConfig

    os.environ.update(MinIOConfig(args.bucket, args.prefix, args.endpoint).environment())
    filesystem = _s3_filesystem(args.endpoint)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config[args.model]["train"]["epochs"] = args.epochs
    parameters, source_checkpoint, source_metrics = resolve_winner(args.search_path, args.model, filesystem)
    trainable = _campaign_trainable(config, args.model, parameters, source_checkpoint, source_metrics)
    campaign = {
        "model": args.model,
        "seeds": list(args.seeds),
        "parameters": parameters,
        "source_search_path": args.search_path,
        "source_objective": source_metrics.get("objective"),
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "campaign_plan.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tuner = tune.Tuner(
        tune.with_resources(trainable, get_model_spec(args.model).resources.as_ray()),
        param_space={"seed": tune.grid_search(list(args.seeds))},
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent),
        run_config=RunConfig(name=f"{args.model}-replicate", storage_path=args.storage_path),
    )
    results = tuner.fit()
    failed = [result.path for result in results if result.error is not None]
    if failed:
        raise RuntimeError(f"Replication trials failed: {failed}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--search-path", required=True, help="Ray ExperimentAnalysis path without s3://")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="spectrum-usage")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--endpoint", default="http://minio:9000")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--storage-path", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.storage_path = args.storage_path or f"s3://{args.bucket}/{args.prefix.strip('/')}"
    launch(args)


if __name__ == "__main__":
    main()
