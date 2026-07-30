"""Train and evaluate the parameter-free LookbackMean1D baseline on all frequency bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .config import inject_parameters
from .optional import require_ray
from .registry import get_model_spec
from .trainable import TuneReporter, run_integrated_trial


SEEDS = (40, 41, 42, 43, 44)

MODEL_NAME = "lookbackmean1d"
PARAMETERS = {}


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
    test_metrics = metrics[metrics["split"].map(lambda v: str(v) == "test" or str(v).endswith("_test"))]
    if test_metrics.empty:
        raise RuntimeError(f"No test metrics produced for {model_name}")
    output: dict[str, float] = {}
    for horizon, frame in test_metrics.groupby("horizon"):
        output[f"test_mae_db_t{int(horizon)}"] = float(frame["mae_db"].mean())
        output[f"test_rmse_db_t{int(horizon)}"] = float(frame["rmse_db"].mean())
    output["test_mean_horizon_mae_db"] = float(test_metrics.groupby("horizon")["mae_db"].mean().mean())
    return output


def launch(args: argparse.Namespace) -> None:
    require_ray()
    from ray import tune
    from ray.tune import RunConfig

    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = get_model_spec(MODEL_NAME)

    def trainable(values):
        seed = int(values["seed"])
        trial_directory = Path(tune.get_context().get_trial_dir())
        reporter = TuneReporter()
        config = inject_parameters(base_config, MODEL_NAME, PARAMETERS, seed=seed)
        run_integrated_trial(config, MODEL_NAME, PARAMETERS, seed, trial_directory, reporter=reporter)
        test = _test_metrics(config, MODEL_NAME, trial_directory)
        tune.report({**(reporter.selected_metrics or {}), **test, "seed": seed, "completed": True},
                    checkpoint=tune.Checkpoint.from_directory(str(trial_directory / "integrated")))

    tuner = tune.Tuner(
        tune.with_resources(trainable, spec.resources.as_ray()),
        param_space={"seed": tune.grid_search(list(SEEDS))},
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent),
        run_config=RunConfig(name=f"{MODEL_NAME}-all-bins", storage_path=args.storage_path),
    )
    results = tuner.fit()
    failed = [result.path for result in results if result.error is not None]
    if failed:
        raise RuntimeError(f"Replication trials failed: {failed}")
    print(f"All {len(SEEDS)} seeds completed for {MODEL_NAME}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="spectrum-usage")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--endpoint", default="http://minio:9000")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--storage-path", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.storage_path = args.storage_path or f"s3://{args.bucket}/{args.prefix.strip('/')}"
    launch(args)


if __name__ == "__main__":
    main()
