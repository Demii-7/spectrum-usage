"""Train/evaluate a parameter-free LookbackMean reference (1D, 2D or 4D) through Ray."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .config import inject_parameters
from .optional import require_ray
from .registry import get_model_spec
from .replicate import _test_metrics
from .robustness import DEFAULT_MAX_FAILURES, build_or_restore, run_config, summarize_results
from .trainable import TuneReporter, run_integrated_trial

MODELS = ("lookbackmean1d", "lookbackmean2d", "lookbackmean4d")
# The model is deterministic and parameter-free, so one seed is enough.
DEFAULT_SEEDS = (42,)
PARAMETERS: dict[str, Any] = {}


def launch(args: argparse.Namespace) -> None:
    require_ray()
    from ray import tune

    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = get_model_spec(args.model)

    def trainable(values):
        seed = int(values["seed"])
        trial_directory = Path(tune.get_context().get_trial_dir())
        reporter = TuneReporter()
        config = inject_parameters(base_config, args.model, PARAMETERS, seed=seed)
        config.setdefault("training", {})["device"] = "cpu"
        run_integrated_trial(config, args.model, PARAMETERS, seed, trial_directory, reporter=reporter)
        test = _test_metrics(config, args.model, trial_directory)
        tune.report({**(reporter.selected_metrics or {}), **test, "seed": seed, "completed": True},
                    checkpoint=tune.Checkpoint.from_directory(str(trial_directory / "integrated")))

    resourced = tune.with_resources(trainable, spec.resources.as_ray())
    experiment_name = f"{args.model}-eval"

    def make_tuner():
        return tune.Tuner(
            resourced,
            param_space={"seed": tune.grid_search(list(args.seeds))},
            tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent),
            run_config=run_config(experiment_name, args.storage_path, max_failures=args.max_failures),
        )

    tuner = build_or_restore(
        f"{args.storage_path.rstrip('/')}/{experiment_name}", resourced, make_tuner, resume=args.resume
    )
    results = tuner.fit()
    failed = [result.path for result in results if result.error is not None]
    if failed:
        raise RuntimeError(f"{args.model} evaluation trials failed: {failed}")
    summarize_results(results, label=experiment_name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="spectrum-usage")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--endpoint", default="http://minio:9000")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--storage-path", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    return parser.parse_args(argv)


def main(argv=None):
    import os

    from .storage import MinIOConfig

    args = parse_args(argv)
    os.environ.update(MinIOConfig(args.bucket, args.prefix, args.endpoint).environment())
    args.storage_path = args.storage_path or f"s3://{args.bucket}/{args.prefix.strip('/')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launch(args)


if __name__ == "__main__":
    main()
