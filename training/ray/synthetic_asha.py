"""Small synthetic ASHA/checkpoint test used during cluster acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import uuid


def run(storage_path: str, *, gpu: float = 0.5) -> dict[str, object]:
    from ray import tune
    from ray.tune import Checkpoint, RunConfig
    from ray.tune.schedulers import ASHAScheduler

    def trial(config):
        quality = float(config["quality"])
        for iteration in range(1, 5):
            score = quality + 1.0 / iteration
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.json"
                path.write_text(
                    json.dumps({"iteration": iteration, "quality": quality, "score": score}),
                    encoding="utf-8",
                )
                tune.report(
                    {"score": score, "training_iteration": iteration},
                    checkpoint=Checkpoint.from_directory(directory),
                )
            time.sleep(0.2)

    name = f"synthetic-asha-{uuid.uuid4().hex[:8]}"
    tuner = tune.Tuner(
        tune.with_resources(trial, {"cpu": 1, "gpu": gpu}),
        param_space={"quality": tune.grid_search([0.0, 1.0, 2.0, 3.0])},
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                time_attr="training_iteration",
                metric="score",
                mode="min",
                max_t=4,
                grace_period=1,
                reduction_factor=2,
            ),
            max_concurrent_trials=4,
        ),
        run_config=RunConfig(name=name, storage_path=storage_path),
    )
    results = tuner.fit()
    successful = [result for result in results if result.error is None]
    if len(successful) != 4:
        raise RuntimeError(f"Expected four successful/pruned results, got {len(successful)}")
    best = results.get_best_result(metric="score", mode="min")
    if best.checkpoint is None:
        raise RuntimeError("Best synthetic ASHA result has no durable checkpoint")
    with best.checkpoint.as_directory() as directory:
        restored = json.loads(
            (Path(directory) / "checkpoint.json").read_text(encoding="utf-8")
        )
    if float(restored["quality"]) != float(best.config["quality"]):
        raise RuntimeError(f"Restored checkpoint does not match best trial: {restored}")
    iterations = [int(result.metrics.get("training_iteration", 0)) for result in successful]
    if not any(iteration < 4 for iteration in iterations):
        raise RuntimeError(f"ASHA did not prune any trial: iterations={iterations}")
    return {
        "experiment": name,
        "storage_path": storage_path,
        "result_count": len(successful),
        "iterations": iterations,
        "best_quality": float(best.config["quality"]),
        "best_score": float(best.metrics["score"]),
        "restored_checkpoint": restored,
        "checkpoint": str(best.checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-path",
        default="s3://spectrum-usage/acceptance",
    )
    parser.add_argument("--gpu", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(run(args.storage_path, gpu=args.gpu), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
