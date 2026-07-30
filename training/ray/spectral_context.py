"""Run full-context versus region-only spectral ablations for one 2D model."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from training.ray.config import inject_parameters
from training.ray.optional import require_ray
from training.ray.replicate import SEEDS, _s3_filesystem, resolve_winner
from training.ray.storage import MinIOConfig
from training.ray.trainable import TuneReporter, run_integrated_trial


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIONS = ROOT / "evaluation/analysis/plan_regions_600_800.csv"
NOISE_RANGES = [[608.5, 621.5], [676.5, 690.5], [777.5, 788.5], [795.5, 799.5]]


def non_noise_regions(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"band_id", "start_mhz", "end_mhz", "is_noise_floor"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Region table is missing columns: {sorted(missing)}")
    noise = frame["is_noise_floor"].astype(str).str.lower().isin({"true", "1"})
    return frame.loc[~noise].to_dict(orient="records")


def regional_config(
    base_config: dict[str, Any],
    region: dict[str, Any],
    condition: str,
    seed: int,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    bounds = [[float(region["start_mhz"]), float(region["end_mhz"])]]
    config["data"]["loss_frequency_ranges"] = bounds
    if condition == "region_only":
        config["data"]["mask"] = {
            "frequency_ranges": bounds,
            "replacement": "low_tail_gaussian",
            "calibration_frequency_ranges": NOISE_RANGES,
            "low_tail_quantile": 0.05,
            "minimum_noise_std_db": 0.05,
            "seed": int(seed),
        }
    elif condition == "full_context":
        config["data"].pop("mask", None)
    else:
        raise ValueError(f"Unknown spectral-context condition: {condition}")
    return config


def _test_region_metrics(
    config: dict[str, Any],
    model_name: str,
    run_directory: Path,
    region: dict[str, Any],
) -> dict[str, float | int]:
    from training.common.evaluation_integrated import evaluate_one_model

    checkpoint = next((run_directory / "integrated/checkpoints").glob("*.pt"), None)
    if checkpoint is None:
        raise RuntimeError(f"No checkpoint found for {model_name}")
    output = run_directory / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    evaluate_one_model(config, model_name, output, checkpoint, skip_plots=True)
    metadata_paths = list((output / "forecasts").glob("*_metadata.json"))
    if len(metadata_paths) != 1:
        raise RuntimeError(f"Expected one forecast metadata file, found {len(metadata_paths)}")
    metadata_path = metadata_paths[0]
    stem = metadata_path.name.removesuffix("_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frequencies = np.asarray(metadata["frequencies_mhz"], dtype=float)
    selected = (
        (frequencies >= float(region["start_mhz"]) - 1e-6)
        & (frequencies <= float(region["end_mhz"]) + 1e-6)
    )
    metrics: dict[str, float | int] = {}
    mae_values = []
    rmse_values = []
    with np.load(output / "forecasts" / f"{stem}_predictions.npz") as predictions, np.load(
        output / "forecasts" / f"{stem}_targets.npz"
    ) as targets:
        for horizon in sorted(int(value) for value in config["windowing"]["horizons"]):
            prediction = np.asarray(predictions[f"t_plus_{horizon}"])[:, selected]
            target = np.asarray(targets[f"t_plus_{horizon}"])[:, selected]
            error = prediction.astype(np.float64) - target.astype(np.float64)
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(error**2)))
            metrics[f"test_mae_db_t{horizon}"] = mae
            metrics[f"test_rmse_db_t{horizon}"] = rmse
            metrics[f"test_n_values_t{horizon}"] = int(error.size)
            mae_values.append(mae)
            rmse_values.append(rmse)
    metrics["test_mean_horizon_mae_db"] = float(np.mean(mae_values))
    metrics["test_mean_horizon_rmse_db"] = float(np.mean(rmse_values))
    return metrics


def _campaign_trainable(base_config, model_name, parameters, regions):
    require_ray()
    from ray import tune

    region_by_id = {str(region["band_id"]): region for region in regions}

    def trainable(values):
        seed = int(values["seed"])
        condition = str(values["condition"])
        region = region_by_id[str(values["band_id"])]
        configured = regional_config(base_config, region, condition, seed)
        resolved = inject_parameters(configured, model_name, parameters, seed=seed)
        trial_directory = Path(tune.get_context().get_trial_dir())
        reporter = TuneReporter()
        run_integrated_trial(
            configured, model_name, parameters, seed, trial_directory, reporter=reporter,
        )
        test = _test_region_metrics(resolved, model_name, trial_directory, region)
        checkpoint = tune.Checkpoint.from_directory(str(trial_directory / "integrated"))
        tune.report({
            **(reporter.selected_metrics or {}),
            **test,
            "model_name": model_name,
            "seed": seed,
            "condition": condition,
            "band_id": str(region["band_id"]),
            "start_mhz": float(region["start_mhz"]),
            "end_mhz": float(region["end_mhz"]),
            "completed": True,
        }, checkpoint=checkpoint)

    return trainable


def launch(args: argparse.Namespace) -> None:
    require_ray()
    from ray import tune
    from ray.tune import RunConfig

    os.environ.update(MinIOConfig(args.bucket, args.prefix, args.endpoint).environment())
    filesystem = _s3_filesystem(args.endpoint)
    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    regions = non_noise_regions(args.regions)
    parameters, _, source_metrics = resolve_winner(
        args.search_path, args.model, filesystem,
    )
    trainable = _campaign_trainable(base_config, args.model, parameters, regions)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "campaign_plan.json").write_text(json.dumps({
        "model": args.model,
        "conditions": ["full_context", "region_only"],
        "regions": regions,
        "seeds": list(SEEDS),
        "parameters": parameters,
        "source_search_path": args.search_path,
        "source_objective": source_metrics.get("objective"),
        "gpu_per_task": args.gpus_per_task,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            {"cpu": args.cpus_per_task, "gpu": args.gpus_per_task},
        ),
        param_space={
            "condition": tune.grid_search(["full_context", "region_only"]),
            "band_id": tune.grid_search([str(region["band_id"]) for region in regions]),
            "seed": tune.grid_search(list(SEEDS)),
        },
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent),
        run_config=RunConfig(name=f"{args.model}-spectral-context", storage_path=args.storage_path),
    )
    results = tuner.fit()
    failed = [result.path for result in results if result.error is not None]
    if failed:
        raise RuntimeError(f"Spectral-context trials failed: {failed}")
    rows = [dict(result.metrics) for result in results if result.metrics.get("completed") is True]
    frame = pd.DataFrame(rows)
    expected = 2 * len(regions) * len(SEEDS)
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} completed rows, found {len(frame)}")
    columns = [
        "model_name", "seed", "condition", "band_id", "start_mhz", "end_mhz",
        "objective", "objective_name", "test_mean_horizon_mae_db",
        "test_mean_horizon_rmse_db", "test_mae_db_t1", "test_rmse_db_t1",
        "test_mae_db_t15", "test_rmse_db_t15", "test_mae_db_t60", "test_rmse_db_t60",
        "test_n_values_t1", "test_n_values_t15", "test_n_values_t60",
    ]
    frame = frame[columns].sort_values(["band_id", "condition", "seed"])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.table.with_suffix(args.table.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(args.table)
    print(f"Wrote {len(frame)} rows to {args.table}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--search-path", required=True)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--bucket", default="spectrum-usage")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--endpoint", default="http://minio:9000")
    parser.add_argument("--storage-path", default=None)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--cpus-per-task", type=float, default=2.0)
    parser.add_argument("--gpus-per-task", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0.0 < args.gpus_per_task <= 0.5:
        parser.error("--gpus-per-task must be greater than zero and at most 0.5")
    args.storage_path = args.storage_path or f"s3://{args.bucket}/{args.prefix.strip('/')}"
    return args


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
