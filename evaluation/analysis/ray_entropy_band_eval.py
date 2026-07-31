#!/usr/bin/env python3
"""Evaluate replicated 2D checkpoints by annotated band and training entropy."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from evaluation.analysis.compute_region_entropy import metrics as entropy_metrics
from training.common.data import chunk_specs, load_chunk
from training.common.windowing import make_window_starts
from training.ray.config import inject_parameters
from training.ray.optional import require_ray
from training.ray.replicate import SEEDS, _s3_filesystem
from training.ray.storage import MinIOConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIONS = ROOT / "evaluation/analysis/plan_regions_600_800.csv"
DEFAULT_OUTPUT = ROOT / "evaluation/results/tables/entropy_band_metrics_2d.csv"
CAMPAIGN_ROOT = "spectrum-usage/spectrum-usage/ray/production-20260729-replicate-v2"
NEURAL_MODELS = (
    "autoformer_csa",
    "lstmattn",
    "residualvanillalstm",
    "temporalconvnet",
    "vanillalstm",
)
LINEAR_MODELS = ("linearar2d", "residuallinearar2d")
MODEL_CONFIGS = {
    **{name: "training/configs/ray_tuning_2d.yaml" for name in NEURAL_MODELS},
    **{name: "training/configs/ray_tuning_linear_2d.yaml" for name in LINEAR_MODELS},
}
ENTROPY_COLUMNS = (
    "E_unc_normalized",
    "E_actual_normalized",
    "diff_E_actual_normalized",
    "weighted_transition_entropy_normalized",
    "temporal_structure_score",
    "fano_predictability",
)
METRIC_NAMES = (
    "mae_db",
    "rmse_db",
    "mse_db2",
    "median_ae_db",
    "p90_ae_db",
    "bias_db",
)


def load_regions(path: Path) -> pd.DataFrame:
    regions = pd.read_csv(path)
    required = {
        "band_id", "start_mhz", "end_mhz", "bin_count",
        "behavior_category", "is_noise_floor",
    }
    missing = required - set(regions.columns)
    if missing:
        raise ValueError(f"Region table is missing columns: {sorted(missing)}")
    regions = regions.sort_values("start_mhz").reset_index(drop=True)
    if regions["band_id"].duplicated().any():
        raise ValueError("Region band IDs must be unique")
    expected_counts = np.rint(regions["end_mhz"] - regions["start_mhz"] + 1).astype(int)
    if not np.array_equal(expected_counts, regions["bin_count"].to_numpy(dtype=int)):
        raise ValueError("Region bin counts do not match inclusive one-MHz bounds")
    if len(regions) > 1 and not np.allclose(
        regions["start_mhz"].to_numpy(dtype=float)[1:],
        regions["end_mhz"].to_numpy(dtype=float)[:-1] + 1.0,
    ):
        raise ValueError("Regions must be contiguous and non-overlapping")
    return regions


def _region_mask(frequencies: np.ndarray, region: pd.Series) -> np.ndarray:
    return (
        (frequencies >= float(region["start_mhz"]) - 1e-6)
        & (frequencies <= float(region["end_mhz"]) + 1e-6)
    )


def compute_training_band_entropy(
    config: dict[str, Any], regions: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute site-median entropy using only the configured fitting interval."""
    chunk = chunk_specs(config)[0]
    model_name = str(config["training"]["models"][0])
    val_fraction = float(config[model_name]["train"].get("val_fraction", 0.1))
    loaded = load_chunk(config, chunk, val_fraction)
    split = loaded.splits[loaded.train_split]
    frequencies = np.asarray(loaded.frequencies, dtype=float)
    output: dict[str, dict[str, Any]] = {}

    for _, region in regions.iterrows():
        frequency_mask = _region_mask(frequencies, region)
        matched = int(np.count_nonzero(frequency_mask))
        if matched != int(region["bin_count"]):
            raise ValueError(
                f"{region['band_id']} expected {int(region['bin_count'])} bins, found {matched}"
            )
        site_values: list[dict[str, float | int]] = []
        for segment in split.segments:
            block = split.raw_dbm[segment.start:segment.end, :][:, frequency_mask]
            trace = np.nanmedian(block, axis=1)
            site_values.append(entropy_metrics(trace, np.asarray(
                (-135, -130, -125, -120, -115, -110, -105, -100, -95),
                dtype=float,
            )))
        if not site_values:
            raise ValueError("Training split has no receiver segments")
        row: dict[str, Any] = {
            "band_id": str(region["band_id"]),
            "start_mhz": float(region["start_mhz"]),
            "end_mhz": float(region["end_mhz"]),
            "bin_count": matched,
            "behavior_category": str(region["behavior_category"]),
            "is_noise_floor": bool(region["is_noise_floor"]),
            "entropy_receiver_count": len(site_values),
            "entropy_train_timesteps_per_receiver": min(
                segment.end - segment.start for segment in split.segments
            ),
            "entropy_definition": "diff_E_actual_normalized",
            "entropy_scope": "T4_training_interval_only",
        }
        for column in ENTROPY_COLUMNS:
            row[column] = float(np.nanmedian([float(value[column]) for value in site_values]))
        row["entropy"] = row["diff_E_actual_normalized"]
        output[row["band_id"]] = row
    return output


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if not np.isfinite(error).all():
        raise ValueError("Forecast errors contain non-finite values")
    absolute = np.abs(error)
    squared = error**2
    return {
        "n_values": int(error.size),
        "mae_db": float(np.mean(absolute)),
        "rmse_db": float(np.sqrt(np.mean(squared))),
        "mse_db2": float(np.mean(squared)),
        "median_ae_db": float(np.median(absolute)),
        "p90_ae_db": float(np.quantile(absolute, 0.9)),
        "bias_db": float(np.mean(error)),
    }


def summarize_forecasts(
    predictions: dict[int, np.ndarray],
    targets: dict[int, np.ndarray],
    frequencies: np.ndarray,
    regions: pd.DataFrame,
    entropy: dict[str, dict[str, Any]],
    *,
    model: str,
    seed: int,
    deterministic: bool = False,
) -> list[dict[str, Any]]:
    if set(predictions) != set(targets):
        raise ValueError("Prediction and target horizons differ")
    rows: list[dict[str, Any]] = []
    for _, region in regions.iterrows():
        band_id = str(region["band_id"])
        mask = _region_mask(np.asarray(frequencies, dtype=float), region)
        if int(mask.sum()) != int(region["bin_count"]):
            raise ValueError(f"Forecast frequencies do not cover {band_id}")
        row = {
            "model_name": model,
            "seed": int(seed),
            "deterministic": bool(deterministic),
            **entropy[band_id],
        }
        for horizon in sorted(predictions):
            prediction = np.asarray(predictions[horizon])[:, mask]
            target = np.asarray(targets[horizon])[:, mask]
            values = error_metrics(prediction, target)
            row[f"t{horizon}_n_forecast_origins"] = int(prediction.shape[0])
            for name, value in values.items():
                row[f"t{horizon}_{name}"] = value
        for metric in METRIC_NAMES:
            row[f"mean_horizon_{metric}"] = float(np.mean([
                row[f"t{horizon}_{metric}"] for horizon in predictions
            ]))
        rows.append(row)
    return rows


def _forecast_archives(output_dir: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    forecast_dir = output_dir / "forecasts"
    metadata_paths = list(forecast_dir.glob("*_metadata.json"))
    if len(metadata_paths) != 1:
        raise RuntimeError(f"Expected one forecast metadata file, found {len(metadata_paths)}")
    metadata_path = metadata_paths[0]
    stem = metadata_path.name.removesuffix("_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(forecast_dir / f"{stem}_predictions.npz") as archive:
        predictions = {
            int(key.removeprefix("t_plus_")): np.asarray(archive[key])
            for key in archive.files if key.startswith("t_plus_")
        }
    with np.load(forecast_dir / f"{stem}_targets.npz") as archive:
        targets = {
            int(key.removeprefix("t_plus_")): np.asarray(archive[key])
            for key in archive.files if key.startswith("t_plus_")
        }
    return predictions, targets, np.asarray(metadata["frequencies_mhz"], dtype=float)


def _campaign_result(seed: int, model: str, campaign_root: str, endpoint: str):
    require_ray()
    from ray import tune

    filesystem = _s3_filesystem(endpoint)
    experiment = f"{campaign_root.rstrip('/')}/{model}/{model}-replicate"
    results = tune.ResultGrid(tune.ExperimentAnalysis(experiment, storage_filesystem=filesystem))
    matches = [
        result for result in results
        if int(result.config.get("seed", -1)) == seed
        and result.metrics.get("completed") is True
        and result.error is None
        and result.checkpoint is not None
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one completed checkpoint for {model} seed {seed}, found {len(matches)}")
    return matches[0]


def evaluate_campaign_seed(
    model: str,
    seed: int,
    parameters: dict[str, Any],
    campaign_root: str,
    endpoint: str,
    regions_records: list[dict[str, Any]],
    entropy: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ray worker entrypoint for one existing model checkpoint."""
    import torch

    from training.common.evaluation_integrated import evaluate_one_model

    config_path = ROOT / MODEL_CONFIGS[model]
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = inject_parameters(base_config, model, parameters, seed=seed)
    config["training"]["device"] = "cuda"
    torch.set_num_threads(2)
    result = _campaign_result(seed, model, campaign_root, endpoint)
    regions = pd.DataFrame(regions_records)

    with tempfile.TemporaryDirectory(prefix=f"entropy-eval-{model}-{seed}-") as temporary:
        root = Path(temporary)
        integrated = root / "integrated"
        result.checkpoint.to_directory(str(integrated))
        checkpoints = list((integrated / "checkpoints").glob("*.pt"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"Expected one checkpoint for {model} seed {seed}, found {len(checkpoints)}")
        output = root / "evaluation"
        output.mkdir()
        evaluate_one_model(config, model, output, checkpoints[0], skip_plots=True)
        predictions, targets, frequencies = _forecast_archives(output)
        return summarize_forecasts(
            predictions, targets, frequencies, regions, entropy, model=model, seed=seed,
        )


def evaluate_lookback_mean(
    config: dict[str, Any],
    regions: pd.DataFrame,
    entropy: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate the deterministic 60-minute lookback mean on matching origins."""
    chunk = chunk_specs(config)[0]
    model_name = str(config["training"]["models"][0])
    val_fraction = float(config[model_name]["train"].get("val_fraction", 0.1))
    loaded = load_chunk(config, chunk, val_fraction)
    split = loaded.splits[loaded.test_split]
    horizons = [int(value) for value in config["windowing"]["horizons"]]
    lookback = int(config["windowing"]["lookback"])
    starts = make_window_starts(
        len(split.raw_dbm), lookback, max(horizons), 1, split.segments,
    )
    prediction_parts = []
    for offset in range(0, len(starts), 1024):
        batch_starts = starts[offset:offset + 1024]
        indices = batch_starts[:, None] + np.arange(lookback)[None, :]
        prediction_parts.append(np.mean(split.raw_dbm[indices], axis=1))
    prediction = np.concatenate(prediction_parts).astype(np.float32)
    predictions = {horizon: prediction for horizon in horizons}
    targets = {
        horizon: split.raw_dbm[starts + lookback + horizon - 1]
        for horizon in horizons
    }
    rows = summarize_forecasts(
        predictions,
        targets,
        np.asarray(loaded.frequencies, dtype=float),
        regions,
        entropy,
        model="lookbackmean2d",
        seed=SEEDS[0],
        deterministic=True,
    )
    return [
        {**row, "seed": seed}
        for seed in SEEDS
        for row in rows
    ]


def launch(args: argparse.Namespace) -> None:
    require_ray()
    import ray

    os.environ.update(MinIOConfig(args.bucket, "", args.endpoint).environment())
    regions = load_regions(args.regions)
    base_config = yaml.safe_load((ROOT / MODEL_CONFIGS[NEURAL_MODELS[0]]).read_text(encoding="utf-8"))
    entropy = compute_training_band_entropy(base_config, regions)
    regions_records = regions.to_dict(orient="records")

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    remote_worker = ray.remote(evaluate_campaign_seed)
    references = []
    for model in (*NEURAL_MODELS, *LINEAR_MODELS):
        plan_path = (
            ROOT / "runs/ray/production-20260729-replicate-v2"
            / model / "campaign_plan.json"
        )
        parameters = json.loads(plan_path.read_text(encoding="utf-8"))["parameters"]
        for seed in SEEDS:
            references.append(remote_worker.options(
                num_cpus=args.cpus_per_task,
                num_gpus=args.gpus_per_task,
            ).remote(
                model, seed, parameters, args.campaign_root, args.endpoint,
                regions_records, entropy,
            ))

    rows: list[dict[str, Any]] = []
    for result_rows in ray.get(references):
        rows.extend(result_rows)
    rows.extend(evaluate_lookback_mean(base_config, regions, entropy))

    frame = pd.DataFrame(rows).sort_values(["model_name", "seed", "start_mhz"])
    expected_rows = (len(NEURAL_MODELS) + len(LINEAR_MODELS) + 1) * len(SEEDS) * len(regions)
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} output rows, produced {len(frame)}")
    keys = ["model_name", "seed", "band_id"]
    if frame.duplicated(keys).any():
        raise RuntimeError(f"Output contains duplicate keys: {keys}")
    leading_columns = ["model_name", "seed", "band_id", "entropy"]
    frame = frame[leading_columns + [
        column for column in frame.columns if column not in leading_columns
    ]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(args.output)
    print(f"Wrote {len(frame)} rows to {args.output}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--campaign-root", default=CAMPAIGN_ROOT)
    parser.add_argument("--bucket", default="spectrum-usage")
    parser.add_argument("--endpoint", default="http://minio:9000")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--cpus-per-task", type=float, default=2.0)
    parser.add_argument("--gpus-per-task", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0.0 < args.gpus_per_task <= 0.5:
        parser.error("--gpus-per-task must be greater than 0 and at most 0.5")
    if args.cpus_per_task <= 0:
        parser.error("--cpus-per-task must be positive")
    return args


def main() -> None:
    launch(parse_args())


if __name__ == "__main__":
    main()
