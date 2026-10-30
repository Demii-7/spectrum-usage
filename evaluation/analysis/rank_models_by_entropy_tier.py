#!/usr/bin/env python3
"""Rank saved forecast runs within low-, mid-, and high-entropy regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIERS = ("low", "mid", "high")
NEURAL_MODELS = {
    "autoformer_csa",
    "convlstm",
    "lstmattn",
    "residualconvlstm",
    "residualvanillalstm",
    "temporalconvnet",
    "timeran",
    "vanillalstm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--entropy-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="Collection run ID, such as 20260628T0436Z")
    parser.add_argument("--frequency-band", default="600_800")
    parser.add_argument("--entropy-column", default="diff_E_actual_normalized")
    return parser.parse_args()


def canonical_node(name: str) -> str:
    value = name.lower().replace(" ", "").replace("-nuc1", "")
    aliases = {"guesthouse": "guesthouse-nuc1", "humanities": "humanities-nuc1"}
    return aliases.get(value, f"{value}-nuc1")


def find_site_grid_indices(cache_path: Path) -> dict[str, tuple[int, int]]:
    with np.load(cache_path, allow_pickle=True) as archive:
        grid_x = np.asarray(archive["grid_x"], dtype=np.float64)
        grid_y = np.asarray(archive["grid_y"], dtype=np.float64)
        site_names = list(archive["site_names"])
        site_lons = np.asarray(archive["site_lons"], dtype=np.float64)
        site_lats = np.asarray(archive["site_lats"], dtype=np.float64)
        permutation = np.asarray(archive["position_permutation"], dtype=np.intp)
    site_lons = site_lons[permutation]
    site_lats = site_lats[permutation]
    origin_lon = float(np.mean(site_lons))
    origin_lat = float(np.mean(site_lats))
    radius = 6_371_000.0
    site_x = radius * np.radians(site_lons - origin_lon) * np.cos(np.radians(origin_lat))
    site_y = radius * np.radians(site_lats - origin_lat)
    result = {}
    for index, name in enumerate(site_names):
        distance = np.hypot(grid_x - site_x[index], grid_y - site_y[index])
        height, width = np.unravel_index(int(distance.argmin()), distance.shape)
        result[str(name)] = (int(height), int(width))
    return result


def source_identity(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    try:
        powder = parts.index("powder")
        return canonical_node(parts[powder + 1]), parts[powder + 2]
    except (ValueError, IndexError):
        return None


def archive_paths(metadata_path: Path) -> tuple[Path, Path]:
    stem = metadata_path.name.removesuffix("_metadata.json")
    return (
        metadata_path.parent / f"{stem}_predictions.npz",
        metadata_path.parent / f"{stem}_targets.npz",
    )


def inspect_forecast(metadata_path: Path, run_id: str, frequency_band: str) -> tuple[dict | None, str | None]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("chunk_id") != f"powder_{frequency_band}":
        return None, "different frequency chunk"

    predictions_path, targets_path = archive_paths(metadata_path)
    if not predictions_path.exists() or not targets_path.exists():
        return None, "missing prediction or target archive"
    with np.load(predictions_path) as predictions:
        horizon_keys = [key for key in predictions.files if key.startswith("t_plus_")]
        forecast_ndim = predictions[horizon_keys[0]].ndim if horizon_keys else 0

    test_paths = [
        Path(path)
        for key, path in sorted(metadata.get("data_files", {}).items())
        if key.startswith("test_")
    ]
    nodes: dict[str, dict] = {}
    if forecast_ndim == 4:
        if len(test_paths) != 1 or test_paths[0].suffix != ".npz" or not test_paths[0].exists():
            return None, "map forecast has no readable test-map archive"
        map_path = test_paths[0]
        with np.load(map_path, allow_pickle=True) as archive:
            map_metadata = json.loads(str(archive["metadata"].item()))
            sources = [Path(item["path"]) for item in map_metadata["files"]]
        grid_indices = find_site_grid_indices(map_path)
        for source, (site_name, grid_index) in zip(sources, grid_indices.items()):
            identity = source_identity(source)
            if identity and identity[1] == run_id:
                nodes[identity[0]] = {"grid_index": grid_index, "site_name": str(site_name)}
    elif forecast_ndim == 2:
        offset = 0
        for path in test_paths:
            identity = source_identity(path)
            if path.suffix != ".csv" or not path.exists():
                continue
            length = len(pd.read_csv(path, usecols=[0]))
            if identity and identity[1] == run_id:
                nodes[identity[0]] = {"row_range": (offset, offset + length)}
            offset += length
    else:
        return None, f"unsupported forecast dimensionality {forecast_ndim}"

    if not nodes:
        return None, "different collection run"
    horizons = sorted(int(key.removeprefix("t_plus_")) for key in horizon_keys)
    if not horizons:
        return None, "no forecast horizons"
    return {
        "metadata_path": metadata_path,
        "metadata": metadata,
        "predictions_path": predictions_path,
        "targets_path": targets_path,
        "model": str(metadata.get("model", "unknown")).lower(),
        "run": str(metadata_path.parent.parent),
        "nodes": nodes,
        "horizons": horizons,
    }, None


def region_mae(
    forecast: dict,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_rows: np.ndarray | None,
    node: str,
    frequency_mask: np.ndarray,
    horizon: int,
) -> float:
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch in {forecast['run']} at horizon {horizon}")

    node_info = forecast["nodes"][node]
    if predictions.ndim == 4:
        height, width = node_info["grid_index"]
        error = predictions[:, frequency_mask, height, width] - targets[:, frequency_mask, height, width]
    elif predictions.ndim == 2:
        start, end = node_info["row_range"]
        if target_rows is None:
            raise ValueError(f"Missing target rows in {forecast['run']}")
        sample_mask = (target_rows >= start) & (target_rows < end)
        error = predictions[sample_mask][:, frequency_mask] - targets[sample_mask][:, frequency_mask]
    else:
        raise ValueError(f"Unsupported forecast shape {predictions.shape} in {forecast['run']}")
    if error.size == 0:
        raise ValueError(f"No errors for {node} in {forecast['run']}")
    return float(np.mean(np.abs(error)))


def main() -> None:
    args = parse_args()
    entropy = pd.read_csv(args.entropy_table, dtype={"run_id": str, "band": str})
    entropy = entropy.loc[
        (entropy["record_type"] == "region_median_trace")
        & (entropy["mode"] == "raw_dbm")
        & (entropy["run_id"] == args.run_id)
        & (entropy["band"] == args.frequency_band)
    ].copy()
    required = {"node", "region_id", "region_start_mhz", "region_end_mhz", args.entropy_column}
    missing = required - set(entropy.columns)
    if missing:
        raise ValueError(f"Entropy table is missing: {', '.join(sorted(missing))}")
    entropy = entropy.dropna(subset=[args.entropy_column])
    entropy["node"] = entropy["node"].map(canonical_node)

    forecasts = []
    skipped = []
    for metadata_path in sorted(args.runs_dir.glob("**/forecasts/*_metadata.json")):
        try:
            forecast, reason = inspect_forecast(metadata_path, args.run_id, args.frequency_band)
        except Exception as exc:
            forecast, reason = None, f"inspection failed: {exc}"
        if forecast:
            forecasts.append(forecast)
        else:
            skipped.append({"metadata": str(metadata_path), "reason": reason})
    if not forecasts:
        raise ValueError("No compatible forecast archives found")

    models = sorted({forecast["model"] for forecast in forecasts})
    common_nodes = set(entropy["node"])
    for model in models:
        model_nodes = set().union(*(set(item["nodes"]) for item in forecasts if item["model"] == model))
        common_nodes &= model_nodes
    if not common_nodes:
        raise ValueError("Compatible model architectures have no common entropy sites")
    entropy = entropy.loc[entropy["node"].isin(common_nodes)].sort_values(
        [args.entropy_column, "node", "region_id"]
    )
    entropy["entropy_tier"] = pd.qcut(entropy[args.entropy_column], 3, labels=TIERS)
    tier_bounds = {
        tier: {
            "minimum": float(group[args.entropy_column].min()),
            "maximum": float(group[args.entropy_column].max()),
            "regions": int(len(group)),
        }
        for tier, group in entropy.groupby("entropy_tier", observed=True)
    }

    region_rows = []
    for forecast in forecasts:
        frequencies = np.asarray(forecast["metadata"]["frequencies_mhz"], dtype=float)
        for horizon in forecast["horizons"]:
            key = f"t_plus_{horizon}"
            with np.load(forecast["predictions_path"]) as archive:
                predictions = archive[key]
                target_rows = archive[f"rows_{key}"] if f"rows_{key}" in archive else None
            with np.load(forecast["targets_path"]) as archive:
                targets = archive[key]
            for _, region in entropy.iterrows():
                node = region["node"]
                if node not in forecast["nodes"]:
                    continue
                frequency_mask = (
                    (frequencies >= float(region["region_start_mhz"]))
                    & (frequencies <= float(region["region_end_mhz"]))
                )
                if not frequency_mask.any():
                    continue
                region_rows.append({
                    "model": forecast["model"],
                    "run": forecast["run"],
                    "horizon": horizon,
                    "entropy_tier": str(region["entropy_tier"]),
                    "node": node,
                    "region_id": int(region["region_id"]),
                    "entropy": float(region[args.entropy_column]),
                    "mae_db": region_mae(
                        forecast, predictions, targets, target_rows, node, frequency_mask, horizon
                    ),
                })
    regions = pd.DataFrame(region_rows)
    expected_regions = entropy.groupby("entropy_tier", observed=True).size().to_dict()
    run_scores = regions.groupby(["model", "run", "horizon", "entropy_tier"], as_index=False).agg(
        mean_mae_db=("mae_db", "mean"),
        region_count=("mae_db", "size"),
    )
    run_scores = run_scores.loc[
        run_scores.apply(lambda row: row["region_count"] == expected_regions[row["entropy_tier"]], axis=1)
    ]
    selected = run_scores.loc[
        run_scores.groupby(["model", "horizon", "entropy_tier"])["mean_mae_db"].idxmin()
    ].copy()
    selected["rank"] = selected.groupby(["horizon", "entropy_tier"])["mean_mae_db"].rank(
        method="min", ascending=True
    ).astype(int)
    selected["entropy_tier"] = pd.Categorical(selected["entropy_tier"], categories=TIERS, ordered=True)
    rankings = selected.sort_values(["horizon", "entropy_tier", "rank", "model"])[
        ["horizon", "entropy_tier", "rank", "model", "run", "mean_mae_db", "region_count"]
    ]
    winners = rankings.loc[rankings["rank"] == 1].copy()
    neural = rankings.loc[rankings["model"].isin(NEURAL_MODELS)].rename(columns={"rank": "overall_rank"}).copy()
    neural["neural_rank"] = neural.groupby(["horizon", "entropy_tier"])["mean_mae_db"].rank(
        method="min", ascending=True
    ).astype(int)
    neural_winners = neural.loc[neural["neural_rank"] == 1][
        [
            "horizon", "entropy_tier", "neural_rank", "overall_rank", "model",
            "run", "mean_mae_db", "region_count",
        ]
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output_dir / "entropy_band_model_rankings.csv", index=False)
    winners.to_csv(args.output_dir / "entropy_band_winners.csv", index=False)
    neural_winners.to_csv(args.output_dir / "entropy_band_neural_winners.csv", index=False)
    provenance = {
        "source": "existing saved forecast predictions; no models were trained",
        "runs_directory": str(args.runs_dir),
        "entropy_table": str(args.entropy_table),
        "entropy_column": args.entropy_column,
        "entropy_tiers": "equal-count tertiles over common regions",
        "tier_bounds": tier_bounds,
        "collection_run_id": args.run_id,
        "frequency_band": args.frequency_band,
        "common_nodes": sorted(common_nodes),
        "architecture_selection": "lowest mean region MAE run per model, horizon, and entropy tier",
        "neural_models": sorted(NEURAL_MODELS),
        "compatible_forecasts": [str(item["metadata_path"]) for item in forecasts],
        "skipped_forecasts": skipped,
    }
    (args.output_dir / "entropy_band_rankings_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Ranked {rankings['model'].nunique()} models across {rankings['horizon'].nunique()} horizons")
    print(f"Common nodes: {', '.join(sorted(common_nodes))}")
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
