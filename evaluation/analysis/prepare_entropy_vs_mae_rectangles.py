#!/usr/bin/env python3
"""Build an annotation-region entropy and MAE table from two model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lookback-run-dir", type=Path, required=True)
    parser.add_argument("--entropy-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--band", default=None)
    parser.add_argument("--entropy-column", default="diff_E_actual_normalized")
    parser.add_argument(
        "--site-grid",
        action="append",
        default=[],
        metavar="NODE=H,W",
        help="Map a node to its grid cell for map forecasts. Repeatable.",
    )
    return parser.parse_args()


def parse_site_grids(values: list[str]) -> dict[str, tuple[int, int]]:
    grids = {}
    for value in values:
        try:
            node, coordinates = value.split("=", 1)
            height, width = (int(item) for item in coordinates.split(",", 1))
        except ValueError as exc:
            raise ValueError(f"Invalid --site-grid {value!r}; expected NODE=H,W") from exc
        grids[node] = (height, width)
    return grids


def load_forecast(run_dir: Path, horizon: int) -> dict:
    forecast_dir = run_dir / "forecasts"
    metadata_paths = sorted(forecast_dir.glob("*_metadata.json"))
    if len(metadata_paths) != 1:
        raise ValueError(f"Expected one forecast metadata file in {forecast_dir}, found {len(metadata_paths)}")
    metadata_path = metadata_paths[0]
    stem = metadata_path.name.removesuffix("_metadata.json")
    key = f"t_plus_{horizon}"
    predictions = np.load(forecast_dir / f"{stem}_predictions.npz")
    targets = np.load(forecast_dir / f"{stem}_targets.npz")
    if key not in predictions or key not in targets:
        raise ValueError(f"Forecast archives in {run_dir} do not contain {key}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "metadata": metadata,
        "predictions": predictions[key],
        "targets": targets[key],
        "target_rows": predictions[f"rows_{key}"],
        "frequencies": np.asarray(metadata["frequencies_mhz"], dtype=float),
    }


def node_row_ranges(metadata: dict) -> dict[str, tuple[int, int]]:
    ranges = {}
    offset = 0
    test_files = [
        Path(path)
        for key, path in sorted(metadata.get("data_files", {}).items())
        if key.startswith("test_") and str(path).endswith(".csv")
    ]
    for path in test_files:
        length = len(pd.read_csv(path, usecols=[0]))
        ranges[path.parents[2].name] = (offset, offset + length)
        offset += length
    return ranges


def region_mae(
    forecast: dict,
    node: str,
    frequency_mask: np.ndarray,
    site_grids: dict[str, tuple[int, int]],
) -> float:
    predictions = forecast["predictions"]
    targets = forecast["targets"]
    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes differ")
    if predictions.ndim == 4:
        if node not in site_grids:
            raise ValueError(f"Map forecast requires --site-grid for {node}")
        height, width = site_grids[node]
        error = predictions[:, frequency_mask, height, width] - targets[:, frequency_mask, height, width]
    elif predictions.ndim == 2:
        ranges = node_row_ranges(forecast["metadata"])
        if node not in ranges:
            raise ValueError(f"Could not map vector forecast rows to node {node}")
        start, end = ranges[node]
        sample_mask = (forecast["target_rows"] >= start) & (forecast["target_rows"] < end)
        error = predictions[sample_mask][:, frequency_mask] - targets[sample_mask][:, frequency_mask]
    else:
        raise ValueError(f"Unsupported forecast layout {predictions.shape}")
    if error.size == 0:
        raise ValueError(f"No forecast errors found for {node}")
    return float(np.mean(np.abs(error)))


def prepare_table(
    entropy: pd.DataFrame,
    model_forecast: dict,
    lookback_forecast: dict,
    entropy_column: str,
    site_grids: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    required = {
        "node", "run_id", "band", "region_id", "region_start_mhz",
        "region_end_mhz", entropy_column,
    }
    missing = required - set(entropy.columns)
    if missing:
        raise ValueError(f"Entropy table is missing column(s): {', '.join(sorted(missing))}")
    if not np.array_equal(model_forecast["frequencies"], lookback_forecast["frequencies"]):
        raise ValueError("Model and LookbackMean runs use different frequencies")

    frequencies = model_forecast["frequencies"]
    rows = []
    for _, region in entropy.iterrows():
        frequency_mask = (
            (frequencies >= float(region["region_start_mhz"]))
            & (frequencies <= float(region["region_end_mhz"]))
        )
        if not frequency_mask.any():
            continue
        node = str(region["node"])
        rows.append({
            "band": f"{node}:R{region['region_id']}",
            "node": node,
            "run_id": region["run_id"],
            "frequency_band": region["band"],
            "region_id": region["region_id"],
            "region_start_mhz": region["region_start_mhz"],
            "region_end_mhz": region["region_end_mhz"],
            "entropy": region[entropy_column],
            "lookback_mae_db": region_mae(lookback_forecast, node, frequency_mask, site_grids),
            "model_mae_db": region_mae(model_forecast, node, frequency_mask, site_grids),
            "model": model_forecast["metadata"].get("model", "model"),
        })
    if not rows:
        raise ValueError("No entropy regions matched the run frequencies")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    entropy = pd.read_csv(args.entropy_table, dtype={"run_id": str, "band": str})
    if "record_type" in entropy:
        entropy = entropy.loc[entropy["record_type"] == "region_median_trace"]
    if "mode" in entropy:
        entropy = entropy.loc[entropy["mode"] == "raw_dbm"]
    if args.run_id is not None:
        entropy = entropy.loc[entropy["run_id"] == args.run_id]
    if args.band is not None:
        entropy = entropy.loc[entropy["band"] == args.band]

    output = prepare_table(
        entropy,
        load_forecast(args.run_dir, args.horizon),
        load_forecast(args.lookback_run_dir, args.horizon),
        args.entropy_column,
        parse_site_grids(args.site_grid),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} regions to {args.output}")


if __name__ == "__main__":
    main()
