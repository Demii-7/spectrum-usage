#!/usr/bin/env python3
"""Build an annotation-region entropy and MAE comparison table from a model run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--entropy-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--band", default=None)
    parser.add_argument("--entropy-column", default="diff_E_actual_normalized")
    parser.add_argument("--lookback-mae-column", default="mae_db")
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


def forecast_artifacts(run_dir: Path) -> tuple[dict, np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    forecast_dir = run_dir / "forecasts"
    metadata_paths = sorted(forecast_dir.glob("*_metadata.json"))
    if len(metadata_paths) != 1:
        raise ValueError(f"Expected one forecast metadata file in {forecast_dir}, found {len(metadata_paths)}")
    metadata_path = metadata_paths[0]
    stem = metadata_path.name.removesuffix("_metadata.json")
    prediction_path = forecast_dir / f"{stem}_predictions.npz"
    target_path = forecast_dir / f"{stem}_targets.npz"
    if not prediction_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Missing prediction or target archive for {stem}")
    return (
        json.loads(metadata_path.read_text(encoding="utf-8")),
        np.load(prediction_path),
        np.load(target_path),
    )


def filter_entropy_table(table: pd.DataFrame, run_id: str | None, band: str | None) -> pd.DataFrame:
    selected = table.copy()
    if run_id is not None:
        selected = selected.loc[selected["run_id"].astype(str) == run_id]
    if band is not None:
        selected = selected.loc[selected["band"].astype(str) == band]
    return selected


def prepare_table(
    entropy_table: pd.DataFrame,
    metadata: dict,
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    entropy_column: str,
    lookback_mae_column: str,
    site_grids: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    required = {
        "node",
        "run_id",
        "band",
        "region_id",
        "region_start_mhz",
        "region_end_mhz",
        entropy_column,
        lookback_mae_column,
    }
    missing = required - set(entropy_table.columns)
    if missing:
        raise ValueError(f"Entropy table is missing column(s): {', '.join(sorted(missing))}")

    frequencies = np.asarray(metadata["frequencies_mhz"], dtype=float)
    if predictions.shape != targets.shape or predictions.shape[1] != len(frequencies):
        raise ValueError("Forecast arrays do not match each other or the metadata frequencies")
    if predictions.ndim not in (2, 4):
        raise ValueError(f"Unsupported forecast layout {predictions.shape}; expected N,F or N,F,H,W")
    if predictions.ndim == 4 and not site_grids:
        raise ValueError("Map forecasts require at least one --site-grid NODE=H,W")
    if predictions.ndim == 2 and entropy_table["node"].nunique() != 1:
        raise ValueError("Vector forecasts require entropy rows for one node")

    rows = []
    for _, annotation in entropy_table.iterrows():
        node = str(annotation["node"])
        mask = (
            (frequencies >= float(annotation["region_start_mhz"]))
            & (frequencies <= float(annotation["region_end_mhz"]))
        )
        if not mask.any():
            continue
        if predictions.ndim == 4:
            if node not in site_grids:
                continue
            height, width = site_grids[node]
            model_mae = np.mean(np.abs(predictions[:, mask, height, width] - targets[:, mask, height, width]))
        else:
            model_mae = np.mean(np.abs(predictions[:, mask] - targets[:, mask]))
        rows.append(
            {
                "band": f"{node}:R{annotation['region_id']}",
                "node": node,
                "run_id": annotation["run_id"],
                "frequency_band": annotation["band"],
                "region_id": annotation["region_id"],
                "region_start_mhz": annotation["region_start_mhz"],
                "region_end_mhz": annotation["region_end_mhz"],
                "entropy": annotation[entropy_column],
                "lookback_mae_db": annotation[lookback_mae_column],
                "model_mae_db": float(model_mae),
                "model": metadata.get("model", "model"),
            }
        )
    if not rows:
        raise ValueError("No annotation regions matched the forecast frequencies and site metadata")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    metadata, prediction_archive, target_archive = forecast_artifacts(args.run_dir)
    key = f"t_plus_{args.horizon}"
    if key not in prediction_archive or key not in target_archive:
        raise ValueError(f"Forecast archives do not contain {key}")
    entropy_table = filter_entropy_table(
        pd.read_csv(args.entropy_table, dtype={"run_id": str, "band": str}),
        args.run_id,
        args.band,
    )
    output = prepare_table(
        entropy_table,
        metadata,
        prediction_archive[key],
        target_archive[key],
        entropy_column=args.entropy_column,
        lookback_mae_column=args.lookback_mae_column,
        site_grids=parse_site_grids(args.site_grid),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} regions to {args.output}")


if __name__ == "__main__":
    main()
