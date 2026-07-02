#!/usr/bin/env python3
"""Compute distance-dependent spatial correlation for spectrum time series."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.make_fixed_grid_maps import (  # noqa: E402
    align_site_data,
    load_json,
    load_locations,
    load_site_frame,
    local_xy,
    node_to_site_name,
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def selected_groups(map_set: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    groups = list(map_set.get("groups", []))
    if not groups:
        raise ValueError("map-set has no groups")

    if requested:
        requested_set = set(requested)
        selected = [group for group in groups if group["name"] in requested_set]
        missing = requested_set - {group["name"] for group in selected}
        if missing:
            raise ValueError(f"unknown group(s): {', '.join(sorted(missing))}")
        return selected

    max_nodes = max(len(group.get("nodes", [])) for group in groups)
    if max_nodes < 3:
        raise ValueError("at least one group with 3 or more nodes is required")
    return [group for group in groups if len(group.get("nodes", [])) == max_nodes]


def site_coordinates(
    settings: dict[str, Any],
    locations: dict[str, dict[str, Any]],
    node_names: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    site_names = [node_to_site_name(node, settings, locations) for node in node_names]
    lats = np.asarray([float(locations[name]["latitude"]) for name in site_names], dtype=np.float64)
    lons = np.asarray([float(locations[name]["longitude"]) for name in site_names], dtype=np.float64)
    return site_names, lats, lons


def pair_distances(lats: np.ndarray, lons: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    origin_lat = float(np.mean(lats))
    origin_lon = float(np.mean(lons))
    x, y = local_xy(lons, lats, origin_lon, origin_lat)

    pairs = list(combinations(range(len(lats)), 2))
    distances = np.asarray([np.hypot(x[i] - x[j], y[i] - y[j]) for i, j in pairs], dtype=np.float64)
    return pairs, distances


def finite_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan"), float("nan")
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan"), float("nan")
    result = spearmanr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue)


def compute_band(
    settings: dict[str, Any],
    locations: dict[str, dict[str, Any]],
    data_root: Path,
    node_runs: dict[str, str],
    group: dict[str, Any],
    band: str,
    time_resolution: str,
    start_time: str | None,
    end_time: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_nodes = list(group["nodes"])
    missing_run_nodes = [node for node in group_nodes if node not in node_runs]
    if missing_run_nodes:
        raise ValueError(f"group {group['name']} has node(s) without run IDs: {', '.join(missing_run_nodes)}")

    frames = {node: load_site_frame(data_root, node, node_runs[node], band, time_resolution) for node in group_nodes}
    node_names, timestamps, freqs_mhz, data = align_site_data(frames, time_resolution, start_time, end_time)
    site_names, lats, lons = site_coordinates(settings, locations, node_names)
    pairs, distances_m = pair_distances(lats, lons)

    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    start = str(timestamps[0])
    end = str(timestamps[-1])

    for freq_index, freq_mhz in enumerate(freqs_mhz):
        pair_corrs = []
        for pair_index, (i, j) in enumerate(pairs):
            r, p = finite_spearman(data[i, :, freq_index], data[j, :, freq_index])
            pair_corrs.append(r)
            pair_rows.append(
                {
                    "group": group["name"],
                    "band": band,
                    "frequency_mhz": float(freq_mhz),
                    "node_i": node_names[i],
                    "node_j": node_names[j],
                    "site_i": site_names[i],
                    "site_j": site_names[j],
                    "distance_m": distances_m[pair_index],
                    "time_spearman_r": r,
                    "time_spearman_p": p,
                }
            )

        pair_corrs_array = np.asarray(pair_corrs, dtype=np.float64)
        spatial_r, spatial_p = finite_spearman(distances_m, pair_corrs_array)
        finite_pair_corrs = pair_corrs_array[np.isfinite(pair_corrs_array)]
        summary_rows.append(
            {
                "group": group["name"],
                "band": band,
                "frequency_mhz": float(freq_mhz),
                "n_nodes": len(node_names),
                "n_timestamps": len(timestamps),
                "n_pairs": len(pairs),
                "n_valid_pairs": int(finite_pair_corrs.size),
                "spatial_spearman_r": spatial_r,
                "spatial_spearman_p": spatial_p,
                "mean_pair_time_spearman_r": float(np.mean(finite_pair_corrs)) if finite_pair_corrs.size else float("nan"),
                "median_pair_time_spearman_r": float(np.median(finite_pair_corrs)) if finite_pair_corrs.size else float("nan"),
                "min_pair_time_spearman_r": float(np.min(finite_pair_corrs)) if finite_pair_corrs.size else float("nan"),
                "max_pair_time_spearman_r": float(np.max(finite_pair_corrs)) if finite_pair_corrs.size else float("nan"),
                "start_time": start,
                "end_time": end,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("evaluation/map_settings/powder_map_settings.json"),
        help="Dataset map settings JSON.",
    )
    parser.add_argument(
        "--map-set",
        type=Path,
        default=Path("evaluation/map_settings/powder_20260628_20260630_9node_map_set.json"),
        help="Map-set JSON with node run IDs, time range, bands, and groups.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override data root from settings.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/spatial_correlation"))
    parser.add_argument("--group", action="append", default=[], help="Group name to process; repeatable.")
    parser.add_argument("--band", action="append", default=[], help="Band to process; repeatable.")
    parser.add_argument("--start-time", default=None, help="Override inclusive UTC start timestamp.")
    parser.add_argument("--end-time", default=None, help="Override inclusive UTC end timestamp.")
    parser.add_argument("--write-pairs", action="store_true", help="Also write pair-level correlations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_json(resolve_path(args.settings))
    map_set = load_json(resolve_path(args.map_set))
    data_root = resolve_path(args.data_root or Path(settings["data_root"]))
    locations = load_locations(resolve_path(Path(settings["locations_json"])), settings["collection_key"])
    output_dir = resolve_path(args.output_dir)

    bands = args.band or map_set.get("bands") or settings.get("default_bands")
    if not bands:
        raise ValueError("no bands specified by --band, map-set, or settings")

    time_resolution = map_set.get("time_resolution", "minute")
    start_time = args.start_time or map_set.get("start_time")
    end_time = args.end_time or map_set.get("end_time")
    groups = selected_groups(map_set, args.group)

    all_summaries = []
    all_pairs = []
    for group in groups:
        if len(group.get("nodes", [])) < 3:
            print(f"Skipping {group['name']}: at least 3 nodes are needed", file=sys.stderr)
            continue
        for band in bands:
            summary, pairs = compute_band(
                settings=settings,
                locations=locations,
                data_root=data_root,
                node_runs=map_set["node_runs"],
                group=group,
                band=band,
                time_resolution=time_resolution,
                start_time=start_time,
                end_time=end_time,
            )
            all_summaries.append(summary)
            if args.write_pairs:
                all_pairs.append(pairs)
            print(f"Computed {group['name']} {band}: {len(summary)} frequency bins")

    if not all_summaries:
        raise ValueError("no groups were processed")

    output_dir.mkdir(parents=True, exist_ok=True)
    map_set_name = map_set.get("name", args.map_set.stem)
    summary_path = output_dir / f"{map_set_name}_spatial_correlation_by_frequency.csv"
    pd.concat(all_summaries, ignore_index=True).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    if args.write_pairs and all_pairs:
        pair_path = output_dir / f"{map_set_name}_pairwise_time_correlations.csv"
        pd.concat(all_pairs, ignore_index=True).to_csv(pair_path, index=False)
        print(f"Wrote {pair_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
