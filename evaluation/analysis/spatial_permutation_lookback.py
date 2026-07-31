#!/usr/bin/env python3
"""Run a 100-permutation geometry test for a direct IDW lookback forecast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SITES = ("cpg", "ebc", "humanities", "madsen", "moran", "sagepoint")
RUNS = {
    "cpg": "20260703T1839Z", "ebc": "20260703T1839Z",
    "humanities": "20260703T1839Z", "madsen": "20260703T1839Z",
    "moran": "20260703T1839Z", "sagepoint": "20260703T1839Z",
}
REGIONS = {
    1: (600.5, 607.5), 3: (622.5, 641.5), 4: (642.5, 646.5),
    5: (647.5, 655.5), 6: (656.5, 675.5), 8: (691.5, 728.5),
    9: (729.5, 734.5), 10: (735.5, 740.5), 11: (741.5, 745.5),
    12: (746.5, 755.5), 13: (756.5, 768.5), 14: (769.5, 776.5),
    16: (789.5, 794.5),
}


def xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (6371000.0 * np.radians(lon - lon.mean()) * np.cos(np.radians(lat.mean())),
            6371000.0 * np.radians(lat - lat.mean()))


def weights(stream_lons: np.ndarray, stream_lats: np.ndarray, target_lons: np.ndarray, target_lats: np.ndarray) -> np.ndarray:
    sx, sy = xy(stream_lons, stream_lats)
    tx, ty = xy(target_lons, target_lats)
    distance = np.hypot(tx[:, None] - sx[None, :], ty[:, None] - sy[None, :])
    result = 1.0 / np.maximum(distance, 1e-6) ** 2
    exact = distance == 0
    for row in range(len(target_lons)):
        if exact[row].any():
            result[row] = 0.0
            result[row, np.flatnonzero(exact[row])[0]] = 1.0
    return result / result.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/powder"))
    parser.add_argument("--locations", type=Path, default=Path("data/locations/powder.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/spatial_permutation"))
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--stride", type=int, default=15)
    args = parser.parse_args()

    location_data = json.loads(args.locations.read_text())
    endpoint_by_site = {}
    for endpoint in location_data["endpoints"]:
        name = endpoint["name"].lower().replace(" ", "")
        endpoint_by_site[name] = (float(endpoint["longitude"]), float(endpoint["latitude"]))
    aliases = {"cpg": "centralparkinggarage", "ebc": "ebc", "humanities": "humanities", "madsen": "madsen", "moran": "moran", "sagepoint": "sagepoint"}
    coords = np.asarray([endpoint_by_site[aliases[site]] for site in SITES], dtype=float)
    frames = []
    timestamps = []
    for site in SITES:
        path = args.data_root / f"{site}-nuc1" / RUNS[site] / "600_800" / "power_1mhz_avg_per_minute.csv"
        frame = pd.read_csv(path)
        stamps = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.floor("min")
        keep = (stamps >= pd.Timestamp("2026-07-03T18:39:00Z")) & (stamps <= pd.Timestamp("2026-07-06T09:14:00Z"))
        selected = frame.loc[keep].apply(pd.to_numeric, errors="coerce").interpolate(limit=9).ffill(limit=9).bfill(limit=9)
        frames.append(selected)
        timestamps.append(stamps.loc[keep].reset_index(drop=True))
    common = timestamps[0]
    for stamp in timestamps[1:]:
        common = common[common.isin(stamp)]
    common = common.sort_values().reset_index(drop=True)
    aligned = [frame.loc[stamp.isin(common)].reset_index(drop=True) for frame, stamp in zip(frames, timestamps)]
    values = np.stack([frame.to_numpy(dtype=float) for frame in aligned], axis=0)
    frequencies = np.asarray([float(column) for column in pd.read_csv(args.data_root / "cpg-nuc1" / RUNS["cpg"] / "600_800" / "power_1mhz_avg_per_minute.csv", nrows=0).columns if column != "timestamp_utc"])
    original = weights(coords[:, 0], coords[:, 1], coords[:, 0], coords[:, 1])
    rng = np.random.default_rng(42)
    permutations = [np.arange(len(SITES))] + [rng.permutation(len(SITES)) for _ in range(args.permutations)]
    rows = []
    max_horizon = 60
    for horizon in (1, 15, 60):
        starts = range(60, values.shape[1] - max_horizon, args.stride)
        truth = np.stack([values[:, start + horizon] for start in starts])
        for permutation_id, permutation in enumerate(permutations):
            assigned = coords[permutation]
            matrix = weights(assigned[:, 0], assigned[:, 1], coords[:, 0], coords[:, 1])
            errors = []
            for start_index, start in enumerate(starts):
                source_mean = values[:, start - 60:start].mean(axis=1)
                forecast = matrix @ source_mean
                errors.append(np.abs(forecast - truth[start_index]).mean(axis=0))
            error_by_frequency = np.mean(np.asarray(errors), axis=0)
            for frequency_index, error in enumerate(error_by_frequency):
                rows.append({"permutation": permutation_id, "horizon": horizon, "frequency_mhz": frequencies[frequency_index], "mae_db": error})
    result = pd.DataFrame(rows)
    original_rows = result[result.permutation == 0].rename(columns={"mae_db": "original_mae_db"})
    perm_rows = result[result.permutation > 0].merge(original_rows[["horizon", "frequency_mhz", "original_mae_db"]], on=["horizon", "frequency_mhz"])
    perm_rows["delta_error_db"] = perm_rows.mae_db - perm_rows.original_mae_db
    summaries = []
    for region_id, (start, end) in REGIONS.items():
        selected = perm_rows[perm_rows.frequency_mhz.between(start, end)]
        for horizon, group in selected.groupby("horizon"):
            summaries.append({"region_id": region_id, "horizon": horizon, "n": len(group), "median_delta_db": group.delta_error_db.median(), "q025_delta_db": group.delta_error_db.quantile(0.025), "q975_delta_db": group.delta_error_db.quantile(0.975), "positive_fraction": (group.delta_error_db > 0).mean()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "permutation_errors.csv", index=False)
    perm_rows.to_csv(args.output_dir / "permutation_deltas.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "permutation_region_summary.csv", index=False)
    print(f"wrote {len(result)} errors and {len(summaries)} region summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
