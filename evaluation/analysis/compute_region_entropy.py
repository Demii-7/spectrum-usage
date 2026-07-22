#!/usr/bin/env python3
"""Compute quantized temporal entropy and predictability for selected POWDER traces.

E_actual uses the LZ78 phrase-count entropy-rate estimate c*log2(c)/n,
where c is the number of phrases.  Fano predictability is obtained by solving
H = h2(1-Pi) + (1-Pi)*log2(Q-1) on Pi in [1/Q, 1].
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NODES = ("guesthouse-nuc1", "humanities-nuc1")
RUNS = ("20260618T0036Z", "20260628T0436Z")
BANDS = ("600_800", "2400_2600")
DEFAULT_EDGES = (-135, -130, -125, -120, -115, -110, -105, -100, -95)
DEFAULT_DIFF_EDGES = (-20, -10, -5, -2, 0, 2, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "powder")
    parser.add_argument("--nodes", nargs="+", default=NODES)
    parser.add_argument("--runs", nargs="+", default=RUNS)
    parser.add_argument("--bands", nargs="+", default=BANDS)
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=ROOT.parent / "data" / "annotations",
    )
    parser.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES,
                        help="Shared raw-dBm bin edges; centered mode subtracts their median.")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results/statistical_analysis/region_entropy.csv")
    parser.add_argument("--metadata-json", type=Path, default=ROOT / "results/statistical_analysis/region_entropy_metadata.json")
    return parser.parse_args()


def quantize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign [edge_i, edge_i+1) states, clipping both tails to Q states."""
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2).astype(np.int16)


def shannon_entropy(states: np.ndarray, q: int) -> float:
    counts = np.bincount(states, minlength=q)
    probabilities = counts[counts > 0] / len(states)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def weighted_transition_entropy(states: np.ndarray, q: int) -> float:
    """Weight next-state surprisal by normalized quantized-state distance."""
    if len(states) < 2:
        return math.nan
    previous = states[:-1]
    current = states[1:]
    counts = np.zeros((q, q), dtype=float)
    np.add.at(counts, (previous, current), 1.0)
    row_totals = counts.sum(axis=1)
    total = row_totals.sum()
    weighted = 0.0
    for source in range(q):
        if row_totals[source] == 0:
            continue
        probabilities = counts[source] / row_totals[source]
        positive = probabilities > 0
        distance = np.abs(np.arange(q) - source) / max(q - 1, 1)
        weighted += (row_totals[source] / total) * np.sum(
            distance[positive] * -probabilities[positive] * np.log2(probabilities[positive])
        )
    return float(weighted)


def lz78_entropy(states: np.ndarray) -> tuple[float, int]:
    """Return c*log2(c)/n and the LZ78 phrase count using trie parsing."""
    if not len(states):
        return math.nan, 0
    trie: dict[int, dict] = {}
    phrases = 0
    node = trie
    for symbol in states:
        key = int(symbol)
        child = node.get(key)
        if child is None:
            node[key] = {}
            phrases += 1
            node = trie
        else:
            node = child
    if node is not trie:  # Count an incomplete final phrase.
        phrases += 1
    return float(phrases * math.log2(max(phrases, 1)) / len(states)), phrases


def fano_predictability(entropy: float, q: int) -> float:
    """Numerically invert Fano's equality for maximum predictability."""
    entropy = float(np.clip(entropy, 0.0, math.log2(q)))
    lo, hi = 0.0, (q - 1.0) / q
    for _ in range(80):
        error = (lo + hi) / 2.0
        binary = 0.0 if error in (0.0, 1.0) else -error * math.log2(error) - (1 - error) * math.log2(1 - error)
        if binary + error * math.log2(q - 1) < entropy:
            lo = error
        else:
            hi = error
    return 1.0 - (lo + hi) / 2.0


def metrics(values: np.ndarray, edges: np.ndarray, include_diff: bool = True) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    q = len(edges) - 1
    if not len(values):
        return {"n_samples": 0, "phrase_count": 0, "E_unc_bits": math.nan,
                "E_actual_bits": math.nan, "E_unc_normalized": math.nan,
                "E_actual_normalized": math.nan, "temporal_structure_score": math.nan,
                "fano_predictability": math.nan}
    states = quantize(values, edges)
    unc = shannon_entropy(states, q)
    actual, phrases = lz78_entropy(states)
    weighted = weighted_transition_entropy(states, q)
    # Finite samples can put the phrase estimate above the alphabet maximum.
    actual = min(actual, math.log2(q))
    result = {
        "n_samples": len(states), "phrase_count": phrases, "E_unc_bits": unc,
        "E_actual_bits": actual, "E_unc_normalized": unc / math.log2(q),
        "E_actual_normalized": actual / math.log2(q),
        "weighted_transition_entropy": weighted,
        "weighted_transition_entropy_normalized": weighted / math.log2(q),
        "temporal_structure_score": 0.0 if unc == 0 else max(0.0, 1.0 - actual / unc),
        "fano_predictability": fano_predictability(actual, q),
    }
    if include_diff and len(values) > 1:
        differences = np.diff(values)
        diff_result = metrics(differences, np.asarray(DEFAULT_DIFF_EDGES, dtype=float), include_diff=False)
        for key in ("E_unc_normalized", "E_actual_normalized", "temporal_structure_score", "fano_predictability"):
            result[f"diff_{key}"] = diff_result[key]
    else:
        for key in ("E_unc_normalized", "E_actual_normalized", "temporal_structure_score", "fano_predictability"):
            result[f"diff_{key}"] = math.nan
    return result


def region_for_frequency(frequency: float, limits: np.ndarray) -> tuple[int, float, float]:
    index = int(np.clip(np.searchsorted(limits, frequency, side="right") - 1, 0, len(limits) - 2))
    return index, float(limits[index]), float(limits[index + 1])


def aggregate_rows(base: dict, rows: list[dict]) -> dict:
    result = dict(base)
    result["bin_count"] = len(rows)
    for metric in ("E_unc_bits", "E_actual_bits", "E_unc_normalized", "E_actual_normalized",
                   "weighted_transition_entropy", "weighted_transition_entropy_normalized",
                   "diff_E_unc_normalized", "diff_E_actual_normalized",
                   "temporal_structure_score", "fano_predictability",
                   "diff_temporal_structure_score", "diff_fano_predictability"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[f"{metric}_median"] = float(np.nanmedian(values))
        result[f"{metric}_iqr"] = float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
        result[f"{metric}_min"] = float(np.nanmin(values))
        result[f"{metric}_max"] = float(np.nanmax(values))
    return result


def main() -> None:
    args = parse_args()
    edges = np.asarray(args.edges, dtype=float)
    if len(edges) < 3 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise SystemExit("--edges requires at least three finite, strictly increasing values")
    centered_edges = edges - np.median(edges)
    if args.boundaries.is_dir():
        frames = []
        for csv in sorted(args.boundaries.joinpath("powder").glob("*.csv")):
            frame = pd.read_csv(csv, dtype={"run_id": str, "band": str})
            frame["dataset"] = "powder"
            frame["node"] = csv.stem
            frames.append(frame)
        boundaries = pd.concat(frames, ignore_index=True)
    else:
        boundaries = pd.read_csv(args.boundaries, dtype={"run_id": str, "band": str})
    output: list[dict] = []
    sources: list[str] = []

    for node in args.nodes:
        for run_id in args.runs:
            for band in args.bands:
                path = args.input_root / node / run_id / band / "power_1mhz_avg_per_minute.csv"
                if not path.exists():
                    raise FileNotFoundError(f"missing selected trace: {path}")
                frame = pd.read_csv(path)
                frequency_columns = [column for column in frame.columns if _is_float(column)]
                if not frequency_columns:
                    raise ValueError(f"no numeric frequency columns in {path}")
                frame = frame[frequency_columns].apply(pd.to_numeric, errors="coerce")
                frequencies = np.asarray([float(column) for column in frequency_columns])
                band_start, band_end = (float(value) for value in band.split("_"))
                match = boundaries[(boundaries["node"] == node) & (boundaries["run_id"] == run_id) & (boundaries["band"] == band)]
                if len(match) != 1:
                    if "region_start_mhz" not in boundaries.columns or len(match) == 0:
                        raise ValueError(f"expected boundary rows for {node}/{run_id}/{band}, found {len(match)}")
                if "region_start_mhz" in match.columns:
                    regions = match.sort_values("region_id")
                    limits = np.asarray(
                        [float(regions.iloc[0]["region_start_mhz"]) - 0.5]
                        + [float(value) + 0.5 for value in regions["region_end_mhz"]],
                        dtype=float,
                    )
                else:
                    text = match.iloc[0]["boundaries_mhz"]
                    manual = [] if pd.isna(text) or not str(text).strip() else [float(value) for value in str(text).split(";")]
                    limits = np.unique(np.asarray([band_start, *manual, band_end]))
                sources.append(str(path))

                for mode, mode_edges in (("raw_dbm", edges), ("median_centered_offset_db", centered_edges)):
                    values = frame.to_numpy(dtype=float)
                    if mode != "raw_dbm":
                        values = values - np.nanmedian(values, axis=0)
                    region_bins: dict[int, list[dict]] = {}
                    for column_index, frequency in enumerate(frequencies):
                        region_id, start, end = region_for_frequency(frequency, limits)
                        base = {"record_type": "bin", "node": node, "run_id": run_id, "band": band,
                                "mode": mode, "region_id": region_id, "region_start_mhz": start,
                                "region_end_mhz": end, "frequency_mhz": frequency}
                        row = {**base, **metrics(values[:, column_index], mode_edges)}
                        output.append(row)
                        region_bins.setdefault(region_id, []).append(row)
                    for region_id, rows in region_bins.items():
                        base = {"record_type": "region_bins", "node": node, "run_id": run_id, "band": band,
                                "mode": mode, "region_id": region_id, "region_start_mhz": rows[0]["region_start_mhz"],
                                "region_end_mhz": rows[0]["region_end_mhz"]}
                        output.append(aggregate_rows(base, rows))
                        indexes = [np.where(frequencies == row["frequency_mhz"])[0][0] for row in rows]
                        median_trace = np.nanmedian(values[:, indexes], axis=1)
                        output.append({**base, "record_type": "region_median_trace", "bin_count": len(rows),
                                       **metrics(median_trace, mode_edges)})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(args.output_csv, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script": str(Path(__file__).resolve()),
        "nodes": list(args.nodes), "runs": list(args.runs), "bands": list(args.bands), "sources": sources,
        "boundaries_csv": str(args.boundaries), "raw_edges_dbm": edges.tolist(),
        "centered_edges_db": centered_edges.tolist(), "state_count": len(edges) - 1,
        "tail_handling": "values below/above edge range are clipped to first/last state",
        "lz_estimator": "LZ78 phrase count c; E_actual=c*log2(c)/n bits/symbol, clipped to log2(Q)",
        "temporal_structure_score": "max(0, 1-E_actual/E_unc), or 0 when E_unc=0",
        "fano_equation": "E_actual=h2(1-Pi)+(1-Pi)*log2(Q-1), solved by bisection",
        "csv_record_types": ["bin", "region_bins", "region_median_trace"],
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _is_float(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
