#!/usr/bin/env python3
"""Plot raw and annotated spectrograms for selected spectrum traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EVALUATION_ROOT.parent
from spectrogram_colormap import ensure_minimum_db_span


DEFAULT_ANNOTATION_ROOT = REPOSITORY_ROOT / "data" / "annotations"
DEFAULT_OUTPUT = EVALUATION_ROOT / "results" / "figures" / "spectrogram_annotations.png"
POWER_FILE = "power_1mhz_avg_per_minute.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", help="Site/dataset to include, such as powder or cosmos.")
    parser.add_argument("--node", action="append", default=[], help="Node to include. Repeatable.")
    parser.add_argument("--run", action="append", default=[], help="Run ID to include. Repeatable.")
    parser.add_argument("--band", action="append", default=[], help="Band to include. Repeatable.")
    parser.add_argument("--input-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--annotation-node", default=None, help="Node whose boundaries apply to every selected trace.")
    parser.add_argument("--annotation-run", default=None, help="Run ID for the shared annotation boundaries.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-time-bins", type=int, default=1500)
    parser.add_argument("--max-frequency-bins", type=int, default=1200)
    parser.add_argument("--boundary-linewidth", type=float, default=0.7)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def load_annotations(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/*.csv")):
        frame = pd.read_csv(path, dtype={"run_id": str, "band": str})
        frame["site"] = path.parent.name
        frame["node"] = path.stem
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no annotation CSVs under {root}")
    return pd.concat(frames, ignore_index=True)


def select_traces(annotations: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = annotations
    filters = (
        ("site", [args.site] if args.site else []),
        ("node", args.node),
        ("run_id", args.run),
        ("band", args.band),
    )
    for column, values in filters:
        if values:
            selected = selected[selected[column].astype(str).isin(values)]
    columns = ["site", "node", "run_id", "band"]
    return selected[columns].drop_duplicates().sort_values(columns).reset_index(drop=True)


def discover_traces(args: argparse.Namespace) -> pd.DataFrame:
    if not args.site:
        raise ValueError("--site is required with --annotation-node")
    rows = []
    root = args.input_root / args.site
    for path in sorted(root.glob(f"*/*/*/{POWER_FILE}")):
        node, run_id, band = path.relative_to(root).parts[:3]
        if args.node and node not in args.node:
            continue
        if args.run and run_id not in args.run:
            continue
        if args.band and band not in args.band:
            continue
        rows.append({"site": args.site, "node": node, "run_id": run_id, "band": band})
    return pd.DataFrame(rows, columns=["site", "node", "run_id", "band"])


def frequency_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if column == "timestamp_utc":
            continue
        try:
            float(column)
        except (TypeError, ValueError):
            continue
        columns.append(column)
    return sorted(columns, key=float)


def block_edges(length: int, maximum: int) -> np.ndarray:
    if maximum < 1:
        raise ValueError("maximum plotted bins must be positive")
    return np.unique(np.linspace(0, length, min(length, maximum) + 1, dtype=int))


def block_nanmean(values: np.ndarray, row_edges: np.ndarray, column_edges: np.ndarray) -> np.ndarray:
    reduced = np.empty((len(row_edges) - 1, len(column_edges) - 1), dtype=np.float32)
    for row_index, (row_start, row_end) in enumerate(zip(row_edges[:-1], row_edges[1:])):
        for column_index, (column_start, column_end) in enumerate(zip(column_edges[:-1], column_edges[1:])):
            block = values[row_start:row_end, column_start:column_end]
            reduced[row_index, column_index] = np.nan if np.isnan(block).all() else np.nanmean(block)
    return reduced


def load_trace(path: Path, max_time_bins: int, max_frequency_bins: int) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    columns = frequency_columns(frame)
    if not columns:
        raise ValueError(f"no frequency columns in {path}")

    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").dt.floor("min")
        frame = frame.dropna(subset=["timestamp_utc"])
        if frame.empty:
            raise ValueError(f"no valid timestamps in {path}")
        frame = frame.groupby("timestamp_utc", sort=True)[columns].mean()
        frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="min", tz="UTC"))
    else:
        start = pd.to_datetime(path.parents[1].name, format="%Y%m%dT%H%MZ", utc=True, errors="coerce")
        if pd.isna(start):
            raise ValueError(f"no timestamps available for {path}")
        frame = frame[columns]
        frame.index = pd.date_range(start, periods=len(frame), freq="min", tz="UTC")
    frame = frame.ffill().bfill()

    frequencies = np.asarray([float(column) for column in columns], dtype=float)
    values = frame.to_numpy(dtype=np.float32)
    row_edges = block_edges(values.shape[0], max_time_bins)
    column_edges = block_edges(values.shape[1], max_frequency_bins)
    times = frame.index[np.minimum((row_edges[:-1] + row_edges[1:]) // 2, len(frame.index) - 1)]
    reduced_frequencies = np.asarray(
        [np.mean(frequencies[start:end]) for start, end in zip(column_edges[:-1], column_edges[1:])]
    )
    return times, reduced_frequencies, block_nanmean(values, row_edges, column_edges)


def boundaries_for(
    annotations: pd.DataFrame,
    trace: pd.Series,
    annotation_node: str | None = None,
    annotation_run: str | None = None,
) -> list[float]:
    mask = np.ones(len(annotations), dtype=bool)
    values = {
        "site": trace["site"],
        "node": annotation_node or trace["node"],
        "run_id": annotation_run or trace["run_id"],
        "band": trace["band"],
    }
    for column, value in values.items():
        mask &= annotations[column].astype(str).to_numpy() == str(value)
    regions = annotations.loc[mask].sort_values("region_id")
    if regions.empty:
        raise ValueError(
            f"no annotations for {trace['site']}/{trace['node']}/{trace['run_id']}/{trace['band']}"
        )
    return (regions["region_end_mhz"].iloc[:-1].astype(float) + 0.5).tolist()


def main() -> int:
    args = parse_args()
    annotations = load_annotations(args.annotation_root)
    selected = discover_traces(args) if args.annotation_node else select_traces(annotations, args)
    if selected.empty:
        raise SystemExit("ERROR: no annotations match the requested filters")

    traces = []
    finite_values = []
    for _, trace in selected.iterrows():
        path = args.input_root / trace["site"] / trace["node"] / trace["run_id"] / trace["band"] / POWER_FILE
        if not path.is_file():
            raise FileNotFoundError(f"missing power trace: {path}")
        times, frequencies, values = load_trace(path, args.max_time_bins, args.max_frequency_bins)
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_values.append(finite)
        traces.append((
            trace,
            times,
            frequencies,
            values,
            boundaries_for(annotations, trace, args.annotation_node, args.annotation_run),
        ))
    if not finite_values:
        raise ValueError("selected traces contain no finite power values")

    vmin, vmax = ensure_minimum_db_span(*np.percentile(np.concatenate(finite_values), [1, 99]))
    figure, axes = plt.subplots(
        len(traces),
        2,
        figsize=(12, max(3.2, 2.8 * len(traces))),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_index, (trace, times, frequencies, values, boundaries) in enumerate(traces):
        frequency_step = float(np.median(np.diff(frequencies))) if len(frequencies) > 1 else 1.0
        extent = [
            mdates.date2num(times[0].to_pydatetime()),
            mdates.date2num(times[-1].to_pydatetime()),
            frequencies[0] - frequency_step / 2,
            frequencies[-1] + frequency_step / 2,
        ]
        for column_index, annotated in enumerate((False, True)):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                values.T,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if annotated:
                for boundary in boundaries:
                    axis.axhline(boundary, color="black", linewidth=args.boundary_linewidth + 0.8, alpha=0.75)
                    axis.axhline(boundary, color="white", linewidth=args.boundary_linewidth, alpha=0.95)
            locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=times.tz))
            if row_index == len(traces) - 1:
                axis.set_xlabel("UTC time")
            if column_index == 0:
                axis.set_ylabel(
                    f"{trace['site']} | {trace['node']}\n{trace['run_id']} | {trace['band']}\nFrequency (MHz)"
                )
            if row_index == 0:
                axis.set_title("Annotated" if annotated else "Raw")

    if args.title:
        figure.suptitle(args.title)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.96, pad=0.015)
    colorbar.set_label("Power (dBm)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"wrote {len(traces)} trace rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
