#!/usr/bin/env python3
"""Convert a zipped SigMF power dataset to per-minute 1 MHz CSV rows."""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def list_sigmf_meta_names(zip_file):
    return sorted(
        name
        for name in zip_file.namelist()
        if name.endswith(".sigmf-meta")
        and "__MACOSX/" not in name
        and "/._" not in name
    )


def floor_to_minute_utc(dt_str):
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0)


def build_bin_ranges(freq_axis_mhz, band_start_mhz, band_width_mhz):
    bin_edges = np.arange(
        band_start_mhz,
        band_start_mhz + band_width_mhz + 1,
        dtype=float,
    )
    starts = np.searchsorted(freq_axis_mhz, bin_edges[:-1], side="left")
    ends = np.searchsorted(freq_axis_mhz, bin_edges[1:], side="left")

    empty = np.nonzero(starts == ends)[0]
    if empty.size:
        mhz = band_start_mhz + int(empty[0])
        raise ValueError(f"No source bins found for 1 MHz bin starting at {mhz} MHz")
    return starts, ends


def full_band_bounds(freq_axis_mhz):
    band_start_mhz = math.floor(float(freq_axis_mhz[0]))
    band_stop_mhz = math.ceil(float(freq_axis_mhz[-1]))
    return band_start_mhz, band_stop_mhz - band_start_mhz


def read_power_db(zip_file, member_name):
    return np.frombuffer(zip_file.read(member_name), dtype="<f4")


def aggregate_1mhz_linear(power_db, starts, ends):
    selected = power_db[starts[0] : ends[-1]]
    linear = np.power(10.0, selected.astype(np.float64) / 10.0)
    relative_starts = starts - starts[0]
    sums = np.add.reduceat(linear, relative_starts)
    counts = ends - starts
    return sums[: len(starts)] / counts


def load_per_minute_matrix(archive_path, band_start_mhz, band_width_mhz, full_band, progress_interval):
    minute_sums = {}
    minute_counts = defaultdict(int)

    with ZipFile(archive_path) as zip_file:
        meta_names = list_sigmf_meta_names(zip_file)
        if not meta_names:
            raise ValueError(f"No SigMF metadata files found in {archive_path}")

        first_meta = json.loads(zip_file.read(meta_names[0]).decode("utf-8"))
        freq_axis_mhz = np.asarray(
            first_meta["global"]["dataset:frequency_axis_MHz"], dtype=np.float64
        )
        if full_band:
            band_start_mhz, band_width_mhz = full_band_bounds(freq_axis_mhz)
        starts, ends = build_bin_ranges(freq_axis_mhz, band_start_mhz, band_width_mhz)

        for index, meta_name in enumerate(meta_names, start=1):
            meta = json.loads(zip_file.read(meta_name).decode("utf-8"))
            capture = meta["captures"][0]
            minute_key = floor_to_minute_utc(capture["core:datetime"])
            data_name = meta_name.replace(".sigmf-meta", ".sigmf-data")
            row_linear = aggregate_1mhz_linear(read_power_db(zip_file, data_name), starts, ends)

            if minute_key not in minute_sums:
                minute_sums[minute_key] = row_linear.copy()
            else:
                minute_sums[minute_key] += row_linear
            minute_counts[minute_key] += 1

            if progress_interval and index % progress_interval == 0:
                print(f"Processed {index}/{len(meta_names)} captures")

    minutes = sorted(minute_sums)
    data = np.empty((len(minutes), band_width_mhz), dtype=np.float32)
    for row_index, minute_key in enumerate(minutes):
        avg_linear = minute_sums[minute_key] / minute_counts[minute_key]
        data[row_index] = 10.0 * np.log10(np.maximum(avg_linear, 1e-30))

    return data, minutes, band_start_mhz


def default_output_path(input_path):
    return input_path.parent / "aerpaw" / f"{input_path.stem}_power_1mhz_avg_per_minute.csv"


def write_csv(data, output_path, band_start_mhz, include_header):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as csv_file:
        if include_header:
            bin_centers = band_start_mhz + np.arange(data.shape[1], dtype=float) + 0.5
            csv_file.write(",".join(f"{center:.1f}" for center in bin_centers) + "\n")

        for row in data:
            csv_file.write(",".join(f"{value:.4f}" for value in row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Zipped SigMF archive")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    parser.add_argument("--band-start-mhz", type=int, default=2813)
    parser.add_argument("--band-width-mhz", type=int, default=250)
    parser.add_argument(
        "--full-band",
        action="store_true",
        help="Export the full frequency span in the archive at 1 MHz resolution.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Write data rows only, without frequency-bin column headers.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5000,
        help="Print progress every N captures. Use 0 to disable.",
    )
    args = parser.parse_args()

    data, minutes, band_start_mhz = load_per_minute_matrix(
        args.input,
        args.band_start_mhz,
        args.band_width_mhz,
        args.full_band,
        args.progress_interval,
    )

    output = args.output or default_output_path(args.input)
    write_csv(data, output, band_start_mhz, not args.no_header)

    print(f"Wrote {output}")
    print(f"Rows: {data.shape[0]} minutes from {len(minutes)} minute buckets")
    print(f"Columns: {data.shape[1]} frequency bins")
    print(f"Frequency span: {band_start_mhz}-{band_start_mhz + data.shape[1]} MHz")
    if minutes:
        print(f"Time span: {minutes[0].isoformat()} to {minutes[-1].isoformat()}")


if __name__ == "__main__":
    main()
