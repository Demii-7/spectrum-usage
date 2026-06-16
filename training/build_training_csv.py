#!/usr/bin/env python3
import argparse
import json
import math
import sys
from array import array
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


def parse_archive_arg(value):
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
    else:
        path = value.strip()
        label = Path(path).stem
    if not label:
        raise argparse.ArgumentTypeError(f"Invalid archive spec: {value!r}")
    return label, Path(path)


def list_sigmf_meta_names(zip_file):
    return sorted(
        name
        for name in zip_file.namelist()
        if name.endswith(".sigmf-meta")
        and "__MACOSX/" not in name
        and "/._" not in name
    )


def read_json(zip_file, member_name):
    return json.loads(zip_file.read(member_name).decode("utf-8"))


def read_float32_array(zip_file, member_name):
    values = array("f")
    values.frombytes(zip_file.read(member_name))
    if sys.byteorder != "little":
        values.byteswap()
    return values


def floor_to_minute_utc(dt_str):
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0)


def build_bin_ranges(freq_axis_mhz, band_start_mhz, band_width_mhz):
    band_stop_mhz = band_start_mhz + band_width_mhz
    ranges = []
    idx = 0
    n = len(freq_axis_mhz)

    while idx < n and freq_axis_mhz[idx] < band_start_mhz:
        idx += 1

    for mhz in range(band_start_mhz, band_stop_mhz):
        start_idx = idx
        while idx < n and freq_axis_mhz[idx] < mhz + 1:
            idx += 1
        end_idx = idx
        if start_idx == end_idx:
            raise ValueError(f"No source bins found for 1 MHz bin starting at {mhz} MHz")
        ranges.append((start_idx, end_idx))

    return ranges


def aggregate_band_linear(raw_powers_db, bin_ranges):
    row = []
    for start_idx, end_idx in bin_ranges:
        lin_sum = 0.0
        count = end_idx - start_idx
        for idx in range(start_idx, end_idx):
            lin_sum += 10.0 ** (raw_powers_db[idx] / 10.0)
        row.append(lin_sum / count)
    return row


def process_archive(label, archive_path, band_start_mhz, band_width_mhz):
    with ZipFile(archive_path) as zip_file:
        meta_names = list_sigmf_meta_names(zip_file)
        if not meta_names:
            raise ValueError(f"No SigMF metadata files found in {archive_path}")

        first_meta = read_json(zip_file, meta_names[0])
        freq_axis_mhz = first_meta["global"]["dataset:frequency_axis_MHz"]
        bin_ranges = build_bin_ranges(freq_axis_mhz, band_start_mhz, band_width_mhz)

        minute_sums = {}
        minute_counts = defaultdict(int)

        for meta_name in meta_names:
            meta = read_json(zip_file, meta_name)
            capture = meta["captures"][0]
            minute_key = floor_to_minute_utc(capture["core:datetime"])
            data_name = meta_name.replace(".sigmf-meta", ".sigmf-data")
            raw_powers_db = read_float32_array(zip_file, data_name)
            row_linear = aggregate_band_linear(raw_powers_db, bin_ranges)

            if minute_key not in minute_sums:
                minute_sums[minute_key] = row_linear
            else:
                current = minute_sums[minute_key]
                for i, value in enumerate(row_linear):
                    current[i] += value
            minute_counts[minute_key] += 1

    minute_rows_db = {}
    for minute_key, row_linear_sum in minute_sums.items():
        sweep_count = minute_counts[minute_key]
        minute_rows_db[minute_key] = [
            10.0 * math.log10(max(value / sweep_count, 1e-30))
            for value in row_linear_sum
        ]

    return {
        "label": label,
        "archive_path": str(archive_path),
        "band_start_mhz": band_start_mhz,
        "band_width_mhz": band_width_mhz,
        "minute_rows_db": minute_rows_db,
        "raw_bin_count": len(freq_axis_mhz),
    }


def build_header(labels, band_start_mhz, band_width_mhz):
    names = []
    for label in labels:
        for mhz in range(band_start_mhz, band_start_mhz + band_width_mhz):
            names.append(f"{label}_{mhz}")
    return names


def write_output_csv(output_path, labels, band_start_mhz, band_width_mhz, merged_rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(build_header(labels, band_start_mhz, band_width_mhz)) + "\n")
        for _, row in merged_rows:
            handle.write(",".join(f"{value:.6f}" for value in row) + "\n")


def write_manifest(output_path, archives, merged_rows, labels, band_start_mhz, band_width_mhz):
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "band_start_mhz": band_start_mhz,
        "band_width_mhz": band_width_mhz,
        "labels": labels,
        "row_count": len(merged_rows),
        "archives": [
            {
                "label": archive["label"],
                "archive_path": archive["archive_path"],
                "minute_count": len(archive["minute_rows_db"]),
                "raw_bin_count": archive["raw_bin_count"],
            }
            for archive in archives
        ],
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Build a low-space training CSV directly from SigMF zip archives"
    )
    parser.add_argument(
        "--archive",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="SigMF zip archive. Repeat for multiple nodes, e.g. CC1=... --archive CC2=...",
    )
    parser.add_argument(
        "--band-start-mhz",
        type=int,
        required=True,
        help="Integer start of the selected sub-band in MHz",
    )
    parser.add_argument(
        "--band-width-mhz",
        type=int,
        default=250,
        help="Number of 1 MHz bins to keep per archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/merged_power_data_sub6GHz_avg_per_minute.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    archive_specs = [parse_archive_arg(value) for value in args.archive]
    archives = [
        process_archive(label, path, args.band_start_mhz, args.band_width_mhz)
        for label, path in archive_specs
    ]

    labels = [archive["label"] for archive in archives]
    common_minutes = set(archives[0]["minute_rows_db"])
    for archive in archives[1:]:
        common_minutes &= set(archive["minute_rows_db"])

    merged_rows = []
    for minute_key in sorted(common_minutes):
        merged_row = []
        for archive in archives:
            merged_row.extend(archive["minute_rows_db"][minute_key])
        merged_rows.append((minute_key, merged_row))

    if not merged_rows:
        raise SystemExit("No common minute buckets found across the provided archives")

    write_output_csv(args.output, labels, args.band_start_mhz, args.band_width_mhz, merged_rows)
    write_manifest(args.output, archives, merged_rows, labels, args.band_start_mhz, args.band_width_mhz)

    print(f"Wrote {len(merged_rows)} rows to {args.output}")
    print(f"Columns: {len(labels)} archive(s) x {args.band_width_mhz} bins = {len(labels) * args.band_width_mhz}")
    print("Archive extraction was not required.")


if __name__ == "__main__":
    main()
