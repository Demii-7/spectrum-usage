#!/usr/bin/env python3
"""Reconstruct TSS-LCD repo CSV using per-node frequency offsets discovered via reverse engineering.

Hypothesis: The repo CSV was produced by selecting DIFFERENT 250-bin frequency ranges
for each node from the raw SigMF data, matching the per-node statistical profile.

Discovered optimal raw bin offsets (CC1: 21000, CC2: 33250, LW1: 27500)
corresponding to ~1347, 2082, and 1737 MHz respectively.
"""

import argparse
import json
import math
import sys
from array import array
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


# Per-node bin offsets (discovered via statistical matching)
# These are raw bin indices into the 98,868-bin SigMF frequency axis
NODE_OFFSETS = {
    'CC1': 21000,   # ~1347-1362 MHz
    'CC2': 33250,   # ~2082-2097 MHz
    'LW1': 27500,   # ~1737-1752 MHz
}
N_BINS = 250


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


def floor_to_minute_utc(dt_str):
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0)


def process_archive(label, archive_path, bin_offset, n_bins):
    with ZipFile(archive_path) as zip_file:
        meta_names = sorted(
            name for name in zip_file.namelist()
            if name.endswith(".sigmf-meta")
            and "__MACOSX/" not in name
            and "/._" not in name
        )
        if not meta_names:
            raise ValueError(f"No SigMF metadata files found in {archive_path}")

        minute_sums = {}
        minute_counts = defaultdict(int)

        for meta_name in meta_names:
            meta = json.loads(zip_file.read(meta_name).decode("utf-8"))
            capture = meta["captures"][0]
            minute_key = floor_to_minute_utc(capture["core:datetime"])
            data_name = meta_name.replace(".sigmf-meta", ".sigmf-data")
            raw = array("f")
            raw.frombytes(zip_file.read(data_name))
            if sys.byteorder != "little":
                raw.byteswap()
            selected = raw[bin_offset:bin_offset + n_bins]

            # Accumulate in linear domain
            if minute_key not in minute_sums:
                minute_sums[minute_key] = [10.0 ** (x / 10.0) for x in selected]
            else:
                cur = minute_sums[minute_key]
                for i, v in enumerate(selected):
                    cur[i] += 10.0 ** (v / 10.0)
            minute_counts[minute_key] += 1

    # Convert linear averages back to dBm
    minute_rows_db = {}
    for minute_key, lin_sum in minute_sums.items():
        count = minute_counts[minute_key]
        minute_rows_db[minute_key] = [
            10.0 * math.log10(max(x / count, 1e-30))
            for x in lin_sum
        ]

    return minute_rows_db


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct TSS-LCD repo CSV using per-node frequency offsets"
    )
    parser.add_argument(
        "--archive",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="SigMF zip archive, e.g. CC1=path.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/reconstructed_repo_csv.csv"),
    )
    args = parser.parse_args()

    # Parse archives
    archive_specs = [parse_archive_arg(v) for v in args.archive]

    # Process each archive with its per-node offset
    archives = []
    for label, path in archive_specs:
        offset = NODE_OFFSETS.get(label)
        if offset is None:
            raise ValueError(f"Unknown node label {label!r}; known: {list(NODE_OFFSETS.keys())}")
        print(f"Processing {label} from {path} (bin offset {offset}, {N_BINS} bins)...")
        minute_rows = process_archive(label, path, offset, N_BINS)
        archives.append({"label": label, "minute_rows_db": minute_rows})
        print(f"  -> {len(minute_rows)} minutes")

    # Intersect common minutes
    common = set(archives[0]["minute_rows_db"])
    for a in archives[1:]:
        common &= set(a["minute_rows_db"])
    common = sorted(common)
    print(f"\nCommon minutes across all nodes: {len(common)}")

    # Build merged rows
    merged_rows = []
    for minute_key in common:
        row = []
        for a in archives:
            row.extend(a["minute_rows_db"][minute_key])
        merged_rows.append(row)

    # Write output (NO header, matching repo format)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")

    n_cols = len(merged_rows[0]) if merged_rows else 0
    print(f"Wrote {len(merged_rows)} rows x {n_cols} columns to {args.output}")
    print("(No header row, matching repo CSV format)")


if __name__ == "__main__":
    main()
