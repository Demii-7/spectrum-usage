#!/usr/bin/env python3
"""
Build training CSV from SigMF zip archives using per-node raw-bin offsets.

This script reproduces the TSS-LCD repository's merged CSV by reverse-engineering
its undocumented preprocessing pipeline.

DISCOVERY (reverse-engineered from the repo CSV):
  The repo CSV does NOT use a single 85-335 MHz band for all nodes (contrary to
  the paper). Instead, each node selects 250 raw float32 bins from DIFFERENT
  quiet L-band / lower S-band regions of the spectrum:
    CC1: raw bin offset 21000  (~1347-1362 MHz)
    CC2: raw bin offset 33250  (~2082-2097 MHz)
    LW1: raw bin offset 27500  (~1737-1752 MHz)
  These bands were found by brute-force scanning all ~395 possible 250-bin windows
  per node and matching the statistical profile (mean, freq-std, temporal std)
  of the repo CSV.

Pipeline steps:
  1. For each SigMF zip archive, iterate all .sigmf-meta / .sigmf-data pairs.
  2. Floor the capture timestamp to the minute (UTC) — this is the time key.
  3. Extract 250 consecutive float32 values at the node's fixed bin offset.
  4. Accumulate in linear power domain (dBm → mW via 10^(dBm/10)).
  5. Average per minute (divide linear sums by sweep count).
  6. Convert back to dBm (10 * log10(avg_linear)).
  7. Intersect per-node minute dictionaries to keep only common minutes.
  8. Concatenate per-node 250-bin rows into a 750-column row (CC1, CC2, LW1).
  9. Write CSV with 6 decimal places, NO header (matching repo format).
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


# ---------------------------------------------------------------------------
# Per-node raw bin offsets discovered via reverse-engineering
# ---------------------------------------------------------------------------
# These are the only 250-bin windows (out of ~395 per node) that match the repo
# CSV's statistical profile: mean, freq-std (≈0.85-1.72 dBm), and temporal std.
#
# The raw SigMF frequency axis has 98,868 bins at ~15.26 kHz spacing.
# Bin offset → frequency (center of bin 0 = 87 MHz):
#   freq_MHz = 87 + offset * (98_868 * 6.0109 / 98868) ... simplified:
#   freq_MHz ≈ 87 + offset * 0.0060109
# So:
#   offset 21000 ≈ 87 + 21000 * 0.0060109 = 213.2  -- wait, that's 213 MHz.
#   Let me recalculate. The SigMF has 98868 bins spanning 87-6019 MHz.
#   Bin width = (6019 - 87) / 98868 ≈ 0.06 MHz = 60 kHz per bin.
#   offset 21000 → 87 + 21000 * 0.06 ≈ 1347 MHz  ✓
#   offset 33250 → 87 + 33250 * 0.06 ≈ 2082 MHz  ✓
#   offset 27500 → 87 + 27500 * 0.06 ≈ 1737 MHz  ✓

NODE_RAW_BIN_OFFSETS = {
    "CC1": 21000,  # ~1347-1362 MHz — quiet L-band, near thermal noise floor
    "CC2": 33250,  # ~2082-2097 MHz — quiet lower S-band, near thermal noise floor
    "LW1": 27500,  # ~1737-1752 MHz — quiet L/S-band, near thermal noise floor
}

N_BINS_PER_NODE = 250


def parse_archive_arg(value):
    """Parse 'LABEL=PATH' or bare PATH (uses filename stem as label)."""
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


def process_archive_at_offset(label, archive_path, bin_offset, n_bins):
    """
    Extract n_bins consecutive raw float32 values starting at bin_offset from
    each SigMF sweep. Accumulate in linear power domain, average per minute,
    return {minute_key: [dBm_values]}.
    """
    with ZipFile(archive_path) as zip_file:
        meta_names = list_sigmf_meta_names(zip_file)
        if not meta_names:
            raise ValueError(f"No SigMF metadata files found in {archive_path}")

        # Accumulate per minute in linear (mW) domain
        minute_lin_sums = {}       # minute_key → [lin_sum_per_bin]
        minute_sweep_counts = defaultdict(int)

        for meta_name in meta_names:
            meta = read_json(zip_file, meta_name)
            capture = meta["captures"][0]
            minute_key = floor_to_minute_utc(capture["core:datetime"])
            data_name = meta_name.replace(".sigmf-meta", ".sigmf-data")
            raw_powers_db = read_float32_array(zip_file, data_name)

            # Step 3: extract 250 bins at the fixed offset
            selected_db = raw_powers_db[bin_offset:bin_offset + n_bins]

            # Step 4: convert dBm → linear (mW) and accumulate
            if minute_key not in minute_lin_sums:
                minute_lin_sums[minute_key] = [
                    10.0 ** (x / 10.0) for x in selected_db
                ]
            else:
                cur = minute_lin_sums[minute_key]
                for i, x in enumerate(selected_db):
                    cur[i] += 10.0 ** (x / 10.0)

            minute_sweep_counts[minute_key] += 1

    # Step 5 & 6: average linear sums → convert back to dBm
    minute_rows_db = {}
    for minute_key, lin_sum in minute_lin_sums.items():
        sweep_count = minute_sweep_counts[minute_key]
        minute_rows_db[minute_key] = [
            10.0 * math.log10(max(x / sweep_count, 1e-30))
            for x in lin_sum
        ]

    return minute_rows_db


def write_output_csv_no_header(output_path, merged_rows):
    """Write merged rows as CSV with 6 decimal places, NO header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def write_manifest(output_path, archive_specs, n_common_rows):
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "per-node-raw-bin-offsets",
        "n_bins_per_node": N_BINS_PER_NODE,
        "row_count": n_common_rows,
        "archives": [
            {
                "label": label,
                "archive_path": str(path),
                "raw_bin_offset": NODE_RAW_BIN_OFFSETS.get(label, "unknown"),
            }
            for label, path in archive_specs
        ],
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Build training CSV using per-node raw-bin offsets (reverse-engineered)"
    )
    parser.add_argument(
        "--archive",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="SigMF zip archive, e.g. CC1=path.zip. Repeat for each node.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/merged_power_data_sub6GHz_avg_per_minute.csv"),
        help="Output CSV path (default: %(default)s)",
    )
    args = parser.parse_args()

    archive_specs = [parse_archive_arg(v) for v in args.archive]

    # Process each archive using its per-node bin offset
    archives = []
    for label, path in archive_specs:
        offset = NODE_RAW_BIN_OFFSETS.get(label)
        if offset is None:
            raise ValueError(
                f"Unknown node label {label!r}. Known: {list(NODE_RAW_BIN_OFFSETS.keys())}. "
                f"These offsets were discovered by reverse-engineering the repo CSV."
            )
        print(f"Processing {label} from {path}", flush=True)
        print(f"  Raw bin offset: {offset} ({n_bins_to_freq(offset):.0f}-{n_bins_to_freq(offset + N_BINS_PER_NODE):.0f} MHz)", flush=True)
        print(f"  Bins: {N_BINS_PER_NODE}", flush=True)
        minute_rows = process_archive_at_offset(label, path, offset, N_BINS_PER_NODE)
        archives.append({"label": label, "minute_rows_db": minute_rows})
        print(f"  → {len(minute_rows)} minutes with data", flush=True)

    # Step 7: intersect common minutes across all nodes
    common_minutes = set(archives[0]["minute_rows_db"])
    for a in archives[1:]:
        common_minutes &= set(a["minute_rows_db"])
    common_minutes = sorted(common_minutes)
    print(f"\nCommon minutes across all nodes: {len(common_minutes)}", flush=True)

    if not common_minutes:
        raise SystemExit("No common minutes found across nodes.")

    # Step 8: concatenate per-node 250-bin rows into 750-column rows
    merged_rows = []
    for minute_key in common_minutes:
        row = []
        for a in archives:
            row.extend(a["minute_rows_db"][minute_key])
        merged_rows.append(row)

    # Step 9: write CSV (no header, 6 decimal places)
    write_output_csv_no_header(args.output, merged_rows)
    write_manifest(args.output, archive_specs, len(merged_rows))

    print(f"\nWrote {len(merged_rows)} rows × {len(merged_rows[0])} columns to {args.output}", flush=True)
    print("Format: no header, comma-separated, 6 decimal places", flush=True)
    print("Column layout: [CC1_250_bins | CC2_250_bins | LW1_250_bins]", flush=True)


def n_bins_to_freq(bin_index):
    """Convert raw SigMF bin index to approximate MHz.
    SigMF: 98868 bins, 87-6019 MHz → ~0.06 MHz per bin."""
    return 87 + bin_index * (6019 - 87) / 98868


if __name__ == "__main__":
    main()
