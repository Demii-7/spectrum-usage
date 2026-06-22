#!/usr/bin/env python3
"""Final reconstruction script — tries multiple averaging strategies to match repo CSV."""

import argparse, json, math, sys
from array import array
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
import numpy as np

N_BINS = 250

# Nominal offsets discovered — will fine-tune via optimization
# CC1 offset 21000 = 1347.16 MHz
# CC2 offset 33250 = 2082.16 MHz  
# LW1 offset 27500 = 1737.16 MHz
NOMINAL_OFFSETS = {'CC1': 21000, 'CC2': 33250, 'LW1': 27500}


def floor_to_minute_utc(dt_str):
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0)


def process_archive(label, archive_path, bin_offset, avg_in_dbm=False):
    """Process a SigMF archive and return per-minute averages for a 250-bin band."""
    with ZipFile(archive_path) as zip_file:
        meta_names = sorted(
            name for name in zip_file.namelist()
            if name.endswith(".sigmf-meta")
            and "__MACOSX/" not in name
            and "/._" not in name
        )
        minute_sums = defaultdict(list)
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
            vals = list(raw[bin_offset:bin_offset + N_BINS])

            if avg_in_dbm:
                minute_sums[minute_key].append(vals)
            else:
                lin_vals = [10.0 ** (x / 10.0) for x in vals]
                if minute_key not in minute_sums:
                    minute_sums[minute_key] = lin_vals
                else:
                    for i, v in enumerate(lin_vals):
                        minute_sums[minute_key][i] += v
            minute_counts[minute_key] += 1

    minute_rows = {}
    for mk, data in minute_sums.items():
        cnt = minute_counts[mk]
        if avg_in_dbm:
            # Average in dBm domain
            arr = np.array(data, dtype=np.float32)
            row = list(arr.mean(axis=0))
        else:
            # Average in linear domain, convert back to dBm
            row = [10.0 * math.log10(max(x / cnt, 1e-30)) for x in data]
        minute_rows[mk] = row

    return minute_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("training/data/reconstructed_final.csv"))
    parser.add_argument("--avg-in-dbm", action="store_true", help="Average in dBm instead of linear")
    parser.add_argument("--offsets", type=str, help="JSON: {label: offset}", default=None)
    args = parser.parse_args()

    offsets = json.loads(args.offsets) if args.offsets else NOMINAL_OFFSETS

    specs = []
    for v in args.archive:
        if "=" in v:
            label, path = v.split("=", 1)
            specs.append((label.strip(), Path(path.strip())))
        else:
            p = Path(v.strip())
            specs.append((p.stem, p))

    archives = []
    for label, path in specs:
        off = offsets.get(label)
        if off is None:
            print(f"Warning: no offset for {label}, using nominal")
            off = NOMINAL_OFFSETS.get(label, 0)
        print(f"Processing {label} offset={off} avg_in_dbm={args.avg_in_dbm}...")
        rows = process_archive(label, path, off, avg_in_dbm=args.avg_in_dbm)
        archives.append({"label": label, "minute_rows_db": rows})
        print(f"  -> {len(rows)} minutes")

    common = sorted(set(archives[0]["minute_rows_db"]) &
                    set(archives[1]["minute_rows_db"]) &
                    set(archives[2]["minute_rows_db"]))
    print(f"\nCommon minutes: {len(common)}")

    merged = []
    for mk in common:
        row = []
        for a in archives:
            row.extend(a["minute_rows_db"][mk])
        merged.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")

    print(f"Wrote {len(merged)} rows x {len(merged[0])} cols to {args.output}")


if __name__ == "__main__":
    main()
