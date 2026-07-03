# Utility scripts

This document describes helper scripts that convert external datasets into the
`power_1mhz_avg_per_minute.csv` format used throughout this directory, and
one planned script for merging data across nodes.

---

## `sigmf_zip_to_csv.py` — convert zipped SigMF to CSV

    python3 evaluation/sigmf_zip_to_csv.py <archive.zip> [options]

Converts a zipped SigMF archive (captures from AERPAW, COSMOS, etc.) into a
per-minute, 1 MHz CSV. Each SigMF metadata+data pair is one capture; captures
are floor-binned to the minute in UTC, averaged linearly in power, and written
as one row per minute.

| Argument | Default | Description |
|---|---|---|
| `input` | *required* | Path to the zipped SigMF archive |
| `--output` | auto | Output CSV path |
| `--band-start-mhz` | 2813 | Lower edge of the frequency band |
| `--band-width-mhz` | 250 | Width of the frequency band |
| `--full-band` | disabled | Export the full frequency span in the archive |
| `--no-header` | disabled | Omit the frequency-bin column header row |
| `--progress-interval` | 5000 | Print progress every N captures |

The default output path is `<archive parent>/aerpaw/<archive stem>_power_1mhz_avg_per_minute.csv`.

---

## `papers/data/convert_spectrum_bands_to_csv.py` — convert Electrosense NPY archive to CSV

    python3 papers/data/convert_spectrum_bands_to_csv.py [options]

Extracts `papers/data/spectrum_bands.tar.gz` (a collection of per-band,
per-session 6-hour PSD matrices from the Electrosense crowdsensing platform)
into CSV files with native ~9.3 kHz frequency resolution.

Each `.npy` file in the archive becomes a CSV with:
- `timestamp_utc` as first column (date inferred from the session directory
  name, start assumed 00:00 UTC)
- One column per frequency bin at native resolution (center frequencies as
  headers in MHz)

Output layout mirrors the `node_id / run_id / band` convention:

    evaluation/electrosense/{site}/{run_id}/{band_start}_{band_stop}/power_native_resolution.csv

| Argument | Default | Description |
|---|---|---|
| `--archive` | `papers/data/spectrum_bands.tar.gz` | Path to the input archive |
| `--output-dir` | `evaluation/electrosense` | Output directory root |
| `--year` | 2022 | Assumed year for date directories (e.g. `Aug_1` → `2022-08-01`) |

The script skips files with zero bandwidth or inverted ranges (4 files in the
current archive). Upload to object storage:

    python3 evaluation/sync_object_storage.py \
      evaluation/electrosense \
      s3://spectrum/electrosense \
      --tool fsspec \
      --env-file evaluation/.env \
      --once

---

## `scripts/correct_lw1_discontinuity.py` — correct AERPAW LW1 discontinuity

    python3 evaluation/scripts/correct_lw1_discontinuity.py [options]

Corrects the known discontinuity in the AERPAW LW1 trace. The discontinuity is
a site-wide level shift at row 6840, consistent with a receiver gain or
calibration state change rather than a spectrum-usage event.

The script preserves the original measurements by renaming the canonical input
file:

    power_1mhz_avg_per_minute.csv

to:

    power_1mhz_avg_per_minute_uncorrected.csv

It then writes the corrected data back to the canonical path,
`power_1mhz_avg_per_minute.csv`, so downstream scripts use the adjusted LW1
trace without special-case logic.

The correction adds a per-frequency offset to all rows before the discontinuity.
For each frequency bin, the offset is the 60-row post-discontinuity mean minus
the 60-row pre-discontinuity mean. The script writes a report and the per-bin
offsets under `evaluation/results/statistical_analysis/`.

| Argument | Default | Description |
|---|---|---|
| `--input` | `evaluation/aerpaw/LW1/20220208T1757Z/87_6020/power_1mhz_avg_per_minute.csv` | Canonical LW1 CSV to correct |
| `--discontinuity-row` | `6840` | First row after the level shift |
| `--window-rows` | `60` | Rows before and after the discontinuity used to estimate offsets |
| `--report` | `evaluation/results/statistical_analysis/lw1_discontinuity_correction.csv` | Summary report path |
| `--offsets` | `evaluation/results/statistical_analysis/lw1_discontinuity_offsets.csv` | Per-frequency offsets path |
| `--force` | disabled | Recreate the uncorrected backup from the current canonical CSV |

After applying the correction, upload both the canonical corrected CSV and the
uncorrected backup to object storage so future downloads reproduce the same file
layout.

---

## `merge_spectrum_data.py` — merge across nodes, time, or both (not yet written)

*This script is planned but not yet implemented.*

The goal is to merge multiple `power_1mhz_avg_per_minute.csv` (or
`power_native_resolution.csv`) files into a single spatio-temporal data
product. Proposed interface:

    python3 evaluation/merge_spectrum_data.py <input_dir> --output <path> [options]

| Argument | Description |
|---|---|
| `--output` | Merged output file (CSV, Parquet, or HDF5) |
| `--mode` | `space`, `time`, or `both` |
| `--resample` | Time resolution to align to (e.g. `1min`, `5min`, `1h`) |
| `--fill` | Gap-filling method: `nan`, `interpolate`, `ffill` |

Spatial merging would align frequency bins across nodes that cover the same
band and average (or stack) the overlapping power measurements. Temporal
merging would resample irregular time grids to a common cadence. Combined
mode (`both`) would produce a regular space–time grid.
