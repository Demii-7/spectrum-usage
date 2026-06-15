## Training data

This directory includes code necessary to retrieve the AERPAW dataset [February 2022: CC1, CC2, LW1 Spectrum Measurements](https://aerpaw.org/dataset/february-2022-cc1-cc2-lw1-spectrum-measurements/) and format it so that it may be used to train a machine learning model on the spectrum usage prediction task.

`build_training_csv.py` can stream SigMF zip archives directly and write a per-minute CSV into `training/data/` without fully extracting the archive contents to disk.

### Prerequisites

- Python 3.8+
- The AERPAW SigMF zip archives for the nodes you want to merge, for example:
  - `ResultsCC1Feb2022_SigMF.zip`
  - `ResultsCC2Feb2022_SigMF.zip`
  - `ResultsLW1Feb2022_SigMF.zip`
- Enough free disk space for the output CSV only. Full archive extraction is not required.

The script uses only Python standard library modules.

### What the script does

- Reads each SigMF zip archive in place
- Loads each spectrum sweep from `.sigmf-data` and `.sigmf-meta`
- Buckets sweeps into UTC minute intervals
- Averages power per minute in linear scale, then converts back to dB
- Keeps a selected `250 MHz` window as `250` `1 MHz` bins per node
- Concatenates node outputs horizontally into one merged training CSV

If you provide `CC1`, `CC2`, and `LW1`, the output will contain `750` columns total.

### How to run

From the repository root:

```bash
python3 "training/build_training_csv.py" \
  --archive CC1="/path/to/ResultsCC1Feb2022_SigMF.zip" \
  --archive CC2="/path/to/ResultsCC2Feb2022_SigMF.zip" \
  --archive LW1="/path/to/ResultsLW1Feb2022_SigMF.zip" \
  --band-start-mhz 2813 \
  --band-width-mhz 250 \
  --output "training/data/merged_power_data_sub6GHz_avg_per_minute.csv"
```

LW1-only example:

```bash
python3 "training/build_training_csv.py" \
  --archive LW1="/path/to/ResultsLW1Feb2022_SigMF.zip" \
  --band-start-mhz 2813 \
  --band-width-mhz 250 \
  --output "training/data/lw1_power_avg_per_minute.csv"
```

### Arguments

- `--archive LABEL=/path/to/archive.zip`
  - Repeat once per node
  - `LABEL` becomes part of the output header
- `--band-start-mhz`
  - Integer start frequency in MHz for the selected sub-band
- `--band-width-mhz`
  - Number of `1 MHz` bins to keep per node
  - Use `250` to match the expected per-node width for a `750`-column merged CSV
- `--output`
  - Output CSV path

### Outputs

- CSV file at the path passed to `--output`
- JSON manifest at `<output>.json` describing:
  - source archives
  - selected band
  - labels
  - row count

### Notes

- The script only keeps minute buckets that exist in all provided archives.
- The example `2813` MHz start is a working placeholder for reconstruction experiments, not a guaranteed exact match to the original external preprocessing used by other projects.
