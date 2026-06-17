# Training Data Pipeline

This directory contains the pipeline for acquiring the AERPAW sub-6 GHz spectrum
monitoring dataset and preprocessing it into a format suitable for training
machine learning models on the spectrum usage prediction task.

The raw dataset consists of spectrum sweeps collected by three fixed sensor nodes
(CC1, CC2, LW1) during February 2022 on the AERPAW testbed. Each sweep records
power spectral density (PSD) across 87–6019 MHz with ~60 kHz resolution (98,868
frequency bins). The preprocessing pipeline in this directory:

1. Reads each node's SigMF zip archive in place (no full extraction required)
2. Buckets individual sweeps into one-minute UTC intervals
3. Averages power within each minute in the linear domain, then converts back to dBm
4. Extracts a 250 MHz sub-band (2813–3062 MHz, 250 bins per node)
5. Merges nodes horizontally into a single CSV (750 columns)

The output is a per-minute averaged power matrix with 6839 time steps (rows) and
750 frequency bins (columns), ready for time-series forecasting models.

## Dataset Source

- **Dataset name:** AERPAW sub-6 GHz spectrum monitoring dataset: Fixed nodes
  CC1, CC2, LW1 (February 2022)
- **DOI:** [10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn)
- **Dryad URL:** https://doi.org/10.5061/dryad.hmgqnk9zn
- **AERPAW page:** https://aerpaw.org/dataset/february-2022-cc1-cc2-lw1-spectrum-measurements/

The dataset contains three archives:

| Archive | Node | Sweeps | Size  | Contents             |
|---------|------|--------|-------|----------------------|
| `ResultsCC1Feb2022_SigMF.zip` | CC1 | 32,529 | ~19 GB | 65,058 SigMF entries |
| `ResultsCC2Feb2022_SigMF.zip` | CC2 | 34,865 | ~21 GB | 69,730 SigMF entries |
| `ResultsLW1Feb2022_SigMF.zip` | LW1 | 21,617 | ~12 GB | 43,234 SigMF entries |

Each SigMF sweep pair consists of a `.sigmf-meta` JSON file (metadata: timestamp,
frequency axis, node info) and a `.sigmf-data` binary file (98,868 float32 PSD
values in dBm).

## Dataset Download Instructions

### Prerequisites

- Python 3.8+
- The `requests` package
- ~52 GB of free disk space for the three zip archives
- ~60 MB of additional free disk space for the output CSV and manifest

### Installation

```bash
pip install requests
```

### Download

From the repository root:

```bash
python3 training/data/download_dryad.py
```

This downloads the following three files into the current directory:

| File | Size |
|------|------|
| `ResultsCC1Feb2022_SigMF.zip` | ~19.08 GB |
| `ResultsCC2Feb2022_SigMF.zip` | ~20.66 GB |
| `ResultsLW1Feb2022_SigMF.zip` | ~12.50 GB |

To download into a different directory, use `--dir`:

```bash
python3 training/data/download_dryad.py --dir /path/to/output
```

### How it works

Dryad is protected by the **Anubis** anti-bot WAF (Web Application Firewall)
that issues a SHA256 proof-of-work challenge before serving file downloads.
Common tools like `datahugger`, `wget`, or `curl` cannot bypass this challenge.

The bundled download script (`training/data/download_dryad.py`) handles it in
three steps:

1. Sends a GET request to the Dryad file stream URL
2. If the Anubis challenge is present, parses the `randomData` and `difficulty`
   from the HTML, then brute-forces a SHA256 nonce using all available CPU cores
   (via `multiprocessing.Pool`)
3. Submits the solution to receive temporary Anubis cookies and streams the
   ZIP file through the same authenticated session

The script retries automatically on rate limiting (HTTP 403) with a 60-second
wait between attempts.

### Expected directory structure after download

```
spectrum-usage/
├── ResultsCC1Feb2022_SigMF.zip   (19 GB)
├── ResultsCC2Feb2022_SigMF.zip   (21 GB)
├── ResultsLW1Feb2022_SigMF.zip   (12 GB)
├── training/
│   ├── build_training_csv.py
│   ├── README.md
│   └── data/
│       └── download_dryad.py
└── ...
```

## Dataset Processing Workflow

### Input files

- `ResultsCC1Feb2022_SigMF.zip`
- `ResultsCC2Feb2022_SigMF.zip`
- `ResultsLW1Feb2022_SigMF.zip`

### Processing stages

1. **Archive enumeration** — The script opens each zip archive and lists all
   `.sigmf-meta` entries (filtering out macOS metadata like `__MACOSX/` or `._`
   files).

2. **Per-sweep extraction** — For each SigMF sweep:
   - The `.sigmf-meta` JSON is parsed to extract the sweep's UTC datetime
   - The datetime is floored to the nearest minute (second=0, microsecond=0)
   - The `.sigmf-data` binary is read as 98,868 float32 values (dBm)
   - **250 consecutive float32 values are sliced** starting at the node's
     fixed raw bin offset (CC1=21000, CC2=33250, LW1=27500), selecting a
     quiet noise-floor region of the spectrum (~1.3–2.1 GHz)

3. **Linear-domain accumulation** — The 250 dBm values are converted to linear
   power (mW) via `10^(dBm/10)` and accumulated per minute bucket.

4. **Per-minute averaging** — Each minute bucket's accumulated linear sum is
   divided by the sweep count for that minute and converted back to dBm:
   `10 * log10(mean_linear)`.

5. **Cross-node merge** — Only minute buckets that exist in ALL provided archives
   are kept (set intersection). This ensures aligned time series across nodes.

6. **CSV writing** — The merged rows are written in chronological order (sorted
   by UTC minute key). Each row is the concatenation of:
   `[CC1_250_bins, CC2_250_bins, LW1_250_bins]`
   Written with 6 decimal places, **no header** (matching the repo CSV format).

7. **Manifest writing** — A JSON manifest is written alongside the CSV recording
   the creation time, method, node labels, row count, and per-archive bin offsets.

### Intermediate outputs

- None (all processing is done in memory; no intermediate files are written)

### Final outputs

- `<output>.csv` — The merged per-minute averaged power CSV
- `<output>.csv.json` — JSON manifest with processing metadata

### Expected runtime

- Processing all three archives (52 GB total, ~89K sweeps): approximately
  8–12 minutes on a modern machine with SSD storage
- Most of the time is spent reading and decompressing the zip archives

### Expected memory requirements

- ~500 MB RAM for processing all three archives
- The script streams each sweep's data file individually from the zip and does
  not hold all sweeps in memory simultaneously

## Data Format

### Raw SigMF format

Each SigMF pair consists of:

- **`<name>.sigmf-meta`** — JSON file with the following structure:
  ```json
  {
    "global": {
      "core:datatype": "rf32_le",
      "core:version": "1.0.0",
      "dataset:frequency_axis_MHz": [87.0, 87.06, ..., 6019.0],
      ...
    },
    "captures": [
      {
        "core:datetime": "2022-02-08T12:50:34-05:00",
        "core:frequency": 100000000,
        ...
      }
    ]
  }
  ```
  - `dataset:frequency_axis_MHz` is an array of 98,868 center frequencies
    spanning 87–6019 MHz with ~60 kHz spacing
  - `core:datetime` is the ISO 8601 timestamp with timezone offset

- **`<name>.sigmf-data`** — Raw binary file containing 98,868 IEEE 754
  single-precision (float32) little-endian power values in dBm

### Processed CSV format

The output CSV has the following structure:

- **No header row** — The CSV is pure data (matching the TSS-LCD repo format).
- **Data rows** — One row per common minute bucket, in chronological order.
  Each value is averaged power in dBm, formatted to 6 decimal places:
  ```
  -133.475915,-133.518531,...,-131.902385
  ```

- **Frequency range:** Per-node noise-floor bands (CC1 ~1347–1362 MHz,
  CC2 ~2082–2097 MHz, LW1 ~1737–1752 MHz)
- **Number of bins:** 250 per node
- **Node layout:** CC1 (columns 0–249), CC2 (columns 250–499), LW1 (columns 500–749)

### Data shape

| Property | Value |
|----------|-------|
| Number of rows (time steps) | 6,839 |
| Number of columns (features) | 750 |
| Rows per node individually | CC1: 10,243, CC2: 9,519, LW1: 10,080 |
| Common minutes (intersection) | 6,839 |
| Time span | ~4.75 days of continuous coverage |
| Power range | −135.36 to −116.00 dBm |
| Power mean | −131.90 dBm |
| Power standard deviation | 3.07 dBm |
| Missing values | None (0 NaN, 0 Inf) |

Each row represents the average power spectral density across all three nodes
during a single UTC minute. Rows are contiguous in time (sorted by UTC minute
key) but exact timestamps are not stored in the CSV — the row index corresponds
to the nth common minute in sorted order.

## Reproduction Workflow

Complete step-by-step workflow to reproduce the dataset from scratch:

### 1. Download the dataset

```bash
pip install requests
python3 training/data/download_dryad.py
```

This downloads the three zip archives to the current directory. To download
to a different location:

```bash
python3 training/data/download_dryad.py --dir /path/to/output
```

Verify the files exist:

```bash
ls -lh Results*Feb2022_SigMF.zip
```

Expected output:
```
-rw-rw-r-- 1 user user 20G ResultsCC1Feb2022_SigMF.zip
-rw-rw-r-- 1 user user 21G ResultsCC2Feb2022_SigMF.zip
-rw-rw-r-- 1 user user 13G ResultsLW1Feb2022_SigMF.zip
```

### 2. Verify download integrity

Each archive contains SigMF metadata/data pairs. Quick verification:

```bash
python3 -c "
from zipfile import ZipFile
for f in ['ResultsCC1Feb2022_SigMF.zip', 'ResultsCC2Feb2022_SigMF.zip', 'ResultsLW1Feb2022_SigMF.zip']:
    z = ZipFile(f)
    meta = [n for n in z.namelist() if n.endswith('.sigmf-meta') and '__MACOSX/' not in n and '/._' not in n]
    data = [n for n in z.namelist() if n.endswith('.sigmf-data') and '__MACOSX/' not in n and '/._' not in n]
    print(f'{f}: {len(meta)} meta, {len(data)} data, total={z.getinfo(meta[0]).file_size if meta else 0}')
    z.close()
"
```

Expected output:
```
ResultsCC1Feb2022_SigMF.zip: 32529 meta, 32529 data
ResultsCC2Feb2022_SigMF.zip: 34865 meta, 34865 data
ResultsLW1Feb2022_SigMF.zip: 21617 meta, 21617 data
```

### 3. Run preprocessing

From the repository root:

```bash
python3 "training/build_training_csv.py" \
  --archive CC1="ResultsCC1Feb2022_SigMF.zip" \
  --archive CC2="ResultsCC2Feb2022_SigMF.zip" \
  --archive LW1="ResultsLW1Feb2022_SigMF.zip" \
  --output "training/data/merged_power_data_sub6GHz_avg_per_minute.csv"
```

This reads each ZIP archive, extracts 250 raw float32 bins at the node's
reverse-engineered bin offset (CC1=21000, CC2=33250, LW1=27500), averages
power per minute in the linear domain, and merges into a single CSV.

### 4. Verify outputs

Check the CSV (no header row):

```bash
# Count rows (data only)
wc -l training/data/merged_power_data_sub6GHz_avg_per_minute.csv
# Expected: 6839

# Count columns
head -1 training/data/merged_power_data_sub6GHz_avg_per_minute.csv | tr ',' '\n' | wc -l
# Expected: 750

# Check file size
ls -lh training/data/merged_power_data_sub6GHz_avg_per_minute.csv
# Expected: ~59 MB
```

Check the manifest:

```bash
python3 -m json.tool training/data/merged_power_data_sub6GHz_avg_per_minute.csv.json
```

Expected manifest structure:
```json
{
  "created_utc": "2026-...",
  "method": "per-node-raw-bin-offsets",
  "n_bins_per_node": 250,
  "row_count": 6839,
  "archives": [
    {"label": "CC1", "archive_path": "ResultsCC1Feb2022_SigMF.zip", "raw_bin_offset": 21000},
    {"label": "CC2", "archive_path": "ResultsCC2Feb2022_SigMF.zip", "raw_bin_offset": 33250},
    {"label": "LW1", "archive_path": "ResultsLW1Feb2022_SigMF.zip", "raw_bin_offset": 27500}
  ]
}
```

### 5. (Optional) Single-node processing

To process only one node:

```bash
python3 "training/build_training_csv.py" \
  --archive LW1="ResultsLW1Feb2022_SigMF.zip" \
  --output "training/data/lw1_power_avg_per_minute.csv"
```

The script automatically uses the correct bin offset for the given label.
This produces a CSV with 250 columns and the node's per-minute averages.

## Script Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--archive LABEL=PATH` | Yes (repeatable) | — | SigMF zip archive with label, e.g. `CC1=ResultsCC1Feb2022_SigMF.zip`. Known labels and their bin offsets: CC1=21000, CC2=33250, LW1=27500. |
| `--output` | No | `training/data/merged_power_data_sub6GHz_avg_per_minute.csv` | Output CSV path (no header, 6 decimal places). A `.json` manifest is written alongside. |

## Troubleshooting

### Missing files / FileNotFoundError

Ensure the zip archives are in the paths passed to `--archive`. The script
requires the exact paths to the `.zip` files.

### No SigMF metadata files found

The zip archive may be corrupted or downloaded incorrectly. Verify the archive
integrity:

```bash
python3 -c "from zipfile import ZipFile; ZipFile('ResultsCC1Feb2022_SigMF.zip').testzip()"
```

An empty output means the archive is intact. Any output listing filenames
indicates corruption.

### Unknown node label

If a label passed to `--archive` is not in `NODE_RAW_BIN_OFFSETS`, the script
raises `ValueError`. The only known labels are `CC1`, `CC2`, and `LW1`.

### No common minute buckets

If no minute keys overlap across all provided archives, the script exits with:
```
No common minute buckets found across the provided archives
```
This can happen if:
- Archives are from different time periods
- A time zone mismatch causes minute keys to not align
- The `--band-start-mhz` or `--band-width-mhz` is set incorrectly

### Interrupted downloads

If downloads are interrupted, re-run the download script. It skips files that
already exist (checked by file size > 1 MB), so it will resume where it left
off. Dryad's servers may rate-limit repeated requests; the script waits 60
seconds and retries automatically when it receives HTTP 403.

### Insufficient disk space

Each zip archive is 12–21 GB. The output CSV is ~59 MB. Ensure at least 60 GB
of free disk space before downloading. The script does NOT extract the archives,
so no additional space is needed beyond the zips and the CSV.

### Processing failures during runtime

The script uses only Python standard library modules — no external dependencies.
If you encounter errors, ensure you are using Python 3.8+ and that the zip
archives are not corrupted.

## Assumptions and Limitations

### Download script (`download_dryad.py`)

- **File stream IDs are hardcoded**: The three Dryad file stream IDs (`4677590`,
  `4677592`, `4677591`) were extracted from the dataset page's
  `a.js-individual-dl` link hrefs at the time of download. If Dryad reorganizes
  the dataset or replaces the files, these IDs will need updating.
- **PoW solver is CPU-intensive**: The Anubis proof-of-work uses all available
  CPU cores via `multiprocessing.Pool` and may take 30–120 seconds per file
  depending on the difficulty level and core count.
- **Resume by file size**: The script skips existing files if their size exceeds
  1 MB. It does not verify checksums, so a partial or corrupted download could
  be treated as complete. Delete the partial file and re-run to force a fresh
  download.
- **Anubis format may change**: If Dryad updates its WAF to use a different
  challenge format (e.g., different HTML structure, different hash algorithm,
  JavaScript-based challenges), the script would break and need updating.

### Build script (`build_training_csv.py`)

- **Per-node bin offsets were reverse-engineered**: The raw bin offsets for CC1
  (21000), CC2 (33250), and LW1 (27500) were discovered by brute-force scanning
  all possible 250-bin windows and matching the repo CSV's statistical profile.
  These offsets are not documented in the original paper and may not generalize
  to other dataset versions or nodes.

## Repository Structure

```
spectrum-usage/
├── training/                       # Training data pipeline
│   ├── build_training_csv.py       # SigMF → per-minute averaged CSV
│   ├── README.md                   # This file
│   └── data/                       # Data directory
│       ├── download_dryad.py       # Dryad downloader (Anubis PoW solver)
│       ├── merged_power_data_sub6GHz_avg_per_minute.csv      # Processed dataset
│       └── merged_power_data_sub6GHz_avg_per_minute.csv.json # Processing manifest
├── evaluation/                     # Evaluation data collection
│   ├── collect_spectrum.py         # USRP-based spectrum acquisition for POWDER/ARA/COSMOS
│   └── README.md
├── models/                         # Model implementations (placeholder)
│   └── README.md
├── plots/                          # Plotting code and generated visualizations
│   └── README.md
├── results/                        # Evaluation results
│   └── README.md
├── ResultsCC1Feb2022_SigMF.zip     # Downloaded zip archives
├── ResultsCC2Feb2022_SigMF.zip
├── ResultsLW1Feb2022_SigMF.zip
└── ...
```

## Verification Checklist

Use this checklist to confirm the pipeline completed successfully:

- [ ] `ResultsCC1Feb2022_SigMF.zip` exists and is ~19 GB
- [ ] `ResultsCC2Feb2022_SigMF.zip` exists and is ~21 GB
- [ ] `ResultsLW1Feb2022_SigMF.zip` exists and is ~12 GB
- [ ] Each zip contains matching `.sigmf-meta` / `.sigmf-data` pairs
- [ ] `training/data/merged_power_data_sub6GHz_avg_per_minute.csv` exists and is ~59 MB
- [ ] CSV has 6839 lines (no header, all data rows)
- [ ] CSV has 750 columns
- [ ] CSV has 0 NaN and 0 Inf values
- [ ] Manifest shows `row_count: 6839`
- [ ] Manifest shows `labels: ["CC1", "CC2", "LW1"]`
- [ ] Manifest shows `method: "per-node-raw-bin-offsets"`
- [ ] Manifest shows `raw_bin_offset: 21000` for CC1, `33250` for CC2, `27500` for LW1
