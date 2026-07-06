# Training Data Pipeline

## Integrated 200 MHz Chunk Models

The integrated training path uses the AERPAW CSV files under `evaluation/aerpaw/` and trains one model per 200 MHz chunk. Each model sees one CC2 chunk as a tensor with shape `(T, 1, 1, 200)`, where `T` is minutes and `200` is the number of 1 MHz bins in the chunk. The shared settings live in `training/common/config.yaml`.

The default chunks are `600-800 MHz`, `2400-2600 MHz`, and `3500-3700 MHz`. The default lookback is 60 minutes. The reported horizons are 1, 5, 15, and 60 minutes. Training can use normalized inputs, but all metrics are written in denormalized dBm.

### Inputs

Put the AERPAW per-minute CSV files in `evaluation/aerpaw/`:

```text
evaluation/aerpaw/ResultsCC1Feb2022_SigMF_power_1mhz_avg_per_minute.csv
evaluation/aerpaw/ResultsCC2Feb2022_SigMF_power_1mhz_avg_per_minute.csv
evaluation/aerpaw/ResultsLW1Feb2022_SigMF_power_1mhz_avg_per_minute.csv
```

Generate these files from the downloaded SigMF ZIP archives with `evaluation/sigmf_zip_to_csv.py`:

```bash
python3 evaluation/sigmf_zip_to_csv.py ResultsCC1Feb2022_SigMF.zip --full-band \
  --output evaluation/aerpaw/ResultsCC1Feb2022_SigMF_power_1mhz_avg_per_minute.csv
python3 evaluation/sigmf_zip_to_csv.py ResultsCC2Feb2022_SigMF.zip --full-band \
  --output evaluation/aerpaw/ResultsCC2Feb2022_SigMF_power_1mhz_avg_per_minute.csv
python3 evaluation/sigmf_zip_to_csv.py ResultsLW1Feb2022_SigMF.zip --full-band \
  --output evaluation/aerpaw/ResultsLW1Feb2022_SigMF_power_1mhz_avg_per_minute.csv
```

Omit `--full-band` to export the default 250 MHz slice.

The training loader reads the configured reference site, interpolates missing values per frequency, selects that site's columns in each configured chunk, and creates chronological train/test splits. The default reference site is `CC2`; the final two days form `CC2_test`.

Per-band metrics use `evaluation/results/step2/band_definitions.csv` when that file exists. The model runners still produce aggregate and per-frequency metrics when band definitions are absent.

### Environment (Docker)

Training was performed inside a Jupyter PyTorch Docker container with CUDA 12 support:

```bash
docker run -d -p 8888:8888 --name jupyter \
  -v /home/cc/spectrum-usage:/home/jovyan/work/spectrum-usage \
  --rm --gpus all \
  quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8
```

The container image includes PyTorch (CUDA-enabled), numpy, pandas, scikit-learn, matplotlib, and Jupyter. After starting the container, attach a shell and navigate to the repo:

```bash
docker exec -it jupyter bash
cd ~/work/spectrum-usage
```

Install additional dependencies:

```bash
pip install pyyaml momentfm==0.1.4 gdown
```

Install `screen` for long-running training jobs (required inside the container):

```bash
apt-get update && apt-get install -y screen
```

### Run Baselines

The evaluation baseline script writes persistence, historical mean, lookback mean, same-time last-3-days mean, AutoReg(60), and LAR metrics. Use `--normalize` to train the learned baselines on normalized inputs and report dBm metrics.

```bash
python3 evaluation/scripts/run_spectrum_steps_5_6.py \
  --normalize \
  --output-dir training/results/baselines
```

### Run LinearAutoRegressive

This runner uses the shared config and trains one direct Ridge autoregressive model per frequency bin, per chunk, and per horizon.

```bash
python3 training/LinearAutoRegressive/train.py
```

Outputs go to `training/results/LinearAutoRegressive/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
chunk_<chunk_id>_training_log.csv
report.txt
checkpoints/
```

### Run ConvLSTM

The integrated ConvLSTM runner trains one model per chunk using `(T, 1, 1, 200)` inputs. It predicts 60 consecutive future minutes and evaluates the configured horizons from that sequence.

```bash
python3 training/ConvLSTM/train_integrated.py
```

Outputs go to `training/results/ConvLSTM/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
<chunk_id>_training_log.csv
checkpoints/
```

For a shorter smoke run, copy `training/common/config.yaml`, reduce `convlstm.epochs`, and pass it with `--config`:

```bash
python3 training/ConvLSTM/train_integrated.py --config /path/to/smoke_config.yaml
```

### Run STS-PredNet

The integrated STS-PredNet runner trains one model per chunk using recursive single-step prediction with closeness and period branches. It evaluates each horizon from the configured list.

```bash
python3 training/STS-PredNet/train_integrated.py
```

Outputs go to `training/results/STS-PredNet/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
<chunk_id>_training_log.csv
checkpoints/
```

### Run TimeRAN

The integrated TimeRAN runner trains a MOMENT forecasting head per chunk using the shared lookback and horizon values.

#### Prerequisites — Download Pretrained Checkpoint

TimeRAN's pretrained backbone weights exceed GitHub's file size limits and must be downloaded from Google Drive before training. The checkpoint path is derived automatically from `config.timeran.checkpoint_size` (default: `base`).

```bash
# Install gdown for Google Drive downloads
pip install momentfm==0.1.4 gdown

# Create checkpoint directories
mkdir -p training/TimeRAN/checkpoints/{small,base,large}

# Download checkpoints from the upstream TimeRAN repository.
#
# NOTE: The upstream TimeRAN README mislabels these file IDs.
# ID 1fJNCk... is the small variant (d_model=512, ~145 MB), NOT base.
# ID 1gz23m... is the base variant (d_model=768, ~433 MB), NOT small.
# We save them with correct names here.
gdown 1fJNCkufmfWC6zHecz10PUyreD0PhBOMJ -O training/TimeRAN/checkpoints/small/TimeRAN_small.pth
gdown 1gz23mmP4ZiNznCloObEaSlVaJH21fyxJ -O training/TimeRAN/checkpoints/base/TimeRAN_base.pth
gdown 1We9zE5BV6Iwkc_EKSAhP28B3wcM7RZRd -O training/TimeRAN/checkpoints/large/TimeRAN_large.pth
```

Without these checkpoints, the pipeline falls back to raw MOMENT weights (no TimeRAN pretraining).

#### Run

```bash
python3 training/TimeRAN/train_integrated.py
```

Outputs go to `training/results/TimeRAN/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
<chunk_id>_training_log.csv
checkpoints/
```

### Run TSS-LCD

The integrated TSS-LCD runner trains a 3-stage latent-conditioned diffusion model per chunk.

Stage 1 trains a Conv2D autoencoder (LSE/LSD) to compress future windows into a latent space.
Stage 2 trains the TSS-CC condition constructor (Temporal/Spectral/Spatial transformer branches) to predict the latent from the lookback window.
Stage 3 trains the diffusion noise-estimation network (Conv1D U-Net) using the latent and TSS-CC condition.

```bash
python3 training/TSS-LCD/train_integrated.py
```

Outputs go to `training/results/TSS-LCD/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
<chunk_id>_training_log.csv
checkpoints/
```

### Run VanillaLSTM

The integrated VanillaLSTM runner trains a direct sequence forecaster per chunk using the shared lookback and horizon settings.

```bash
python3 training/VanillaLSTM/train_integrated.py
```

Outputs go to `training/results/VanillaLSTM/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
<chunk_id>_training_log.csv
checkpoints/
```

Integrated `VanillaLSTM` runs now also export forecast artifacts under a
`forecasts/` subdirectory using the same shared format as the other integrated
forecast exporters.

### Run Autoformer-CSA

The integrated Autoformer-CSA runner trains the restored Autoformer implementation per chunk against the shared chunk pipeline.

```bash
python3 training/Autoformer-CSA/train_integrated.py
```

Outputs go to `training/results/Autoformer-CSA/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
<chunk_id>_training_log.csv
checkpoints/
```

### Run DSwinLSTM-I

The integrated DSwinLSTM-I runner uses the current CSV-based first-pass integration, reshaping each chunk into a pseudo-map before training.

```bash
python3 training/DSwinLSTM-I/train_integrated.py
```

Outputs go to `training/results/DSwinLSTM-I/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
<chunk_id>_training_log.csv
checkpoints/
```

### Run DeepSPred

The integrated DeepSPred runner converts chunk CSV data into colormap spectrogram frames and evaluates the configured minute horizons from frame predictions.

```bash
python3 training/DeepSPred/train_integrated.py
```

Outputs go to `training/results/DeepSPred/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
<chunk_id>_training_log.csv
checkpoints/
```

### Assemble Overall Results

After the baseline and integrated model jobs finish, combine their metric files:

```bash
python3 -m training.common.assemble_results
```

The assembler reads these directories by default:

```text
training/results/baselines/
training/results/LinearAutoRegressive/
training/results/ConvLSTM/
training/results/STS-PredNet/
training/results/TimeRAN/
training/results/TSS-LCD/
training/results/VanillaLSTM/
training/results/Autoformer-CSA/
training/results/DSwinLSTM-I/
training/results/DeepSPred/
```

It writes combined outputs to `training/results/overall/`:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
metrics_summary.md
```

Use `--input-dir` to combine a different set of model output directories:

```bash
python3 -m training.common.assemble_results \
  --input-dir training/results/baselines \
  --input-dir training/results/LinearAutoRegressive \
  --input-dir training/results/ConvLSTM \
  --output-dir training/results/overall
```

### Long-Interval Forecast Plots

The long-interval plot script creates horizon-by-horizon forecast plots for AutoReg, LAR, and lookback mean on a selected CC2 test interval. It writes PNG and CSV files under `evaluation/results/figures/long_interval_forecasts/`.

```bash
python3 evaluation/scripts/plot_autoreg_long_interval_by_horizon.py --transition variable
python3 evaluation/scripts/plot_autoreg_long_interval_by_horizon.py --transition falling
python3 evaluation/scripts/plot_autoreg_long_interval_by_horizon.py --transition rising
```

The `variable` run also writes the compatibility filenames `cc2_autoreg_by_horizon_long_interval.png` and `cc2_autoreg_by_horizon_long_interval.csv`.

### Shared Config

Edit `training/common/config.yaml` to change chunks, horizons, lookback, normalization, or model hyperparameters. The model runners also accept `--config /path/to/config.yaml`.

Use that single shared file for integrated runs. The intended workflow is:

1. edit `training/common/config.yaml`
2. set the current band, map paths, and model hyperparameters
3. run one integrated trainer
4. update the same config file for the next experiment

Key fields:

```yaml
data:
  data_dir: evaluation/aerpaw
  reference_site: CC2
  train_map_path:
  test_map_path:
  map_key: map_db
  chunk_id:
  chunks:
    - id: chunk_600_800
      start_mhz: 600.0
      end_mhz: 800.0

windowing:
  lookback: 60
  horizons: [1, 5, 15, 60]

preprocessing:
  normalize: true

evaluation:
  prediction_start_row:

convlstm:
  input_sequence_length: 60
  prediction_horizon: 60

stsprednet:
  lc: 36
  lp: 3
  period_interval: 1440
  epochs: 25
  learning_rate: 0.0002

timeran:
  checkpoint_size: base
  epochs: 10
  learning_rate: 1.0e-5
  training_mode: linear_probing

tss_lcd:
  autoencoder_epochs: 300
  tss_epochs: 200
  diffusion_epochs: 1000

vanillalstm:
  input_sequence_length: 60
  prediction_horizon: 60

autoformer_csa:
  seq_len: 60
  label_len: 30
  pred_len: 60

dswinlstm_i:
  input_sequence_length: 60
  prediction_horizon: 60

deepspred:
  minutes_per_frame: 60
  input_frames: 1
  output_frames: 1
```

Map-specific fields:

- `data.train_map_path`: interpolated-map `.npz` used for model training
- `data.test_map_path`: interpolated-map `.npz` used for forecasting and evaluation
- `data.map_key`: key inside the `.npz`, usually `map_db`
- `data.chunk_id`: optional label used when naming exported forecast artifacts
- `evaluation.prediction_start_row`: optional 1-based data-row boundary for forecast export and scoring inside `test_map_path`

`prediction_start_row` is useful when the test map contains earlier rows only
for historical context. Rows before that boundary stay available as model
history, but exported forecasts and evaluation begin at the configured row.

Example POWDER map configuration for `600_800`:

```yaml
data:
  train_map_path: data/powder_20260618T0036Z_humanities_guesthouse_600_800.npz
  test_map_path: data/powder_temporal_test_split_humanities_guesthouse_600_800.npz
  map_key: map_db
  chunk_id: powder_600_800
  chunks:
    - id: chunk_600_800
      start_mhz: 600.0
      end_mhz: 800.0

evaluation:
  prediction_start_row: 8883
```

For `2400_2600`, edit the same file and swap `train_map_path`, `test_map_path`, `chunk_id`, and the single entry under `data.chunks`.

Integrated map-mode runs now also export forecasts under the model output directory:

- `forecasts/<chunk_id>_<model>_predictions.npz`
- `forecasts/<chunk_id>_<model>_targets.npz`
- `forecasts/<chunk_id>_<model>_metadata.json`

For `STS-PredNet`, integrated map mode trains only on `data.train_map_path` and uses `data.test_map_path` for context plus evaluation. This avoids fitting on test-era targets while still allowing long-history branches to look back into earlier rows of the test map.

For `ConvLSTM`, the same split-map mechanism applies: training reads `data.train_map_path`, forecasting/evaluation reads `data.test_map_path`, and forecast export starts at `evaluation.prediction_start_row` when that field is set.

Set each model's prediction/input length to at least the largest configured horizon.

## Legacy TSS-LCD Reconstruction

This directory contains the pipeline for acquiring the AERPAW sub-6 GHz spectrum
monitoring dataset and preprocessing it into the format used by the TSS-LCD (https://github.com/Xlab2024/TSS-LCD)
repository. The raw dataset consists of spectrum sweeps collected by three fixed
sensor nodes (CC1, CC2, LW1) during February 2022 on the AERPAW testbed. Each
sweep records power spectral density (PSD) across 87–6019 MHz with ~60 kHz
resolution (98,868 frequency bins). The preprocessing pipeline in this directory:

1. Reads each node's SigMF zip archive in place (no full extraction required)
2. Buckets individual sweeps into one-minute UTC intervals
3. Extracts 250 consecutive raw float32 bins from a node-specific offset in each sweep
4. Converts to linear power, averages per minute, converts back to dBm
5. Merges nodes horizontally into a single 750-column CSV

The output is a per-minute averaged power matrix with 6839 time steps (rows)
and 750 columns, ready for experimentation and comparison against the TSS-LCD
repository CSV.

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
Tools like `datahugger`, `wget`, or `curl` cannot bypass this challenge.
`datahugger` can enumerate the dataset files using the Dryad stash URL, but it
cannot download them because it does not solve the Anubis proof-of-work challenge.

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

### Processing stages (performed by `training/build_training_csv.py`)

1. **Archive enumeration** — The script opens each zip archive and lists all
   `.sigmf-meta` entries (filtering out macOS metadata like `__MACOSX/` or `._`
   files).

2. **Per-sweep extraction** — For each SigMF sweep:
   - The `.sigmf-meta` JSON is parsed to extract the sweep's UTC datetime
   - The datetime is floored to the nearest minute (second=0, microsecond=0)
   - The `.sigmf-data` binary is read as 98,868 float32 values (dBm)
   - **250 consecutive float32 values are sliced** starting at a per-node raw
     bin offset. The offsets used here (CC1=21000, CC2=33250, LW1=27500) were
     determined through reverse-engineering (see "Reverse-Engineered Findings"
     below).

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

- Processing all three archives (52 GB total, ~89K sweeps): typically 8–12
  minutes on the development machine used for testing; actual runtime depends
  on storage speed and CPU performance.
- Most of the time is spent reading and decompressing the zip archives.

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

- **Frequency range:** Per-node frequency bands determined by reverse-engineering
  (see "Reverse-Engineered Findings"): CC1 ~1347–1362 MHz, CC2 ~2082–2097 MHz,
  LW1 ~1737–1752 MHz
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

## Relationship to the Official TSS-LCD Dataset

This pipeline attempts to reconstruct the merged CSV used by the TSS-LCD
repository (`merged_power_data_sub6GHz_avg_per_minute.csv`) from the original
Dryad SigMF source data. The exact preprocessing steps used by the original
paper authors are not publicly documented — the repo CSV was committed as-is
without a generation script.

Several parameters in this pipeline were **reverse-engineered** by matching
the statistical profile of the repository CSV and may differ from the authors'
original preprocessing workflow. See "Reverse-Engineered Findings" below for
details.

The key differences between this reconstruction and the official repo CSV are
documented in the project's reverse-engineering report
(`/home/cc/spectrum-usage/reverse_engineering_report.md`).

### Which script does what

| Step | Script | Purpose |
|------|--------|---------|
| Download from Dryad | `training/data/download_dryad.py` | Solves Anubis PoW, downloads 3 ZIPs |
| Build merged CSV | `training/build_training_csv.py` | Reads ZIPs, extracts 250-bin slices, averages per minute, merges to 750-column CSV |

## Integrated Config

The integrated training and evaluation runners use a single shared config file:

- `training/common/config.yaml`

Training and evaluation are separate scripts. First train, then evaluate:

```bash
# Train
./.venv/bin/python training/ConvLSTM/train_integrated.py --config training/common/config.yaml

# Evaluate (uses checkpoint from training)
./.venv/bin/python training/ConvLSTM/evaluate_integrated.py --config training/common/config.yaml
```

Edit that one config file manually between experiments rather than creating
per-band or per-run YAML variants.  A typical run sequence is:

1. edit `training/common/config.yaml`
2. run `train_integrated.py` for the target model
3. run `evaluate_integrated.py` for the same model
4. update the same config file for the next band or model

### Per-model training and evaluation

| Model | Train | Evaluate |
|-------|-------|----------|
| VanillaLSTM | `training/VanillaLSTM/train_integrated.py` | `training/VanillaLSTM/evaluate_integrated.py` |
| ConvLSTM | `training/ConvLSTM/train_integrated.py` | `training/ConvLSTM/evaluate_integrated.py` |
| STS-PredNet | `training/STS-PredNet/train_integrated.py` | `training/STS-PredNet/evaluate_integrated.py` |
| DeepSPred | `training/DeepSPred/train_integrated.py` | `training/DeepSPred/evaluate_integrated.py` |
| Autoformer-CSA | `training/Autoformer-CSA/train_integrated.py` | `training/Autoformer-CSA/evaluate_integrated.py` |
| DSwinLSTM-I | `training/DSwinLSTM-I/train_integrated.py` | `training/DSwinLSTM-I/evaluate_integrated.py` |
| TimeRAN | `training/TimeRAN/train_integrated.py` | `training/TimeRAN/evaluate_integrated.py` |
| TSS-LCD | `training/TSS-LCD/train_integrated.py` | `training/TSS-LCD/evaluate_integrated.py` |

All scripts accept `--config` (path to `training/common/config.yaml`) and
`--output-dir` (optional override).  The evaluation scripts auto-discover
checkpoints from the training output directory; use `--checkpoint` to override.

For TSS-LCD, evaluation requires three checkpoint flags because training
produces separate autoencoder, TSS-CC, and diffusion checkpoints:

```bash
./.venv/bin/python training/TSS-LCD/evaluate_integrated.py \
    --config training/common/config.yaml \
    --ae-checkpoint  /path/to/autoencoder.pt \
    --tss-checkpoint /path/to/tss.pt \
    --diff-checkpoint /path/to/diffusion.pt
```

### Training outputs

Each `train_integrated.py` run produces under the output directory:

- `checkpoints/<chunk_id>_<model>.pt` — saved model weights, config, and metadata
- `<chunk_id>_training_log.csv` — epoch-level loss history

### Evaluation outputs

Each `evaluate_integrated.py` run produces under the output directory:

- `aggregate_metrics.csv` — per-chunk/per-horizon/per-split aggregate metrics
- `per_frequency_metrics.csv` — per-frequency-bin metrics
- `per_band_metrics.csv` — per-band metrics (if band definitions configured)
- `report.txt` — human-readable summary
- `forecasts/` — forecast artifacts (map-mode only; see below)

### Map train/test fields

For integrated interpolated-map experiments, the shared config supports:

```yaml
data:
  train_map_path:
  test_map_path:
  map_key: map_db
  chunk_id:

evaluation:
  prediction_start_row:
```

Field meanings:

- `data.train_map_path`: path to the interpolated-map `.npz` used for model training
- `data.test_map_path`: path to the interpolated-map `.npz` used for forecasting and evaluation
- `data.map_key`: key inside the `.npz` file, usually `map_db`
- `data.chunk_id`: optional label used when naming exported forecast artifacts
- `evaluation.prediction_start_row`: optional 1-based data-row boundary for test scoring/export inside `test_map_path`

`prediction_start_row` is useful when the test map includes earlier historical
rows for context, but only later rows should count as the actual test region.
Rows before that boundary remain available as model history; forecast export and
evaluation start at the configured row.

### Example POWDER map config

This is the intended single-config workflow for a `600_800` POWDER map run:

```yaml
data:
  train_map_path: data/powder_20260618T0036Z_humanities_guesthouse_600_800.npz
  test_map_path: data/powder_temporal_test_split_humanities_guesthouse_600_800.npz
  map_key: map_db
  chunk_id: powder_600_800
  chunks:
    - id: chunk_600_800
      start_mhz: 600.0
      end_mhz: 800.0

evaluation:
  prediction_start_row: 8883
```

For `2400_2600`, edit the same file and swap:

1. `data.train_map_path`
2. `data.test_map_path`
3. `data.chunk_id`
4. the single entry under `data.chunks`

`prediction_start_row` is a 1-based data-row number. In the example above, rows
before `8883` remain available as historical context, but exported forecasts and
evaluation begin at row `8883`.

### Forecast export

Evaluation in map mode exports saved forecasts under the model output
directory in a `forecasts/` subdirectory:

- `<chunk_id>_<model>_predictions.npz`
- `<chunk_id>_<model>_targets.npz`
- `<chunk_id>_<model>_metadata.json`

The `.npz` payload stores one array per requested horizon, keyed as:

- `t_plus_1`
- `t_plus_5`
- `t_plus_15`
- `t_plus_60`

and also stores the corresponding zero-based target rows for each horizon.

### Training-Evaluation Separation

All integrated models now separate training from evaluation:

- **Training** (`train_integrated.py`): loads the training set, fits the model,
  saves a checkpoint (and optionally a training log CSV).  No test-set inference
  or metric computation happens during training.

- **Evaluation** (`evaluate_integrated.py`): loads a saved checkpoint, runs
  inference on the test set, computes aggregate/per-frequency/per-band metrics,
  writes metric CSVs and `report.txt`, and exports forecast artifacts in map
  mode.

This split ensures that training and test-set evaluation are independent
steps that can be run at different times, on different hardware, or with
different config overrides.

## Reverse-Engineered Findings

The following parameters were **not** documented in the AERPAW paper or Dryad
metadata. They were discovered by comparing the raw SigMF data against the
TSS-LCD repository's merged CSV.

### Per-node raw bin offsets

The repository CSV uses **different 250-bin frequency ranges per node**, not a
single shared band as implied by the paper's "85–335 MHz" claim. The discovered
offsets and their approximate frequencies are:

| Node | Raw bin offset | Approximate frequency range | Rationale |
|------|---------------|-----------------------------|-----------|
| CC1 | 21000 | ~1347–1362 MHz | Matched repo CSV mean, freq-std, and temporal std |
| CC2 | 33250 | ~2082–2097 MHz | Matched repo CSV mean, freq-std, and temporal std |
| LW1 | 27500 | ~1737–1752 MHz | Matched repo CSV mean, freq-std, and temporal std |

These offsets were the only 250-bin windows (out of ~395 candidates per node)
whose statistical profile — mean power, frequency standard deviation (~0.85–1.72
dBm), and adjacent-bin correlation (~0.998) — matched the official repo CSV.

### Why not 85–335 MHz

The 85–335 MHz band contains strong VHF/UHF signals (FM radio, TV, cellular)
producing ~13 dBm frequency standard deviation across 250 bins. The repo CSV
has only ~0.85–1.72 dBm frequency standard deviation — consistent with a
thermal-noise-floor region, not a signal-rich band. This discrepancy was the
primary clue that the repo CSV used different frequency ranges.

### How the offsets were discovered

A brute-force scan tested every possible 250-bin window across the full 87–6019
MHz spectrum for each node (~395 windows × 200 sweeps per window). For each
window, the per-bin mean, per-bin frequency standard deviation, and temporal
standard deviation were computed and compared against the repo CSV's per-node
statistics. Only the three windows listed above matched all three metrics
simultaneously.

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

- **Per-node bin offsets were reverse-engineered**: See "Reverse-Engineered
  Findings" above for how CC1=21000, CC2=33250, and LW1=27500 were discovered.
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
