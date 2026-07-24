# Training Data Pipeline

## Integrated Pipeline Overview

The training pipeline is organized around a shared `training/common/` package that provides all
data-loading, preprocessing, windowing, model-construction, forecasting, and result-generation
utilities.  The same code path is used for every supported model family.

**Main entry points** (both in `training/common/`):

- `train_integrated.py` — model-agnostic trainer that iterates over configured frequency chunks
- `evaluation_integrated.py` — model-agnostic evaluator that loads a saved checkpoint, runs
  inference on the test split, and writes metrics, forecast artifacts, and plots

The model is selected through the shared configuration key `training.model_name`.  The pipeline
supports:

| Model | Config name | Data representation | Model layout |
|-------|-------------|---------------------|--------------|
| Vanilla LSTM | `vanillalstm` | CSV frequency vectors | `(B, T, F)` |
| ConvLSTM | `convlstm` | Spectrum maps | `(B, T, F, H, W)` |
| ConvLSTM-FM | `convlstmfm` | Spectrum maps | `(B, T, F, H, W)` |
| DSwinLSTM-I | `dswinlstm_i` | Spectrum maps | `(B, T, F, H, W)` |
| Autoformer-CSA | `autoformer_csa` | CSV frequency vectors | `(B, T, F)` |
| TimeRAN | `timeran` | CSV frequency vectors | `(B, T, F)` |
| Lookback Mean | `lookbackmean1d`/`2d`/`4d` | CSV / maps | `(B, T, F)` or `(B, T, F, H, W)` |
| Linear AutoRegressive | `linearar1d`/`2d`/`4d` | CSV / maps | `(B, T, F)` or `(B, T, F, H, W)` |
| Residual Vanilla LSTM | `residualvanillalstm` | CSV frequency vectors | `(B, T, F)` |
| Residual ConvLSTM | `residualconvlstm` | Spectrum maps | `(B, T, F, H, W)` |
| Residual Linear AR | `residuallinearar1d`/`2d`/`4d` | CSV / maps | `(B, T, F)` or `(B, T, F, H, W)` |
| TCN | `temporalconvnet` | CSV frequency vectors; joint or independent features | `(B, T, F)` |
| LSTM-Attention | `lstmattn` | CSV frequency vectors | `(B, T, F)` |
| ARIMA | `arima` | CSV frequency vectors | `(B, T, F)` |

Architecture classes live under `models/` (e.g. `VanillaLSTM.py`, `ConvLSTM.py`,
`DSwinLSTM_I.py`).  The shared model factory (`model_factory.py`) imports the correct
class and derives input dimensions from the loaded training data.

### Shared forecasting behavior

One-step models (`prediction_horizon=1`) use:

- **teacher-forced rollout** during supervised training — each step receives the ground-truth
  target as the next input; and
- **autoregressive rollout** when future targets are not supplied (validation and final evaluation)
  — each step feeds its own prediction back into the input window.

Direct multi-step models (`prediction_horizon == rollout_horizon`) produce the full forecast
horizon in a single forward call.

Training, validation, and evaluation all call the same `forecast()` function from
`training/common/forecasting.py`.

**Per-horizon validation:** During validation, `train_integrated.py` tracks per-step MSE
for each configured horizon (`val_loss_t{h}` columns in the training log).  A separate
teacher-forcing diagnostic pass computes `val_teacher_loss` (one-step models only).

### Data representations

Source data layouts (as read by the loaders):

```text
CSV spectrum data:
(T, F)

Spectrum-map data:
(T, H, W, F)
```

The shared windowing and layout utilities (`windowing.py`) convert these into the layouts expected
by the selected model.  Users do not manually reshape the data before training.

### Dataset behavior

The integrated data interface loads all dataset types through the single entry point
`load_chunk()` in `training/common/data.py`.  It reads source files configured under
`data.files` (each entry has a `path` and `partition: train|test`), dispatches to
`data_sources.py` for raw CSV/NPZ loading, and for 4D maps uses `map_builder.py` to
construct spatial grids from collection-point coordinates.

Source files are partitioned into train and test sets by their `partition` label.
Validation is carved from the training partition.  Loaded splits carry sequence
segments so windows cannot cross files, sites, or frequency-bin boundaries.

Frequency chunks are configured centrally under `data.chunks`.  Training runs
independently for each configured chunk.

When `data.prediction_start_row` is configured, the non-overlapping portion of the
long test recording before that row extends the training set.  Timestamps are required
for this extension.  After chronological construction, optional `data.max_rows`
truncation is applied.  Completely unusable map timesteps (all-NaN) are removed
before map cleaning and normalization.

### Normalization and imputation

Normalization is per-frequency (z-score).  Statistics are:

- fitted only from the training portion *before* the validation split;
- reused unchanged for validation and testing;
- stored with the checkpoint for later denormalization; and
- used to convert predictions back to dBm during evaluation (via `metrics.py`).

Validation and test data never fit their own normalization statistics.

When `preprocessing.impute: true` in config, missing values are filled by
`clean_spectrum_data()` in `training/common/preprocessing.py`:
- **4D map data** (`T, H, W, F`): imputation is disabled — any NaN values
  raise an error. All-NaN timesteps are removed earlier in the data-loading
  pipeline.
- **2D CSV data** (`T, F`): temporal linear interpolation per frequency column.
- After imputation any remaining NaN or non-finite values raise an error.

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
pip install pyyaml gdown
pip install setuptools --upgrade
pip install numpy
pip install momentfm==0.1.4 --no-deps
pip install transformers
```

**Note:** `momentfm==0.1.4` pins exact versions of `numpy`, `huggingface-hub`, and
`transformers` that lack Python 3.13 wheels.  The sequence above installs the
package without its pinned dependencies (`--no-deps`) and separately installs
`transformers` (latest) and `numpy` (system), which avoids Rust build requirements
for `tokenizers` and `numpy` on Python 3.13.

Install `screen` for long-running training jobs (required inside the container):

```bash
apt-get update && apt-get install -y screen
```

## Quickstart

The basic workflow is three steps:

```bash
# 1. Edit the shared configuration
#    Set training.model_name, data.files, data.chunks, and model-specific params
#    vim training/common/config.yaml

# 2. Train
python3 training/common/train_integrated.py \
    --config training/common/config.yaml \
    --name my_experiment

# 3. Evaluate
python3 training/common/evaluation_integrated.py \
    --config training/common/config.yaml \
    --name my_experiment
```

All models use the same two entry points.  Model-specific settings live in the
shared config under `{model_name}.model` and `{model_name}.train`.
See [Training](#training) and [Evaluation](#evaluation) below for all available
flags.

## Training

### Training command

```bash
python3 training/common/train_integrated.py [--config CONFIG] [--name NAME] [--output-dir PATH]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `training/common/config.yaml` | Path to the YAML configuration file |
| `--name NAME` | `<model>_<timestamp>` | Experiment name; outputs go to `runs/<name>/` |
| `--output-dir PATH` | `runs/<name>/` | Override the output directory (alternative to `--name`) |

Example:

```bash
python3 training/common/train_integrated.py --config training/common/config.yaml
```

The trainer reads `training.model_name` from the config, iterates over each
chunk in `data.chunks`, and independently trains one model per chunk.

### Training outputs

```text
runs/<experiment_name>/
  config.yaml                            # Copy of the config used for training
  checkpoints/
    {chunk_id}_{model_name}.pt            # Best-epoch state dict + metadata
  {chunk_id}_training_log.csv             # Per-epoch loss history
  {chunk_id}_training_summary.txt         # Best-epoch summary statistics
```

Each checkpoint stores:

- `model_name` — validated against config on load
- `model_state_dict` — best-epoch weights
- `normalization` — per-frequency mean/std for denormalization during evaluation
- `frequencies` — frequency array, validated against loader output on evaluation
- `training_results` — best/final/average losses and timing

## Evaluation

### Evaluation command

```bash
python3 training/common/evaluation_integrated.py --name <experiment_name>
```

One of `--name` or `--output-dir` is required.  The evaluator locates checkpoints
inside the run directory.

| Flag | Description |
|------|-------------|
| `--name NAME` | Experiment name (maps to `runs/<name>/`) |
| `--output-dir PATH` | Direct path to the run directory (alternative to `--name`) |
| `--config PATH` | Config path (defaults to `training/common/config.yaml`) |
| `--checkpoint PATH` | Override checkpoint path; supports `{chunk_id}` and `{model_name}` substitution |
| `--skip-plots` | Skip forecast plot generation |

Example:

```bash
python3 training/common/evaluation_integrated.py --name dswinlstm_i_20260722_190853
```

### Evaluation outputs

Written to the run directory alongside training outputs:

```text
aggregate_metrics.csv        # Per-chunk, per-horizon, per-split aggregate metrics
per_frequency_metrics.csv    # Per-frequency-bin metrics
per_band_metrics.csv         # Per-frequency-band metrics (if band definitions exist)
report.txt                   # Human-readable summary
forecasts/                   # Forecast artifacts (map mode only)
```

### What happens during evaluation

1. The shared loader and preprocessing recreate the same train/test split used
   during training.
2. The model factory builds the configured architecture.
3. The checkpoint weights are loaded and validated against the current model name,
   frequencies, and normalization statistics.
4. The test split is windowed with the configured lookback.
5. For one-step models, predictions are generated via autoregressive rollout.
   Direct multi-step models produce the full horizon in one forward pass.
6. Normalized predictions are denormalized back to dBm when normalization is enabled.
7. Aggregate, per-frequency, and band-level MAE and RMSE are calculated.
8. Forecast data and evaluation metadata are exported (map mode: `.npz` + `.json`).
9. Optionally, forecast and error plots are generated.

## Model-specific considerations

All models use the same `train_integrated.py` / `evaluation_integrated.py` entry
points.  The table below notes their specific requirements:

| Model | Config name | Data rep | Layout | Notes |
|-------|-------------|----------|--------|-------|
| Vanilla LSTM | `vanillalstm` | 1d | `(B,T,F)` | One-step autoregressive |
| ConvLSTM | `convlstm` | 4d | `(B,T,F,H,W)` | Has validation magnitude guard |
| ConvLSTM-FM | `convlstmfm` | 4d | `(B,T,F,H,W)` | One-step; optional masked-reconstruction backbone pretraining before fine-tuning (see `training/ConvLSTM-FM/README.md`) |
| DSwinLSTM-I | `dswinlstm_i` | 4d | `(B,T,F,H,W)` | Multi-step direct; `prediction_horizon == rollout_horizon` |
| Autoformer-CSA | `autoformer_csa` | 2d | `(B,T,F)` | Multi-step direct; `prediction_horizon == rollout_horizon`; `input_sequence_length` must match `seq_len` |
| TimeRAN | `timeran` | 1d | `(B,T,F)` | Requires pretrained MOMENT checkpoint download (see "Run TimeRAN" below) |
| Lookback Mean | `lookbackmean1d/2d/4d` | any | varies | Parameter-free baseline; predicts mean of lookback window |
| Linear AR | `linearar1d/2d/4d` | any | varies | Ridge regression per bin |
| Residual Vanilla LSTM | `residualvanillalstm` | 1d | `(B,T,F)` | Predicts residuals of lookback mean |
| Residual ConvLSTM | `residualconvlstm` | 4d | `(B,T,F,H,W)` | Predicts residuals of lookback mean |
| Residual Linear AR | `residuallinearar1d/2d/4d` | any | varies | Predicts residuals of lookback mean |
| TCN | `temporalconvnet` | 1d/2d | `(B,T,F)` | `feature_mode: independent` applies a shared local TCN per bin; `joint` mixes bins as convolution channels |
| LSTM-Attention | `lstmattn` | 2d | `(B,T,F)` | Joint multi-bin LSTM with temporal attention |
| ARIMA | `arima` | 1d | `(B,T,F)` | Classical ARIMA model |

## Configuration reference

The single shared file `training/common/config.yaml` controls every aspect of
training and evaluation.  Key top-level sections:

```yaml
data:
  representation: 2d             # "2d" (CSV vectors) or "4d" (spatial maps)
  band_definitions_path: ...      # Optional CSV defining frequency bands for metrics
  reference_site: guesthouse      # Name used for split keys
  files:                          # List of source files with partition labels
    - path: data/<file>.csv
      partition: train            # "train" or "test"
  map:                            # Required for 4d representation
    name: powder_600_800          # Map name (used for cache filename)
    locations: data/locations/powder.json  # Collection-point coordinates JSON
    collection_key: endpoints     # Key into the locations JSON
    output_dir: data/maps         # Directory for cached .npz map files
    grid:                         # Spatial grid dimensions
      height: 10
      width: 10
    force_rebuild: false          # Set true to regenerate cached maps
    permute: false                # Randomize site ordering
    permute_seed: 42
  concat: rows                    # Row-wise CSV concatenation (1d/2d only)
  frequency_bins:                 # Optional: select specific frequency bins
  frequency_ranges:               # Optional: select frequency ranges
  mask:                           # Optional: frequency masking
    frequency_ranges: [[650.0, 660.0]]
    noise_floor: -120.0
  prediction_start_row:           # 1-based row where test evaluation begins
  max_rows:                       # Optional row limit for debugging
  chunks:
    - id: powder_600_800
      start_mhz: 600.0
      end_mhz: 800.0

data_loader:
  num_workers: 0
  pin_memory: auto

windowing:
  lookback: 60                    # Input sequence length
  horizons: [1, 5, 15, 60]       # Forecast horizons to report

training:
  device: auto                    # "auto", "cuda", or "cpu"
  model_name: dswinlstm_i         # Model to train (see supported models table)

preprocessing:
  normalize: true                 # Per-frequency z-score normalization
  impute: true                    # NaN imputation
  max_missing_gap: 5              # Max consecutive NaNs to interpolate
```

Model-specific configuration lives under each model's key (e.g. `dswinlstm_i.model.*`,
`vanillalstm.train.*`).  See `training/common/config.yaml` for the full set of parameters.

### Optimizer configuration

The integrated trainer dispatches the optimizer based on the `optimizer` field in the
model's `train` section.  Supported values: `adam`, `adamw`, `sgd` (when using SGD
the optional `momentum` field may also be set).

## Pipeline walkthrough

### Training flow

For each chunk in `data.chunks`, `train_integrated.py`:

1. Loads source files via `load_chunk()` — reads CSVs, builds 4D maps if needed,
   applies frequency selection, imputation, normalization, and splits into train/validation/test.
2. Builds temporal windows via `build_window_loaders()` — creates overlapping
   lookback → horizon windows from the training and validation splits.
3. Constructs the model through `build_model()` — the factory selects the architecture
   class, infers input dimensions from training data, and returns an uninitialized model.
4. Moves the model to the configured device and initializes the optimizer.
5. For each epoch: trains on all training windows (teacher-forced for one-step models),
   then validates on all validation windows (autoregressive rollout).
6. Saves a checkpoint with the best validation epoch's weights, normalization stats,
   frequencies, and model metadata.

### Evaluation flow

`evaluation_integrated.py`:

1. Loads the same config and recreates the train/test split.  The test split is
   identical to what was used during training (same files, same preprocessing).
2. Builds the model through the factory with the same architecture.
3. Loads the checkpoint — validates that model name, frequencies, and normalization
   statistics match the current config and data.
4. Windows the test split with the configured lookback.
5. Runs inference: autoregressive rollout for one-step models, single forward pass
   for direct multi-step models.
6. Denormalizes predictions back to dBm.
7. Computes aggregate, per-frequency, and band-level MAE and RMSE.
8. Exports forecasts as `.npz` + `.json` (map mode only).
9. Optionally generates forecast and error plots.

### Run STS-PredNet

The integrated STS-PredNet runner trains one model per chunk using recursive single-step prediction with closeness and period branches.

```bash
python3 training/STS-PredNet/train_integrated.py
```

Training outputs go to `training/results/STS-PredNet/` by default:

```text
<chunk_id>_training_log.csv
checkpoints/
```

#### Evaluate

Loads the checkpoint saved by training, runs inference on the test set, and writes metrics.

```bash
python3 training/STS-PredNet/evaluate_integrated.py
```

Evaluation outputs go to `training/results/STS-PredNet/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
```

### Run TimeRAN

TimeRAN is now supported by the shared integrated pipeline.  See
"Training through the common integrated pipeline" above.

TimeRAN trains a MOMENT forecasting head per chunk.  Its pretrained backbone
weights exceed GitHub's file size limits and must be downloaded separately.
Without these checkpoints, the pipeline falls back to raw MOMENT weights (no
TimeRAN pretraining).

```bash
# Install momentfm and gdown (see Docker/Environment section for Python 3.13 notes)
pip install gdown
pip install momentfm==0.1.4 --no-deps

# Create checkpoint directories
mkdir -p training/TimeRAN/checkpoints/{small,base,large}

# NOTE: The upstream TimeRAN README mislabels these file IDs.
# ID 1fJNCk... is the small variant (d_model=512, ~145 MB), NOT base.
# ID 1gz23m... is the base variant (d_model=768, ~433 MB), NOT small.
gdown 1fJNCkufmfWC6zHecz10PUyreD0PhBOMJ -O training/TimeRAN/checkpoints/small/TimeRAN_small.pth
gdown 1gz23mmP4ZiNznCloObEaSlVaJH21fyxJ -O training/TimeRAN/checkpoints/base/TimeRAN_base.pth
gdown 1We9zE5BV6Iwkc_EKSAhP28B3wcM7RZRd -O training/TimeRAN/checkpoints/large/TimeRAN_large.pth
```

Set `training.model_name: timeran` in the configuration and use
`training/common/train_integrated.py` and
`training/common/evaluation_integrated.py`. TimeRAN has no model-specific
integrated runner.

### Run TSS-LCD

The integrated TSS-LCD runner trains a 3-stage latent-conditioned diffusion model per chunk.

Stage 1 trains a Conv2D autoencoder (LSE/LSD) to compress future windows into a latent space.
Stage 2 trains the TSS-CC condition constructor (Temporal/Spectral/Spatial transformer branches) to predict the latent from the lookback window.
Stage 3 trains the diffusion noise-estimation network (Conv1D U-Net) using the latent and TSS-CC condition.

```bash
python3 training/TSS-LCD/train_integrated.py
```

Training outputs go to `training/results/TSS-LCD/` by default:

```text
<chunk_id>_training_log.csv
checkpoints/
```

#### Evaluate

Loads the three checkpoints saved by training, runs inference on the test set, and writes metrics.

Because TSS-LCD produces separate weights for each stage, three checkpoint flags are required:

```bash
python3 training/TSS-LCD/evaluate_integrated.py \
    --ae-checkpoint  training/results/TSS-LCD/checkpoints/<chunk_id>_autoencoder.pt \
    --tss-checkpoint training/results/TSS-LCD/checkpoints/<chunk_id>_tss.pt \
    --diff-checkpoint training/results/TSS-LCD/checkpoints/<chunk_id>_diffusion.pt
```

Evaluation outputs go to `training/results/TSS-LCD/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
```

### Run VanillaLSTM

Vanilla LSTM is supported by the shared integrated pipeline. See
"Training through the common integrated pipeline" above. The old
model-specific evaluator has been removed.

### Run Autoformer-CSA

Autoformer-CSA is supported by the shared integrated pipeline.  It uses
CSV vector data (`representation: 2d`) and produces direct multi-step
forecasts (`prediction_horizon == rollout_horizon`).  Set
`training.model_name: autoformer_csa` in the configuration and use
`training/common/train_integrated.py` and
`training/common/evaluation_integrated.py`.

Key configuration fields under `autoformer_csa.model`:

| Field | Description |
|-------|-------------|
| `input_sequence_length` | Encoder input length (`seq_len` in Autoformer terms); must match the lookback window size |
| `prediction_horizon` | Forecast horizon (`pred_len`); must equal `max(windowing.horizons)` |
| `label_len` | Decoder's known future timesteps (uses the last `label_len` steps of the encoder input) |
| `d_model` | Transformer model dimension |
| `d_ff` | Feed-forward network hidden dimension |
| `encoder_layers` / `decoder_layers` | Number of encoder/decoder layers |
| `n_heads` | Number of attention heads |
| `moving_avg` | Moving average kernel size |
| `dropout` | Dropout rate |
| `factor` | Attention factor |
| `csam_kernel_size` | Kernel size for CSA-MLP block |
| `output_attention` | Whether to output attention weights (default `false`) |

Training hyperparameters live under `autoformer_csa.train` (`batch_size`,
`epochs`, `learning_rate`, etc.).  No separate runner or checkpoint download
is needed.

### Run DSwinLSTM-I

The integrated DSwinLSTM-I runner uses CSV-based first-pass integration, reshaping each chunk into a pseudo-map before training.

```bash
python3 training/DSwinLSTM-I/train_integrated.py
```

Training outputs go to `training/results/DSwinLSTM-I/` by default:

```text
<chunk_id>_training_log.csv
checkpoints/
```

#### Evaluate

Loads the checkpoint saved by training, runs inference on the test set, and writes metrics.

```bash
python3 training/DSwinLSTM-I/evaluate_integrated.py
```

Evaluation outputs go to `training/results/DSwinLSTM-I/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
```

### Run DeepSPred

The integrated DeepSPred runner converts chunk CSV data into colormap spectrogram frames.

```bash
python3 training/DeepSPred/train_integrated.py
```

Training outputs go to `training/results/DeepSPred/` by default:

```text
<chunk_id>_training_log.csv
checkpoints/
```

#### Evaluate

Loads the checkpoint saved by training, runs inference on the test set, and writes metrics.

```bash
python3 training/DeepSPred/evaluate_integrated.py
```

Evaluation outputs go to `training/results/DeepSPred/` by default:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
report.txt
```

### Assemble Overall Results

After training and evaluation jobs finish, combine their metric files:

```bash
python3 -m training.common.assemble_results
```

By default the assembler reads from `training/results/*` (one directory per
model).  For results from specific runs, use `--input-dir` to point to each
model's output directory and `--output-dir` to set the combined output location:

```bash
python3 -m training.common.assemble_results \
  --input-dir runs/<name>/vanillalstm \
  --input-dir runs/<name>/convlstm \
  --output-dir runs/overall
```

The default output directory (when `--output-dir` is omitted) is
`runs/overall/`.  Combined outputs:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
metrics_summary.md
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

All integrated training and evaluation settings live in the single shared file
`training/common/config.yaml`.  See the **Configuration reference** section under
"Training through the common integrated pipeline" above for the current keys and
their descriptions.

The intended workflow is:

1. edit `training/common/config.yaml`
2. run `training/common/train_integrated.py`
3. run `training/common/evaluation_integrated.py`
4. update the same config file for the next experiment

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



## Troubleshooting

### Missing timestamps while prediction_start_row is enabled

The POWDER loader requires `timestamp_utc` columns in CSV files or a `timestamps`
array in `.npz` archives when `data.prediction_start_row` is configured.  If
timestamps are missing, the loader raises an error.  Either add timestamps to the
source files or remove `prediction_start_row` from the config.

### Mismatched train/test frequency columns

The POWDER loader validates that all files in the same split share the same
frequency columns.  Train and test files must also contain the configured chunk
range.  If the column sets differ, the loader raises an error.  Check that the
files were generated with consistent frequency boundaries.

### Mismatched map frequencies

When training with spectrum maps, the pipeline validates that train and test
partitions contain identical frequency arrays.  If they differ, evaluation
statistics will be misaligned.  Verify that all source files under `data.files`
were generated with consistent frequency ranges.

### Incorrect map shape

The map loader expects 4D arrays shaped `(T, H, W, F)`.  Archives with different
dimensionality raise an error.  Verify the source `.npz` with `np.load(path)
[map_key].shape`.

### Insufficient rows for lookback and horizon

The windowing module requires at least `lookback + rollout_horizon` rows in
each split.  If the training or test split is too short, a `ValueError` is
raised.  Shorten `windowing.lookback`, reduce `horizons`, use smaller chunks,
or collect longer recordings.

### Empty training or validation windows

If `val_fraction` is too large or the training split is very short, the
training or validation portion may contain zero valid windows.  Reduce
`val_fraction` or increase the training recording length.

### CUDA device unavailable

Set `training.device: cpu` in the config, or ensure a CUDA-capable GPU and
compatible PyTorch version are available.  When `device: auto` is set, the
pipeline falls back to CPU if CUDA is not found.

### Checkpoint / configuration mismatch

The evaluation script validates that the checkpoint model name matches the
configured `training.model_name`.  If they differ, evaluation raises an error.
Use the same config for training and evaluation, or pass `--checkpoint` to
specify the correct checkpoint path.

### Checkpoint frequency mismatch

The checkpoint stores the frequency array used during training.  If evaluation
loads data with different frequencies, the loader raises an error.  Confirm
that the same config (same chunk ranges) is used for both training and
evaluation.

### Wrong model choice for the configured data

Vanilla LSTM expects 2D frequency-vector data `(T, F)`.  ConvLSTM expects 4D
map data `(T, H, W, F)`.  If the model name in the config does not match the
loaded data shape, the model factory raises a `ValueError`.  Verify that
`training.model_name` is consistent with the data source (CSV vs. map).

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
├── training/                       # Training and evaluation pipeline
│   ├── build_training_csv.py       # SigMF → per-minute averaged CSV
│   ├── README.md                   # This file
│   ├── common/                     # Shared integrated pipeline
│   │   ├── config.yaml             # Central configuration
│   │   ├── config.smoke.yaml       # Smoke-test configuration
│   │   ├── config.py               # YAML loading and path resolution
│   │   ├── data.py                 # Chunk specs and loader dispatch
│   │   ├── data_sources.py         # Raw CSV/NPZ file loading
│   │   ├── data_loader.py          # PyTorch DataLoader construction
│   │   ├── map_builder.py          # 4D spatial map construction from CSVs + locations
│   │   ├── preprocessing.py        # Normalization, imputation, cleaning
│   │   ├── windowing.py            # Temporal window construction, layout conversion
│   │   ├── validation_diagnostics.py # Prediction guard checks, diagnostics
│   │   ├── forecasting.py          # Teacher-forced / autoregressive / direct forecast
│   │   ├── model_factory.py        # Model construction and checkpoint handling
│   │   ├── train_integrated.py     # Model-agnostic training entry point
│   │   ├── evaluation_integrated.py# Model-agnostic evaluation entry point
│   │   ├── metrics.py              # Denormalization and error calculation
│   │   ├── results.py              # Metric aggregation and CSV output
│   │   ├── forecast_export.py      # Forecast artifact serialization
│   │   ├── plot_forecasts.py       # Forecast and error plot generation
│   │   ├── runtime.py              # Device selection, timestamps, log rows
│   │   ├── pipeline_train_trace.txt# Detailed pipeline execution trace
│   │   ├── training_configs.txt    # Training config snapshots
│   │   ├── assemble_results.py     # Cross-model result assembly
│   │   ├── test_data_loader.py     # DataLoader tests
│   │   └── test_map_builder.py     # Map builder tests
│   ├── grid_search/                # Hyperparameter search scripts and results
│   │   ├── grid_search.py          # Grid search runner
│   │   ├── plot_hyperparameter_search.py # Search result visualization
│   │   ├── config_convlstm.yaml    # ConvLSTM grid search config
│   │   └── results/                # Search artifacts, plots, summaries
│   ├── data/                       # Dataset files
│   │   ├── download_dryad.py       # Dryad downloader (Anubis PoW solver)
│   │   └── merged_power_data_sub6GHz_avg_per_minute.csv(.json)
│   ├── ConvLSTM/                   # ConvLSTM-specific scripts (legacy)
│   ├── VanillaLSTM/                # VanillaLSTM-specific scripts (legacy)
│   ├── STS-PredNet/                # STS-PredNet entry points
│   ├── TimeRAN/                    # TimeRAN entry points
│   ├── TSS-LCD/                    # TSS-LCD entry points
│   ├── Autoformer-CSA/             # Autoformer-CSA entry points
│   ├── DSwinLSTM-I/                # DSwinLSTM-I entry points
│   ├── DeepSPred/                  # DeepSPred entry points
│   └── LinearAutoRegressive/       # LinearAutoRegressive entry points
├── models/                         # Model architecture definitions
│   ├── VanillaLSTM.py              # VanillaLSTMForecaster
│   ├── ConvLSTM.py                 # ConvLSTMForecaster
│   ├── DSwinLSTM_I.py              # DSwinLSTM-IForecaster
│   ├── TimeRAN.py                  # TimeRANForecaster
│   ├── LinearAutoregressive.py     # LinearAutoregressiveForecaster
│   ├── LookbackMean.py             # LookbackMeanForecaster
│   ├── ResidualVanillaLSTM.py      # ResidualVanillaLSTMForecaster
│   ├── ResidualConvLSTM.py         # ResidualConvLSTMForecaster
│   ├── ResidualLinearAutoregressive.py # ResidualLinearAutoregressiveForecaster
│   └── README.md
├── evaluation/                     # Evaluation data collection
│   ├── collect_spectrum.py         # USRP-based spectrum acquisition
│   └── README.md
├── plots/                          # Plotting code and visualizations
│   └── README.md
├── results/                        # Evaluation results
│   └── README.md
├── Miscellaneous/                  # Experimental runs (hidden_size experiments, etc.)
├── presentation/                   # Presentation artifacts organized by date
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
