# ConvLSTM Spectrum Prediction — Reconstructed Model

> **Based on:** *Convolutional LSTM-based Long-Term Spectrum Prediction for Dynamic Spectrum Access* — Shawel, Woldegebreal, Pollin (EUSIPCO 2019)
>
> **Reference implementation:** https://github.com/ndrplz/ConvLSTM_pytorch (cloned locally as reference)
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

---

## Quick Start

### Setup

```bash
cd /home/cc/spectrum-usage
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib pyyaml scikit-learn tqdm
```

### Train the Model

```bash
# Default config (T_in=12, T_out=6, 80/10/10 split)
python3 training/ConvLSTM/train.py

# With overrides
python3 training/ConvLSTM/train.py \
    --batch-size 64 \
    --epochs 150 \
    --lr 0.0001 \
    --input-len 24 \
    --pred-horizon 12
```

Training creates `training/ConvLSTM/checkpoints/` with `best_model.pt`, `last_model.pt`, and `normalization_stats.pt`.

> **Note:** Checkpoints contain model weights, optimizer state, and normalization stats. They exceed GitHub's 100 MB file limit and are gitignored. To use a trained model on another machine, either re-train there or manually copy the `checkpoints/` directory.

### Evaluate

```bash
python3 training/ConvLSTM/evaluate.py \
    --checkpoint training/ConvLSTM/checkpoints/best_model.pt
```

Output: per-horizon and per-node RMSE/MAE/R², spectrogram plots, full-test `predictions.csv` / `ground_truth.csv`, and `metadata.json`.

### Run Inference on New Data

```bash
python3 training/ConvLSTM/inference.py \
    --checkpoint training/ConvLSTM/checkpoints/best_model.pt \
    --input /path/to/new_measurements.csv \
    --output predictions.csv
```

---

## Scripts Reference

### `train.py` — Train a new model

```bash
python3 training/ConvLSTM/train.py [--config CONFIG] [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration file |
| `--batch-size` | from config | Override batch size |
| `--epochs` | from config | Override max epochs |
| `--lr` | from config | Override learning rate |
| `--input-len` | from config | Override input sequence length (T_in) |
| `--pred-horizon` | from config | Override prediction horizon (T_out) |

Output: `checkpoints/best_model.pt`, `checkpoints/last_model.pt`, `checkpoints/normalization_stats.pt`.

### `evaluate.py` — Evaluate a trained model

```bash
python3 training/ConvLSTM/evaluate.py --checkpoint CHECKPOINT [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` checkpoint from training (required) |
| `--config` | from checkpoint | Path to config (overrides checkpoint's embedded config) |
| `--horizons` | `[1, 3, 6]` | Specific future time steps to report metrics for |
| `--output` | `evaluation/` | Output directory for metrics, plots, and CSVs |

Output (CSV mode): `evaluation/metrics.json`, `evaluation/predictions.csv`, `evaluation/ground_truth.csv`, `evaluation/metadata.json`, `evaluation/spectrogram_*.png`, `evaluation/error_analysis.png`.

Output (map mode): `evaluation/metrics.json`, `evaluation/map_comparison_t*.png`, `evaluation/spatial_rmse_map.png`, `evaluation/per_frequency_rmse.png`.

### `inference.py` — Predict on new CSV data

```bash
python3 training/ConvLSTM/inference.py --checkpoint CHECKPOINT --input CSV [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` checkpoint (required) |
| `--input` | — | Input CSV with same format as training data (required) |
| `--output` | `predictions.csv` | Output CSV path |
| `--t-in` | from checkpoint config | Input sequence length (must match training) |
| `--t-out` | from checkpoint config | Prediction horizon (must match training) |

Output: CSV with all predicted PSD frames flattened row-by-row plus a companion metadata JSON describing window/horizon layout.

### `dataset.py` — Data loading and preprocessing (library)

Imported by `train.py` and `evaluate.py`. Key functions:

| Function | Returns | Description |
|----------|---------|-------------|
| `create_datasets(csv_path, n_nodes, n_bins, ...)` | `(train_ds, val_ds, test_ds, stats)` | Loads CSV (CSV mode), normalizes (z-score), creates sliding windows, splits chronologically or randomly |
| `create_interpolated_map_datasets(map_path, map_key, ...)` | `(train_ds, val_ds, test_ds, stats)` | Loads `.npz` map (map mode), normalizes per frequency, creates sliding windows, splits |
| `SpectrumDataset(data_3d, t_in, t_out, indices)` | PyTorch `Dataset` | Returns `(X, y)` tuples of shape `(T_in, 1, H, W)` and `(T_out, 1, H, W)` (CSV mode) |
| `InterpolatedMapDataset(data_4d, t_in, t_out, indices)` | PyTorch `Dataset` | Returns `(X, y)` of shape `(T_in, F, H, W)` and `(T_out, F, H, W)` (map mode) |
| `load_csv(path)` | `ndarray (T, 750)` | Loads CSV via `numpy.loadtxt` |
| `load_map_npz(path, key)` | `ndarray (T, F, H, W)` | Loads `.npz`, transposes from `(T, H, W, F)` to `(T, F, H, W)` |
| `clean_nan_csv(data_3d, ...)` / `clean_nan_map(data_4d, ...)` | `(cleaned, stats)` | Trims only trailing all-NaN timesteps, then fills internal NaNs from local time neighbours followed by nearby frequency bins/channels |
| `denormalize(data, mean, std)` | `ndarray` | Reverses z-score normalization |

`stats` dict (`{"mean": ndarray, "std": ndarray}`) is saved alongside checkpoints and used by `evaluate.py` and `inference.py` for denormalization.

### `utils.py` — Metrics and helpers (library)

Imported by all training/evaluation scripts. Key functions:

| Function | Description |
|----------|-------------|
| `compute_metrics(pred, target)` | Returns `{"rmse", "mae", "r2"}` across all dimensions |
| `compute_metrics_per_horizon(pred, target)` | Per-timestep metrics (`rmse_t1`, `rmse_t2`, ...) |
| `compute_metrics_per_node(pred, target, names)` | Per-node metrics (`rmse_CC1`, `mae_LW1`, ...) |
| `save_checkpoint(path, model, optimizer, ...)` | Saves model weights, optimizer state, config, norm stats |
| `load_checkpoint(path, device)` | Loads a saved checkpoint |
| `get_device(device_str)` | Returns `torch.device` ("auto" → cuda if available) |
| `set_seed(seed)` | Seeds Python, NumPy, and PyTorch RNGs |

## File Structure

```
training/ConvLSTM/
├── README.md                # This file
├── config.yaml              # Configuration / hyperparameters
├── dataset.py               # SpectrumDataset, data loading, normalization, windowing
├── model.py                 # ConvLSTMCell, ConvLSTM, ConvLSTMPredictor
├── train.py                 # Training loop, logging, checkpointing
├── evaluate.py              # Evaluation on test set, metrics, visualizations
├── utils.py                 # Helpers: normalization, metrics, seeding, device setup
├── inference.py             # Predict on new data, convert to CSV/plots
└── requirements.txt         # Dependencies: torch, numpy, matplotlib, PyYAML, etc.
```

### Module Responsibilities

| File | Contents |
|------|----------|
| `dataset.py` | `SpectrumDataset` / `InterpolatedMapDataset` (torch `Dataset`s), CSV loading, `.npz` loading, z-score normalization, sliding windows, train/val/test splitting for both CSV and map modes |
| `model.py` | `ConvLSTMCell`, `ConvLSTM` (multi-layer, from reference), `ConvLSTMPredictor` (seq2seq encoder–decoder with optional channel projection) |
| `train.py` | Training loop, data format branching, teacher forcing, gradient clipping, LR scheduling, early stopping, checkpoint saving |
| `evaluate.py` | Test set evaluation, data format branching, RMSE/MAE/R² per horizon (+ per node in CSV mode, per frequency in map mode), spectrogram/map visualization, prediction CSV export |
| `utils.py` | Normalization statistics, metrics, seed setting, device detection, denormalization |
| `inference.py` | Load checkpoint + normalization stats, predict on arbitrary input, save predictions as CSV |
| `config.yaml` | All hyperparameters (see Configuration Reference) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

### State Flow Between Modules

```
config.yaml
    │
    ▼
dataset.py ──► train.py ──► model.pt
                              │
                              ▼
                         evaluate.py ──► metrics, plots, predictions.csv
                              │
                              ▼
                         inference.py ──► predictions on new data
```

---

## Configuration Reference

All hyperparameters are in `config.yaml`. Key settings:

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| Data | `format` | `csv` | Data format: `csv` (node-frequency) or `interpolated_map` (pre-interpolated grid) |
| Data | `dataset_path` | `training/data/merged_power_data_sub6GHz_avg_per_minute.csv` | Input CSV path (CSV mode) |
| Data | `n_bins_per_node` | 250 | Frequency bins per node (CSV mode) |
| Data | `n_nodes` | 3 | Number of sensor nodes (CSV mode) |
| Data | `node_names` | `["CC1","CC2","LW1"]` | Node labels for plots and per-node metrics (CSV mode) |
| Data | `map_path` | `""` | Path to .npz with pre-interpolated spatial map (map mode) |
| Data | `map_key` | `map_db` | Key inside the .npz containing the map array (map mode) |
| Data | `n_freq_bins` | 200 | Number of frequency channels in the map (map mode; becomes ConvLSTM input channels) |
| Data | `grid_height` | 50 | Spatial grid height / rows (map mode) |
| Data | `grid_width` | 50 | Spatial grid width / columns (map mode) |
| Preprocessing | `normalization` | `zscore` | Normalization method (`zscore`, `minmax`, or `none`) |
| Preprocessing | `fit_on_train_only` | true | Compute normalization stats on training set only (true) or full dataset (false) |
| Preprocessing | `imputation.impute` | true | In CSV and map modes, trim trailing all-NaN timesteps and fill internal NaNs; if false, leave NaNs in place and warn |
| Preprocessing | `imputation.window_steps` | 2 | Radius for local time-neighbour fill and frequency-neighbour fallback |
| Windowing | `input_sequence_length` | 12 | Past minutes (T_in) |
| Windowing | `prediction_horizon` | 6 | Future minutes (T_out) |
| Windowing | `stride` | 1 | Window stride |
| Split | `train_ratio` | 0.8 | Training set fraction |
| Split | `val_ratio` | 0.1 | Validation set fraction (set to 0 for train/test only) |
| Split | `chronological_split` | true | Chronological (true) or random (false) split |
| Model | `input_channels` | 1 | Input channel count (1 for CSV; n_freq_bins for map mode) |
| Model | `use_channel_projection` | false | Apply 1×1 Conv2d before encoder to reduce channel count (map mode) |
| Model | `channel_projection_dim` | 16 | Target channel count after projection (only if `use_channel_projection: true`) |
| Model | `hidden_channels` | [32, 64] | Encoder layer hidden sizes |
| Model | `kernel_size` | [[3,3], [1,1]] | Encoder kernel sizes |
| Model | `num_encoder_layers` | 2 | Number of encoder ConvLSTM layers |
| Model | `decoder_hidden_channels` | 32 | Decoder hidden size |
| Model | `decoder_kernel_size` | [1,1] | Decoder ConvLSTM kernel size (per paper §III-A) |
| Model | `dropout` | 0.3 | Dropout probability |
| Model | `use_batch_norm` | true | Use batch normalization |
| Model | `decoder_lstm_hidden` | 128 | Regular LSTM hidden size in decoder |
| Model | `fc_hidden_channels` | 0 | FC intermediate channels (0 = single 1×1 Conv2d; >0 enables 2-layer MLP) |
| Model | `fc_kernel_size` | [3,3] | FC intermediate kernel (only if `fc_hidden_channels > 0`) |
| Model | `cell_activation` | `relu` | ConvLSTM cell activation (g candidate and h = o·activation(c)). Options: `relu`, `tanh`, `sigmoid`, `gelu`, `leaky_relu`, `elu` |
| Model | `fc_intermediate_activation` | `relu` | Activation for the optional FC intermediate layer (only if `fc_hidden_channels > 0`). Same options |
| Training | `batch_size` | 32 | Mini-batch size |
| Training | `epochs` | 100 | Max training epochs |
| Training | `learning_rate` | 0.0002 | Initial learning rate |
| Training | `optimizer` | nadam | Optimizer choice (`adam` or `nadam`) |
| Training | `beta1` | 0.9 | Adam/NADAM beta1 |
| Training | `beta2` | 0.999 | Adam/NADAM beta2 |
| Training | `epsilon` | 1e-8 | Adam/NADAM epsilon |
| Training | `weight_decay` | 0.004 | L2 weight decay |
| Training | `lr_scheduler` | `reduce_on_plateau` | LR scheduler (`reduce_on_plateau` or `none`) |
| Training | `lr_patience` | 10 | ReduceLROnPlateau patience (epochs without improvement before halving LR) |
| Training | `early_stopping_patience` | 20 | Epochs without val loss improvement before stopping |
| Training | `gradient_clip_norm` | 5.0 | Max gradient norm for clipping (0 = disabled) |
| Training | `loss` | mse | Loss function (`mse` or `mae`) |
| Training | `teacher_forcing_ratio` | 1.0 | Teacher forcing probability (1.0 = always on, 0 = pure autoregressive) |
| Training | `noise_std` | 0.2 | Gaussian noise std added to training inputs (paper §III-B) |
| Training | `seed` | 42 | Random seed for reproducible training |
| Evaluation | `metrics` | `["rmse","mae","r2"]` | Metrics to report |
| Evaluation | `eval_horizons` | `[1, 3, 6]` | Specific future time steps for per-horizon reporting |
| Device | `device` | auto | `cuda`, `cpu`, or `auto` |

---

## About The Model

## 1. Purpose

The model performs **long-term spectrum prediction** using the processed AERPAW 5 CSV dataset. Given a window of past per-minute power spectral density (PSD) measurements from three fixed sensor nodes, it predicts the PSD values for multiple future time steps.

This is a **multi-input multi-output (multi-step) time series regression** problem:

```
⟨χ_{t-n}, ..., χ_{t-2}, χ_{t-1}⟩  ⟶  ⟨χ_t, χ_{t+1}, ..., χ_{t+m}⟩
```

where:
- `n` = input sequence length (past observations in minutes)
- `m` = prediction horizon (future minutes to forecast)
- `χ_t` = a 2D power spectrogram slice of shape `(3 nodes, 250 frequency bins)` at time `t`

The model captures **joint spatial-spectral-temporal dependencies**:
- **Spatial**: correlations across the three fixed nodes (CC1, CC2, LW1)
- **Spectral**: correlations across adjacent frequency bins within each node's 250-bin band
- **Temporal**: sequential dependencies across time

Target use case: enabling Dynamic Spectrum Access (DSA) by predicting spectrum availability minutes in advance.

---

## 2. Dataset and Input Format

### Raw CSV Format

The processed AERPAW 5 CSV has the following characteristics:

| Property | Value |
|----------|-------|
| Rows (time steps) | 6,839 |
| Columns (features) | 750 |
| Header | None (pure data) |
| Format | Comma-separated, 6 decimal places |
| Values | Power Spectral Density in **dBm** |
| Range | −137.78 to −105.57 dBm |
| Missing values | 0 (no NaN, no Inf) |

**Column layout (per row):**

| Columns | Node | Frequency range | SigMF raw bin offset |
|---------|------|-----------------|----------------------|
| 0–249   | CC1  | ~1347–1362 MHz  | 21000                |
| 250–499 | CC2  | ~2082–2097 MHz  | 33250                |
| 500–749 | LW1  | ~1737–1752 MHz  | 27500                |

These per-node offsets were discovered by reverse-engineering the TSS-LCD repository CSV against the raw SigMF data. Each node selects a different quiet L-band / lower S-band region near the thermal noise floor.

Each row corresponds to a 1-minute averaged UTC interval. Rows are in chronological order.

### CSV → Tensor Conversion

The conversion pipeline is:

1. **Load**: `numpy.loadtxt(csv_path, delimiter=',')` → array of shape `(6839, 750)`

2. **Reshape**: `(6839, 750)` → `(6839, 3, 250)`
   - Axis 0: time (minutes)
   - Axis 1: node (0=CC1, 1=CC2, 2=LW1)
   - Axis 2: frequency bin (0–249)

3. **Normalize**: Apply z-score normalization per frequency bin (feature-wise across the time dimension):
   ```
   χ_normalized = (χ − μ_bin) / σ_bin
   ```
   where `μ_bin` and `σ_bin` are computed from the **training set only** (to avoid data leakage).

4. **Window**: Create overlapping sliding windows of length `T_in + T_out`:
   - For each window starting at index `i`, extract:
     - `X = windows[i : i + T_in]` → input sequence
     - `Y = windows[i + T_in : i + T_in + T_out]` → target sequence
   - Stride defaults to 1 (sliding window).

### Expected Tensor Shape

After CSV → tensor conversion, the input to the ConvLSTM model has shape:

```
(B, T_in, C, H, W)
```

| Dimension | Symbol | Value | Meaning |
|-----------|--------|-------|---------|
| Batch size | B | 32 (configurable) | Number of independent sequences in a batch |
| Input sequence length | T_in | 12 (configurable) | Number of past minutes used as input |
| Channels | C | 1 | Single-channel power spectrogram slice |
| Height | H | 3 | Number of sensor nodes (spatial dimension) |
| Width | W | 250 | Number of frequency bins per node (spectral dimension) |

The 2D spatial structure `(H=3, W=250)` is treated as a **spatial-spectral map**, analogous to an image where:
- The 3 rows correspond to the 3 sensor nodes
- The 250 columns correspond to frequency bins
- The single channel value is the normalized PSD (dBm)

This is the key design choice that enables ConvLSTM's 2D convolutions to jointly learn:
- **Cross-node spatial patterns** (vertical convolution across the 3 nodes)
- **Cross-frequency spectral patterns** (horizontal convolution across the 250 bins)
- **Spatial-spectral correlations** (2D convolution kernels)

### Interpolated Map Format (Alternative)

When `data.format: interpolated_map`, the model reads a pre-generated `.npz` file instead of a CSV. This format represents spectrum data as a dense 3D spatial map — power values interpolated onto a regular 2D grid at each time step.

**Source**: These `.npz` files are produced by a separate interpolation script (e.g., inverse distance weighting) using raw sensor coordinates.

**Shape in `.npz` file:**

| Axis | Label | Description |
|------|-------|-------------|
| 0 | T | Time steps (minutes) |
| 1 | H | Grid height (e.g. 50) |
| 2 | W | Grid width (e.g. 50) |
| 3 | F | Frequency channels (e.g. 200) |

On load, the array is transposed to `(T, F, H, W)` so frequency becomes the channel dimension — each frequency channel is treated as a separate input channel to the ConvLSTM.

**Map → Tensor Conversion:**

1. **Load**: `load_map_npz(path, key)` — loads `.npz`, extracts the key (default `map_db`), transposes `(T, H, W, F)` → `(T, F, H, W)`.
2. **Handle Missing Values**: `clean_nan_map()` optionally cleans NaNs before normalization and windowing:
   - If `preprocessing.imputation.impute: true`, trailing timesteps that are entirely NaN are dropped.
   - Internal missing `(t, f, h, w)` values are first filled from valid neighbours within the local time window `[t-window_steps, t-1] ∪ [t+1, t+window_steps]`.
   - If temporal neighbours are unavailable, the loader falls back to nearby frequency channels at the same timestep and spatial location.
   - Internal chronology is preserved because non-trailing timesteps are never removed.
   - If `preprocessing.imputation.impute: false`, NaNs are left in place and a warning is printed.
3. **Normalize**: Per-frequency z-score normalized across the time dimension, using stats `(F, 1, 1)` (broadcast over H×W).
4. **Window**: Identical sliding-window logic as CSV mode, producing `X: (T_in, F, H, W)` and `y: (T_out, F, H, W)`.

**Expected Tensor Shape (Map Mode):**

```
(B, T_in, F, H, W)
```

| Dimension | Symbol | Value | Meaning |
|-----------|--------|-------|---------|
| Batch size | B | 32 | Number of independent sequences |
| Input sequence length | T_in | 12 | Past minutes |
| Frequency channels | F | 200 | Frequency bins (input channels) |
| Grid height | H | 50 | Spatial rows |
| Grid width | W | 50 | Spatial columns |

**Key difference from CSV mode:** The channel dimension holds frequency, and the spatial dimensions (H, W) represent a real 2D spatial grid rather than synthetic rows-of-nodes. Optionally, a `1×1` Conv2d channel projection (configurable via `use_channel_projection` / `channel_projection_dim`) can reduce F before the encoder to lower memory usage.

---

## 3. Model Architecture

The architecture follows the paper's **sequence-to-sequence** design with an encoder–decoder structure.

### 3.1 Overview

Shapes shown for **CSV mode** (`C=1, H=n_nodes=3, W=n_bins=250`). In **map mode**, `C=n_freq_bins` (e.g. 200), `H=grid_height` (e.g. 50), `W=grid_width` (e.g. 50), and an optional `1×1` Conv2d channel projection may precede the encoder.

```
Input: (B, T_in, C, H, W)
        │
        ▼
┌─────────────────────────────────────────────┐
│  [Optional] Channel Projection              │
│  1×1 Conv2d(C → proj_dim) — only when       │
│  use_channel_projection=true (map mode)      │
│        ↓ (B, T_in, proj_dim, H, W)          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│              ENCODER                        │
│  ┌──────────────────────────────────┐       │
│  │ ConvLSTM Layer 1                 │       │
│  │  input_dim=C (or proj_dim)       │       │
│  │  hidden=32, kernel=(3,3)         │       │
│  │  activation=ReLU                 │       │
│  │     ↓ (B, T_in, 32, H, W)       │       │
│  │ ConvLSTM Layer 2                 │       │
│  │  input_dim=32, hidden=64         │       │
│  │  kernel=(1,1), padding=0         │       │
│  │  activation=ReLU                 │       │
│  │     ↓ (B, T_in, 64, H, W)       │       │
│  └──────────────────────────────────┘       │
│  Last states: (h_enc, c_enc)                │
│  each (B, 64, H, W)                         │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│         DECODER                             │
│  ┌──────────────────────────────────┐       │
│  │ LSTM (regular)                   │       │
│  │  Flatten encoder states          │       │
│  │  → (B, 64*H*W)                  │       │
│  │  LSTM(hidden=128)                │       │
│  │  → (B, 128)                      │       │
│  │  Unflatten → (B, 32, H, W)      │       │
│  └──────────────────────────────────┘       │
│        │                                     │
│        ▼                                     │
│  ┌──────────────────────────────────┐       │
│  │ ConvLSTM Layer 3                 │       │
│  │  input_dim=C (or proj_dim)       │       │
│  │  hidden=32, kernel=(1,1)         │       │
│  │  activation=ReLU                 │       │
│  │  Dropout(p=0.3)                  │       │
│  │  BatchNorm(32)                   │       │
│  │     ↓ (B, T_out, 32, H, W)      │       │
│  └──────────────────────────────────┘       │
│        │                                     │
│        ▼                                     │
│  ┌──────────────────────────────────┐       │
│  │ FC (Dense) Output Layer          │       │
│  │  Conv2d(32 → C, k=1)            │       │
│  │     ↓ (B, T_out, C, H, W)       │       │
│  └──────────────────────────────────┘       │
└─────────────────────────────────────────────┘
        │
        ▼
Output: (B, T_out, C, H, W)
        Squeeze channels → (B, T_out, H, W) in CSV mode
```

### 3.2 Layer-by-Layer Specification

#### Encoder — ConvLSTM Layer 1

| Parameter | Value |
|-----------|-------|
| Type | `ConvLSTMCell` (2D convolutional LSTM cell) |
| Input channels | C (1 for CSV mode, n_freq_bins for map mode, or proj_dim if channel projection enabled) |
| Hidden channels | 32 |
| Kernel size | `(3, 3)` |
| Padding | `(1, 1)` (same convolution, preserves H×W) |
| Bias | True |
| Activation (gates) | Sigmoid |
| Activation (cell candidate g) | Configurable (`cell_activation`; default ReLU per paper §III-A) |
| Activation (output H) | Configurable (`cell_activation`; default ReLU per paper §III-A) |
| Output shape | `(B, T_in, 32, H, W)` |

**Role**: Learns low-level spatial-spectral features. The `3×3` kernel captures local correlations between adjacent spatial positions and adjacent spectral bins simultaneously. Padding of 1 preserves the `H×W` spatial dimensions.

#### Encoder — ConvLSTM Layer 2

| Parameter | Value |
|-----------|-------|
| Type | `ConvLSTMCell` (2D convolutional LSTM cell) |
| Input channels | 32 (from Layer 1) |
| Hidden channels | 64 |
| Kernel size | `(1, 1)` |
| Padding | `(0, 0)` |
| Bias | True |
| Output shape | `(B, T_in, 64, H, W)` |

**Role**: Compresses the learned features into a higher-dimensional latent space. The `1×1` kernel performs pointwise convolution — it mixes information across channels at each spatial location without mixing adjacent spatial positions. This acts as a learned feature aggregation layer.

#### State Transfer (Encoder → Decoder) — Implementation Choice

> **Note:** The paper states there is *"an LSTM hidden layer that is used to capture memory and hidden states from the encoder output"* but does not specify its exact dimensionality or connectivity. The design below is our interpretation.

The encoder's final hidden and cell states from ConvLSTM Layer 2 (`h_enc`, `c_enc` — each `(B, 64, H, W)`) are fed into a regular LSTM layer at the decoder:

1. **Flatten**: Each state tensor is flattened from `(B, 64, H, W)` to `(B, 64 × H × W)`.
2. **LSTM**: The flattened state passes through a regular LSTM cell with hidden size 128, producing a context vector `(B, 128)`.
3. **Unflatten → ConvLSTM init**: The context vector is projected back to a 4D tensor:
   - `Linear(128 → 32 × H × W)` → `(B, 32, H, W)`
   - This projected tensor initializes the decoder ConvLSTM's hidden state `h_dec_0` and cell state `c_dec_0`.

This LSTM layer acts as the memory-capture mechanism referenced in the paper: it compresses the encoder's spatial-spectral representation into a compact code and initializes the decoder's generative process. Dimensionalities are empirical; tune as needed.

#### Decoder — ConvLSTM Layer 3

| Parameter | Value |
|-----------|-------|
| Type | `ConvLSTMCell` (2D convolutional LSTM cell) |
| Input channels | C (1 for CSV mode, n_freq_bins for map mode, or proj_dim if projection enabled) |
| Hidden channels | 32 |
| Kernel size | `(1, 1)` |
| Padding | `(0, 0)` |
| Bias | True |
| Activation (gates) | Sigmoid |
| Activation (cell candidate g) | Configurable (`cell_activation`; default ReLU per paper §III-A) |
| Activation (output H) | Configurable (`cell_activation`; default ReLU per paper §III-A) |
| Dropout | `p = 0.3` (after hidden state, before output) |
| Batch norm | Layer-wise batch normalization on hidden states |
| Output shape | `(B, T_out, 32, H, W)` |

**Role**: Generates the output sequence step by step. Initial hidden and cell states `(B, 32, H, W)` come from the LSTM projection of the encoder's final states. At each decoder time step, the input is the previous predicted frame (or ground-truth with teacher forcing). Because the kernel is `1×1`, each spatial location is processed independently — the decoder ConvLSTM learns temporal dynamics per spatial location without mixing positions.

#### Decoder — Fully Connected (Dense) Output Layer

This implements the "fully connected (dense) layer" from the paper §III-A. The default is a single `1×1` convolution (equivalent to a per-location dense layer):

| Layer | Type | Input → Output | Kernel | Padding |
|-------|------|----------------|--------|---------|
| FC | `Conv2d` | 32 → C | (1, 1) | (0, 0) |

Input: `(B, 32, H, W)` — single time step of decoder ConvLSTM hidden state.

Output: `(B, C, H, W)` — squeezed to `(B, H, W)` in CSV mode.

**Role**: Linearly projects the 32-channel hidden representation to the output channels (C=1 for CSV mode, C=n_freq_bins for map mode). The `1×1` kernel ensures each spatial location is projected independently, exactly matching a fully-connected layer applied per position.

**Optional alternative** (not in the paper): Replace the single `1×1` Conv2d with a small two-layer MLP convolution to add non-linearity:

| Layer | Type | Input → Output | Kernel | Padding |
|-------|------|----------------|--------|---------|
| FC1 | `Conv2d` | 32 → 16 | (3, 3) | (1, 1) |
| Activation | Configurable (`fc_intermediate_activation`; default ReLU) | — | — | — |
| FC2 | `Conv2d` | 16 → 1 | (1, 1) | (0, 0) |

This version incorporates local spatial-spectral context (via the 3×3 kernel) before the final projection. Enabled by setting `fc_hidden_channels > 0` in config.yaml.

### 3.3 ConvLSTM Equations

Each ConvLSTMCell follows the paper-style formulation with peephole connections from the previous cell state into the gates, and with configurable cell candidate and output activations (`cell_activation`, default ReLU per paper §III-A):

```
i_t = σ(W_xi ∗ X_t + W_hi ∗ H_{t-1} + W_ci ⊙ C_{t-1} + b_i)
f_t = σ(W_xf ∗ X_t + W_hf ∗ H_{t-1} + W_cf ⊙ C_{t-1} + b_f)
o_t = σ(W_xo ∗ X_t + W_ho ∗ H_{t-1} + W_co ⊙ C_{t-1} + b_o)
g_t = activation(W_xg ∗ X_t + W_hg ∗ H_{t-1} + b_g)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
H_t = o_t ⊙ activation(C_t)
```

where:
- `∗` = 2D convolution
- `⊙` = Hadamard (element-wise) product
- `σ` = sigmoid gate activation
- `activation` = configurable function (`cell_activation`; default ReLU per paper §III-A)
- `W_ci, W_cf, W_co` = learned peephole weights applied element-wise to `C_{t-1}`
- `i, f, o, g` = input gate, forget gate, output gate, cell candidate
- `C, H` = cell state, hidden state

### 3.4 Prediction Target

The model predicts **the next `T_out` consecutive minutes** of PSD values across all spatial and frequency dimensions:

- **CSV mode**: `(H=n_nodes, W=n_bins)` — e.g. single-frame `(3 × 250)`, multi-frame `(T_out × 3 × 250)`
- **Map mode**: `(F=n_freq_bins, H=grid_h, W=grid_w)` — e.g. single-frame `(200 × 50 × 50)`, multi-frame `(T_out × 200 × 50 × 50)`

The output is in **normalized (z-score) space**. Predictions must be un-normalized for interpretation:
```
value = prediction × σ + μ
```

---

## 4. Output Format

### Raw Output Tensor Shape

```
(B, T_out, C, H, W)
```

| Dimension | Meaning |
|-----------|---------|
| B | Batch size |
| T_out | Number of predicted future minutes |
| C | Output channels (1 in CSV mode, n_freq_bins in map mode) |
| H | Spatial height (n_nodes in CSV mode, grid_height in map mode) |
| W | Spatial width (n_bins in CSV mode, grid_width in map mode) |

### Meaning of Each Output Element

Each output element represents the **predicted normalized power spectral density** for a batch sample `b`, future time step `t`, at a location defined by `(channel, row, column)`.

### Post-Processing: Normalization → Physical Units

```python
pred_physical = pred_normalized * stats_std + stats_mean
```

Denormalization shapes depend on the data format:
- **CSV mode**: `stats_mean/std` have shape `(n_nodes, n_bins)` — broadcast against `(H, W)`
- **Map mode**: `stats_mean/std` have shape `(n_freq_bins, 1, 1)` — broadcast over `H×W`

### CSV Output Format (CSV Mode Only)

Predicted data can be written as CSV with the same format as the input:
- No header
- 6 decimal places
- 750 columns per row (CC1[0–249], CC2[250–499], LW1[500–749])
- One row per predicted minute

### Visualization

**CSV mode**: Per-node spectrograms — select a specific node `pred[:, :, node_idx, :]`, transpose to frequency-vs-time, `imshow`.

**Map mode**: Spatial map comparisons — side-by-side ground truth vs prediction for selected time steps, spatial RMSE heatmap, and per-frequency RMSE line plot.

---

## 5. Training Pipeline

### Data Loading Process

The data loading pipeline branches based on `data.format` in config:

- **CSV mode** (`format: csv`): `create_datasets(...)` → `SpectrumDataset`
- **Map mode** (`format: interpolated_map`): `create_interpolated_map_datasets(...)` → `InterpolatedMapDataset`

**CSV mode** (`SpectrumDataset`):
```python
# X shape: (num_samples, T_in, 1, n_nodes, n_bins)
# y shape: (num_samples, T_out, 1, n_nodes, n_bins)
```
1. Loads CSV as `numpy.ndarray` → reshapes to `(T, n_nodes, n_bins)`
2. Computes normalization statistics (per-node, per-bin mean/std)
3. Normalizes the data
4. Generates sliding windows → `(X, y)` tuples

**Map mode** (`InterpolatedMapDataset`):
```python
# X shape: (num_samples, T_in, n_freq_bins, grid_h, grid_w)
# y shape: (num_samples, T_out, n_freq_bins, grid_h, grid_w)
```
1. Loads `.npz` via `load_map_npz(path, key)` → transposes to `(T, F, H, W)`
2. Computes per-frequency z-score statistics (shape `(F, 1, 1)`)
3. Normalizes the data (broadcast over H×W)
4. Generates sliding windows → `(X, y)` tuples

### Normalization

**Method**: Z-score (standard score) normalization, applied **per frequency bin** across the time dimension.

- **CSV mode**: For each node `n` and bin `b`: `μ_{n,b}`, `σ_{n,b}` computed across time → shape `(n_nodes, n_bins)`
- **Map mode**: For each frequency channel `f`: `μ_f`, `σ_f` computed across time × H × W → shape `(F, 1, 1)` (broadcast over H×W)

```
X_normalized = (X − μ) / σ
```

- Statistics are computed **only from the training set** to prevent data leakage (controlled by `fit_on_train_only` in config).
- The same `μ` and `σ` are applied to validation and test sets.
- Saved alongside the model checkpoint for inference-time denormalization.

### Train / Validation / Test Split

**Ratios**: Configurable in `config.yaml` under `split`. Default is 80/10/10, but `val_ratio` can be set to 0 for train/test only. Ratios can sum to less than 1.0 (remaining data unused) or to 1.0 (all data used).

**Strategy**: Default is **chronological split** (recommended for time series). If `chronological_split: false`, splits are random.

### Window Generation

Both modes use the same sliding-window logic: extract overlapping windows of length `T_in + T_out` with configurable stride. Each window produces `(X, y)` of shapes:

- **CSV mode**: `X: (T_in, 1, n_nodes, n_bins)`, `y: (T_out, 1, n_nodes, n_bins)`
- **Map mode**: `X: (T_in, F, grid_h, grid_w)`, `y: (T_out, F, grid_h, grid_w)`

### Loss Function

**Mean Squared Error (MSE)** (default):

```
L = (1 / (B × T_out × C × H × W)) × Σ(predicted − target)²
```

MSE penalizes larger errors more heavily, appropriate for PSD regression. MAE can be used as an alternative by setting `loss: mae` in config.

### Optimizer

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | NADAM (default) | Paper §III-B |
| Learning rate | 0.0002 | Paper §III-B |
| β₁ | 0.9 | Paper §III-B |
| β₂ | 0.999 | Paper §III-B |
| ε | 1×10⁻⁸ | Paper §III-B |
| Weight decay (λ) | 0.004 | Paper §III-B |
| Gradient clip norm | 5.0 | Common practice |

### Batch Size

- Default: **32**
- Reduce to 16 or 8 for smaller GPUs.

### Epochs

- Default: **100**
- Early stopping with patience of 20 epochs based on validation loss
- Learning rate reduction on plateau (factor 0.5, patience 10)

### Teacher Forcing (Implementation Choice, Not in Paper)

The decoder receives ground-truth frames as input during training with probability `teacher_forcing_ratio` (default 1.0 — always on). This speeds up convergence by providing real data at every decoder step. Set to 0 for pure autoregressive decoding, or 0.5 for mixed scheduling.

### Regularization

| Technique | Value | Paper reference |
|-----------|-------|-----------------|
| Dropout (encoder & decoder) | p = 0.3 | §III-B |
| Batch normalization | After ConvLSTM layers | §III-B |
| Gaussian input noise | σ = 0.2 | §III-B |
| Weight decay (L2) | λ = 0.004 | §III-B (NADAM parameter) |

### Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| RMSE | √(mean((ŷ − y)²)) | Primary metric, same unit as dBm |
| MAE | mean(|ŷ − y|) | Robust to outliers |
| R² | 1 − (SS_res / SS_tot) | Proportion of variance explained |

Metrics are computed:
- **Per horizon**: RMSE at specified horizons (eval_horizons), measuring error accumulation
- **Per node** (CSV mode only): RMSE for each sensor node separately
- **Per frequency** (map mode only): RMSE for each frequency channel separately
- **Spatial RMSE map** (map mode only): 2D heatmap of RMSE per spatial location
- **Overall**: Average across all dimensions

---

## 6. Assumptions and Design Decisions

### Assumptions

1. **In CSV mode** (`format: csv`), the input has `n_nodes × n_bins_per_node` columns (default 750) ordered as per `node_names` in config.
2. **In CSV mode**, the file has no header and contains only comma-separated dBm values.
3. **Rows (time axis) are in strict chronological order** with no gaps (both modes).
4. **Normalization statistics are computed per frequency bin** across the time dimension, not globally (both modes).
5. **The train/val/test split is chronological**, not random, because this is time-series data.
6. **In CSV mode**, the per-node-offset CSV (`merged_power_data_sub6GHz_avg_per_minute.csv`) is the target, produced by `training/build_training_csv.py` using per-node raw SigMF bin offsets (CC1=21000, CC2=33250, LW1=27500). Each node's 250 bins are from a different quiet L-band / lower S-band region, not a shared 87–336 MHz band.
7. **In map mode** (`format: interpolated_map`), the `.npz` file has shape `(T, H, W, F)` with the specified key (default `map_db`). The spatial grid is a regular 2D mesh (e.g. 50×50) with each cell representing an interpolated PSD value at that spatial location.

### Design Decisions

1. **2D ConvLSTM with (H, W) spatial map**: In CSV mode, nodes are treated as height and frequency bins as width (default 3 × 250). In map mode, the grid is a regular spatial mesh (e.g. 50 × 50) with frequency as the channel dimension. Both allow 2D kernels to learn spatial-spectral patterns.

2. **Input channels vary by mode**: CSV mode uses C=1 (scalar PSD per location). Map mode uses C=n_freq_bins (one channel per frequency), with an optional `1×1` channel projection to reduce memory.

3. **Seq2seq with teacher forcing**: The decoder generates outputs autoregressively. Teacher forcing (default 1.0, always on) is our addition; set lower in config.yaml for mixed or pure autoregressive decoding.

4. **No spatial interpolation (CSV mode) / Pre-interpolated map (map mode)**: The paper used IDW interpolation across 1600 grid points from 5 sensors. Our AERPAW CSV dataset has 3 fixed nodes without a regular spatial grid — we preserve the raw per-node data. The map mode accepts pre-interpolated `.npz` files, enabling the model to work on dense spatial grids when interpolation has been applied offline.

5. **Per-bin z-score normalization**: Spectrum data has different power levels across frequency bands. Normalizing per bin ensures each frequency contributes equally to the loss. In map mode, statistics are `(F, 1, 1)` and broadcast over the spatial grid.

---

## 7. Deviations from the Original Paper

| Aspect | Paper (2019) | Our Reconstruction | Reason |
|--------|-------------|-------------------|--------|
| Dataset | Electrosense (5 sensors, 450–520 MHz) | AERPAW (3 nodes, per-node L-band offsets) or interpolated .npz map | Our target dataset; map mode enables dense spatial grids |
| Spatial dimension | 40×40 IDW grid → 1600 locations | 3 fixed nodes (CSV) or configurable grid (map) | AERPAW has sparse, fixed nodes; map mode matches paper's dense grid |
| Input time steps | 120 (6 hours × 3 min resolution) | Configurable (e.g. 12) | Set in config; 1-min vs 3-min resolution |
| Prediction horizon | 50 steps (150 min) | Configurable (e.g. 6) | Set in config.yaml |
| Encoder layers | 2 ConvLSTM | 2 ConvLSTM | Same |
| Decoder layers | LSTM + ConvLSTM + FC | LSTM + ConvLSTM + FC (1×1 Conv2d; 2-layer MLP optional) | Same structure; FC details are our interpretation |
| Input channels | 1 (power per grid point) | 1 (CSV) or n_freq_bins (map) | Map mode uses frequency as channel dimension |
| Activation | ReLU (output, per §III-A) | Configurable (`cell_activation`; default ReLU) | Config-driven; matches paper at default |
| Optimizer | NADAM | Adam | NADAM not in standard PyTorch |
| Framework | R + TensorFlow | Python + PyTorch | Our stack |

---

## References

1. **Original paper:** B. S. Shawel, D. H. Woldegebreal, S. Pollin, "Convolutional LSTM-based Long-Term Spectrum Prediction for Dynamic Spectrum Access," EUSIPCO 2019. ([link](https://new.eurasip.org/Proceedings/Eusipco/eusipco2019/Proceedings/papers/1570533330.pdf))
2. **ConvLSTM paper:** X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, W.-C. Woo, "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting," NIPS 2015.
3. **Reference implementation:** [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch) — base for our `ConvLSTMCell` / `ConvLSTM` modules.
4. **AERPAW dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022). DOI: [10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn).
