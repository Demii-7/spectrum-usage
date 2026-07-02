# DSwinLSTM-I Spectrum Prediction — Reconstructed Model

> **Based on:** *Robust Imputation SwinLSTM for Spectrum Map Prediction of Incomplete Data* (target architecture)
>
> **Supporting code reference (vanilla SwinLSTM only, no imputation):** https://github.com/SongTang-x/SwinLSTM — ICCV 2023 paper implementation
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

**This is a paper reconstruction, not a direct adaptation of the SwinLSTM repo.** The repo provides vanilla SwinLSTM building blocks (Swin Transformer blocks, SwinLSTM cell, patch embedding/merging/expanding, reconstruction layer). The imputation unit (SwinLSTM-I), encoder-decoder separation, and mask-aware training pipeline are implemented from scratch following the DSwinLSTM-I paper. The repo is a supporting reference for the Swin Transformer mechanics only.

> **Primary path:** prepared interpolated maps with hierarchical patch merging/expanding enabled. CSV pseudo-maps remain available as an AERPAW adaptation path.

---

## Quick Start

### Setup

```bash
cd /home/cc/spectrum-usage
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib pyyaml scikit-learn tqdm timm einops
```

### Train the Model

```bash
# Default config (full 3-node mode, T_in=10, T_out=10, 80/10/10 split)
python3 training/DSwinLSTM-I/train.py

# CC2-only smoke test
python3 training/DSwinLSTM-I/train.py --config training/DSwinLSTM-I/smoke_test/config.yaml
```

Training creates `training/DSwinLSTM-I/checkpoints/` with `best_model.pt`, `last_model.pt`, and `normalization_stats.pt`.

> **Note:** Checkpoints contain model weights, optimizer state, and normalization stats. They exceed GitHub's 100 MB file limit and are gitignored. To use a trained model on another machine, either re-train there or manually copy the `checkpoints/` directory.

### Evaluate

```bash
python3 training/DSwinLSTM-I/evaluate.py \
    --checkpoint training/DSwinLSTM-I/checkpoints/best_model.pt
```

Output: RMSE/MAE/R²/NRMSE(dB), timing metrics, full prediction exports, plots, and `metadata.json`.

### Run Inference on New Data

```bash
python3 training/DSwinLSTM-I/inference.py \
    --checkpoint training/DSwinLSTM-I/checkpoints/best_model.pt \
    --input /path/to/new_measurements.csv \
    --output predictions.csv
```

---

## Scripts Reference

### `train.py` — Train a new model

```bash
python3 training/DSwinLSTM-I/train.py [--config CONFIG] [--cc2-only]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration file |
| `--cc2-only` | false | Enable CC2-only smoke mode (overrides config) |

Output: `checkpoints/best_model.pt`, `checkpoints/last_model.pt`, `checkpoints/normalization_stats.pt`.

### `evaluate.py` — Evaluate a trained model

```bash
python3 training/DSwinLSTM-I/evaluate.py --checkpoint CHECKPOINT [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` checkpoint from training (required) |
| `--config` | from checkpoint | Path to config (overrides checkpoint's embedded config) |
| `--output` | `evaluation/` | Output directory for metrics, plots, and CSVs |
| `--cc2-only` | false | Enable CC2-only smoke mode |

Output: `evaluation/metrics.json`, `evaluation/predictions.csv`, `evaluation/ground_truth.csv`, `evaluation/predictions_dbm.csv`, `evaluation/ground_truth_dbm.csv`, `evaluation/spectrogram_*.png`, `evaluation/error_analysis.png`.

### `inference.py` — Predict on new CSV or interpolated-map data

```bash
python3 training/DSwinLSTM-I/inference.py --checkpoint CHECKPOINT --input CSV [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` checkpoint (required) |
| `--input` | — | Input CSV or NPZ map matching the training mode (required) |
| `--output` | `predictions.csv` / `predictions.npz` | Output path |
| `--config` | from checkpoint | Config YAML (overrides checkpoint) |

Output: full prediction export plus companion metadata JSON.

### `dataset.py` — Data loading and preprocessing (library)

Imported by `train.py`, `evaluate.py`, and `inference.py`. Key functions:

| Function | Returns | Description |
|----------|---------|-------------|
| `load_and_split(config, cc2_only)` | `(train, val, test, stats, node_names)` | Loads CSV pseudo-maps or prepared interpolated maps, cleans NaNs in map mode, splits chronologically, normalizes |
| `SpectrumMapDataset(data, config, split)` | PyTorch `Dataset` | Returns `(X_masked, mask, Y)` tuples of shape `(T_in, H, W, F)`, `(T_in, H, W, F)`, `(T_out, H, W, F)` |
| `random_mask(shape, missing_rate)` | `Tensor (T, H, W, F)` | Element-wise Bernoulli mask |
| `block_mask(shape, missing_rate)` | `Tensor (T, H, W, F)` | Contiguous block mask |
| `frequency_mask(shape, missing_rate)` | `Tensor (T, H, W, F)` | Masks entire frequency bins |
| `node_mask(shape, missing_rate, n_nodes)` | `Tensor (T, H, W, F)` | Masks entire sensor nodes |
| `node_column_slice(node_names, bins_per_node)` | `list[int]` | Maps node names to CSV column indices |

`stats` dict (for example `{"method", "dmin", "dmax", "range", "grid_height", "grid_width", "n_freq_bins"}`) is saved alongside checkpoints and used by `evaluate.py` and `inference.py` for denormalization and output reconstruction.

### `utils.py` — Metrics and helpers (library)

Imported by all training/evaluation scripts. Key functions:

| Function | Description |
|----------|-------------|
| `load_config(path)` | Loads and returns YAML config as dict |
| `get_device(device_str)` | Returns `torch.device` ("auto" → cuda if available) |
| `set_seed(seed)` | Seeds Python, NumPy, and PyTorch RNGs |
| `normalize_minmax(data, dmin, dmax, range)` | MinMax normalization to `[-1, 1]` |
| `denormalize(data, stats)` | Reverses MinMax normalization to dBm |
| `compute_metrics(pred, target)` | Returns `{"rmse", "mae", "r2", "nrmse_db"}` |
| `compute_metrics_per_horizon(pred, target)` | Per-timestep metrics (`rmse_t1`, `rmse_t2`, ...) |
| `compute_metrics_per_node(pred, target, names)` | Per-node metrics (`rmse_CC1`, `mae_LW1`, ...) |
| `save_checkpoint(path, model, optimizer, ...)` | Saves model weights, optimizer state, config, norm stats |
| `load_checkpoint(path, device)` | Loads a saved checkpoint |
| `init_logger(log_dir, name)` | Sets up file+console logger for training |
| `plot_spectrogram_comparison(gt, pred, node_idx, name, path)` | Side-by-side spectrogram figure |
| `plot_error_analysis(errors, node_names, path)` | Per-node error heatmap |

---

## File Structure

```
training/DSwinLSTM-I/
├── README.md                # This file
├── config.yaml              # Full configuration / hyperparameters
├── dataset.py               # AERPAWDataset, mask generators, CSV loading, normalization, windowing
├── model.py                 # SwinLSTMCell, SwinLSTMCellI, PatchEmbed, MaskPool, Encoder, Decoder, Reconstruction, DSwinLSTM_I
├── train.py                 # Training loop, logging, checkpointing
├── evaluate.py              # Evaluation on test set, metrics, visualizations
├── utils.py                 # Helpers: config, normalization, metrics, seeding, device setup, plotting
├── inference.py             # Predict on new data, convert to CSV
├── checkpoints/             # Created during training (best_model.pt, last_model.pt, normalization_stats.pt)
├── evaluation/              # Created during evaluation (metrics.json, predictions.csv, plots)
└── smoke_test/              # CC2-only smoke test config
    └── config.yaml
```

### Module Responsibilities

| File | Contents |
|------|----------|
| `dataset.py` | `AERPAWDataset` (torch `Dataset`), CSV loading, node selection, pseudo-map reshape, MinMax normalization, sliding windows, mask generators (random/block/frequency/node) |
| `model.py` | `SwinLSTMCell` (vanilla), `SwinLSTMCellI` (with imputation), `PatchEmbed`, `MaskPool`, `Encoder` (SwinLSTMCellI stack), `Decoder` (SwinLSTMCell stack), `Reconstruction` (exact-size linear), `DSwinLSTM_I` (full model) |
| `train.py` | Training loop, teacher forcing, gradient clipping, early stopping, checkpoint saving, timing logs |
| `evaluate.py` | Final test evaluation only, RMSE/MAE/R²/NRMSE(dB), timing, full prediction export |
| `utils.py` | Config loading, MinMax normalization, metrics, seed setting, device detection, denormalization, checkpointing, plotting |
| `inference.py` | Load checkpoint + normalization stats, predict on arbitrary input, save predictions as CSV |
| `config.yaml` | All hyperparameters (see Configuration Reference) |
| `README.md` | This file |

### State Flow Between Modules

```
config.yaml
    │
    ▼
dataset.py ──► train.py ──► model.pt (checkpoints/)
                              │
                              ▼
                         evaluate.py ──► metrics.json, predictions.csv, plots (evaluation/)
                              │
                              ▼
                         inference.py ──► predictions on new data
```

---

## Configuration Reference

All hyperparameters are in `config.yaml`. Key settings:

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| Data | `dataset_path` | `training/data/merged_power_data_sub6GHz_avg_per_minute.csv` | Input CSV path |
| Data | `n_nodes` | 3 | Number of sensor nodes for pseudo-map |
| Data | `bins_per_node` | 250 | Frequency bins per node |
| Data | `node_names` | `["CC1","CC2","LW1"]` | Node labels for plots and per-node metrics |
| Data | `selected_nodes` | `["CC1","CC2","LW1"]` | Subset of nodes to use |
| Data | `cc2_only_smoke_test` | false | CC2-only mode (H=1, W=250) |
| Windowing | `input_sequence_length` | 10 | Past minutes (T_in) |
| Windowing | `prediction_horizon` | 10 | Future minutes to predict (T_out) |
| Windowing | `train_stride` | 1 | Window stride for training |
| Windowing | `val_stride` | null (defaults to T_out) | Window stride for validation |
| Windowing | `test_stride` | null (defaults to T_out) | Window stride for testing |
| Split | `train_ratio` | 0.8 | Training set fraction |
| Split | `val_ratio` | 0.1 | Validation set fraction |
| Split | `chronological_split` | true | Chronological (true) or random (false) |
| Preprocessing | `normalization` | `minmax` | Normalization method |
| Preprocessing | `minmax_range` | `[-1, 1]` | Target range for MinMax |
| Preprocessing | `fit_on_train_only` | true | Compute stats on training split only |
| Preprocessing | `missing_rate` | 0.3 | Fraction of input masked as missing |
| Preprocessing | `missing_strategy` | `random` | Mask strategy (random/block/frequency/node) |
| Preprocessing | `mask_targets` | false | Never mask target Y |
| Model | `map_height` | 3 | Pseudo-map height (nodes) |
| Model | `map_width` | 250 | Pseudo-map width (frequency bins) |
| Model | `input_channels` | 1 | Input channel dimension (F) |
| Model | `patch_shape` | `[1, 2]` | Rectangular patch size (height, width) |
| Model | `embed_dim` | 128 | Embedding dimension after patch projection |
| Model | `encoder_units` | 2 | Number of SwinLSTM-I encoder cells |
| Model | `decoder_units` | 2 | Number of vanilla SwinLSTM decoder cells |
| Model | `swin_depths` | `[2, 6, 6, 2]` | Swin Transformer blocks per cell |
| Model | `num_heads` | `[4, 8, 8, 4]` | Attention heads per cell |
| Model | `window_size` | 4 | Swin attention window size |
| Model | `use_patch_merging` | false | Disabled (H=3 odd) |
| Model | `use_patch_expanding` | false | Disabled (paired with merging) |
| Model | `use_imputation_unit` | true | Enable SwinLSTM-I in encoder |
| Model | `output_activation` | tanh | Reconstruction activation (matches MinMax range) |
| Model | `decoder_feedback` | `hidden_state` | Decoder feedback mode |
| Model | `padding_mode` | `reflect` | Spatial padding for non-divisible dims |
| Training | `batch_size` | 4 | Mini-batch size |
| Training | `epochs` | 400 | Max training epochs |
| Training | `learning_rate` | 0.0001 | Initial learning rate |
| Training | `optimizer` | adam | Optimizer |
| Training | `loss` | mse | Loss function |
| Training | `early_stopping` | true | Enable early stopping |
| Training | `patience` | 30 | Early stopping patience |
| Training | `gradient_clip` | 5.0 | Gradient clipping max norm |
| Training | `seed` | 42 | Random seed |
| Evaluation | `metrics` | `["rmse","mae","r2","nrmse_db"]` | Metrics to report |
| Evaluation | `export_predictions` | true | Save CSV exports |
| Paths | `checkpoints_dir` | checkpoints | Checkpoint output directory |
| Paths | `evaluation_dir` | evaluation | Evaluation output directory |
| Device | `device` | auto | `cuda`, `cpu`, or `auto` |

---

## About The Model

## 1. Purpose

DSwinLSTM-I (Deep SwinLSTM with Imputation) performs **joint imputation and prediction** of future spectrum maps from incomplete or corrupted historical maps. Unlike a two-stage pipeline (impute missing data first, then predict), DSwinLSTM-I integrates missing-value estimation inside the recurrent cell.

Key properties:

- **Joint imputation-prediction.** Missing entries are inferred from prior hidden and cell states, not from an external imputation method.
- **Mask-aware.** A binary mask `M_t` indicates which entries of the input `P_t` are observed (1) and which are missing/corrupted (0). Only the input is masked; the target remains complete.
- **Swin Transformer backbone.** Swin Transformer blocks (shifted-window multi-head self-attention) replace the convolutional operations in ConvLSTM for capturing global spatial dependencies.
- **Encoder-decoder architecture.** The encoder uses **SwinLSTM-I units** (with imputation). The decoder uses **vanilla SwinLSTM units** (prediction only). This separation lets the encoder focus on reconstructing missing entries while the decoder focuses on forecasting.

This is a **multi-input multi-output (multi-step) time series regression** problem with missing data:

```
⟨χ_{t-n}, ..., χ_{t-1}⟩, ⟨M_{t-n}, ..., M_{t-1}⟩  ⟶  ⟨χ_t, χ_{t+1}, ..., χ_{t+m}⟩
```

where:
- `n` = input sequence length (past observations in minutes)
- `m` = prediction horizon (future minutes to forecast)
- `χ_t` = a 2D pseudo-spectrum-map of shape `(H nodes, W frequency bins, F=1 channel)` at time `t`

The model captures **joint spatial-spectral-temporal dependencies**:
- **Spatial**: correlations across the three fixed nodes (CC1, CC2, LW1)
- **Spectral**: correlations across adjacent frequency bins within each node's 250-bin band
- **Temporal**: sequential dependencies across time

Target use case: enabling Dynamic Spectrum Access (DSA) with incomplete or corrupted sensor data.

---

## 2. Dataset and Input Format

### Raw CSV Format

The processed AERPAW CSV has the following characteristics:

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

| Columns | Node | Frequency range |
|---------|------|-----------------|
| 0–249   | CC1  | ~1347–1362 MHz  |
| 250–499 | CC2  | ~2082–2097 MHz  |
| 500–749 | LW1  | ~1737–1752 MHz  |

Rows are 1-minute averaged UTC intervals in strict chronological order.

### CSV → Tensor Conversion

1. **Load**: `numpy.loadtxt(csv_path, delimiter=',')` → array of shape `(6839, 750)`
2. **Select nodes**: Filter columns by configured node names (default all 3 nodes) via `node_column_slice()`
3. **Reshape**: `(T, n_nodes * bins_per_node)` → `(T, n_nodes, bins_per_node, 1)` — pseudo-spectrum-map
   - Axis 0: time (minutes)
   - Axis 1: node (0=CC1, 1=CC2, 2=LW1)
   - Axis 2: frequency bin (0–249)
   - Axis 3: channel (F=1)
4. **Normalize**: Apply MinMax normalization to `[-1, 1]` across the full training split:
   ```
   χ_normalized = (χ − data_min) / (data_max − data_min) × 2 − 1
   ```
   Statistics are computed from the **training set only** (when `fit_on_train_only: true`).
5. **Window**: Create sliding windows of length `T_in + T_out`:
   - `X = windows[i : i + T_in]` → input sequence
   - `Y = windows[i + T_in : i + T_in + T_out]` → target sequence

### Mask Generation

For each training input window, a binary mask `M` is generated:

```
M_t[i,j] = 1 if observed, 0 if missing/corrupted
```

Strategies (configurable via `missing_strategy`):

- **random**: Each element independently masked with probability `missing_rate` (Bernoulli).
- **block**: Contiguous blocks masked along the frequency axis (more realistic for spectrum dropout).
- **frequency**: Entire frequency bins masked at random time steps (simulates band blocking).
- **node**: Entire sensor nodes masked at random time steps (simulates node failure).

**Mask is applied only to input X. Target Y is never masked.** The mask is passed alongside X to the model's imputation unit.

### Expected Tensor Shapes

```
Full mode:
    X: (B, T_in, H=3,  W=250, F=1)   — Incomplete input maps
    M: (B, T_in, H=3,  W=250, F=1)   — Binary mask
    Y: (B, T_out, H=3, W=250, F=1)   — Complete target maps

CC2 smoke mode:
    X: (B, T_in, H=1,  W=250, F=1)   — Incomplete input maps
    M: (B, T_in, H=1,  W=250, F=1)   — Binary mask
    Y: (B, T_out, H=1, W=250, F=1)   — Complete target maps
```

Internally, the model permutes to channel-first: `(B, T, F, H, W)`.

---

## 3. Model Architecture

### 3.1 Overview

Shapes shown for the primary interpolated-map path. CSV pseudo-map mode follows the same interface after conversion to `(T, H, W, F)`.

```
Input X: (B, T_in, F=1, H=3, W=250)
Mask M:  (B, T_in, F=1, H=3, W=250)
          │
          ▼
 ┌─────────────────────────────────────────────┐
 │              Patch Embedding                 │
 │  Conv2d(F → embed_dim, kernel=patch_shape,   │
 │         stride=patch_shape)                  │
 │  → LayerNorm                                 │
 │  Output: (B, L0, embed_dim)                    │
 └──────────────────────┬──────────────────────┘
          │
 ┌─────────────────────────────────────────────┐
 │              MaskPool                        │
 │  AvgPool2d(kernel=patch_shape)              │
 │  Downsamples mask from pixel to token space  │
 │  Output: (B, T_in, L0, 1) → expand to         │
 │          (B, T_in, L0, embed_dim)             │
 └──────────────────────┬──────────────────────┘
          │
 ┌─────────────────────────────────────────────┐
 │           Encoder (SwinLSTM-I units)          │
 │                                              │
 │  For each time step t:                       │
 │    1. Imputation unit:                       │
 │       P_hat = σ(W_p·c_{t-1} + U_p·h_{t-1}+b)│
 │       x_t = m_t ⊙ x_t + (1-m_t) ⊙ P_hat     │
 │    2. SwinLSTM update:                       │
 │       F_t = SwinTransformer(x_t, h_{t-1})    │
 │       gate = σ(F_t), cell = tanh(F_t)        │
 │       c_t = gate ⊙ (c_{t-1} + cell)          │
 │       h_t = gate ⊙ tanh(c_t)                 │
 │                                              │
 │  Unit 0: SwinLSTM-I at stage 0               │
 │  Patch Merging                               │
 │  Unit 1: SwinLSTM-I at stage 1               │
 └──────────────────────┬──────────────────────┘
          │  (final h_t, c_t)
          ▼
 ┌─────────────────────────────────────────────┐
 │          Decoder (vanilla SwinLSTM units)     │
 │                                              │
 │  Autoregressive prediction:                  │
 │  For each output step t:                     │
 │    h_t, c_t = SwinLSTM(h_{t-1}, c_{t-1})    │
 │    y_hat_t = Reconstruction(h_t)             │
 │    y_hat_t = tanh(y_hat_t)                   │
 │                                              │
 │  Unit 1: SwinLSTM at stage 1                 │
 │  Patch Expanding                             │
 │  Unit 0: SwinLSTM at stage 0                 │
 │  Feedback: pixel_feedback with teacher       │
 │  forcing in training, autoregressive eval    │
 └──────────────────────┬──────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────┐
 │          Reconstruction (exact-size)          │
 │  Linear(embed_dim → patch_h*patch_w*F)      │
 │  Reshape → (B, F, H, W)                     │
 │  tanh activation                             │
 └──────────────────────┬──────────────────────┘
          │
Output: (B, T_out, F=1, H=3, W=250)
```

### 3.2 Patch Embedding

Each input map is divided into non-overlapping patches of configurable shape (paper default `2×2`). For arbitrary map sizes, the model pads the spatial dimensions to a hierarchy-compatible size before patch embedding and crops predictions back to the original size after reconstruction.

```
Input:  (B, F, H, W)
→ Conv2d(F → embed_dim, kernel=(1,2), stride=(1,2))
→ Flatten spatial: (B, L=H/pH * W/pW, embed_dim)
→ LayerNorm
Output: (B, L=3×125=375, embed_dim=128)
```

The paper-aligned interpolated-map path uses square `2×2` patches with patch merging/expanding enabled. CSV pseudo-map mode may use the same padding-aware patching as an adaptation.

### 3.3 MaskPool

The binary mask at pixel resolution is downsampled to token resolution via average pooling:

```
Input:  mask (B, T_in, H=3, W=250, F=1)
→ Permute to (B, T_in, F, H, W), merge batch×time
→ AvgPool2d(kernel=patch_shape=(1,2), stride=patch_shape)
→ Reshape back: (B, T_in, H_p=3, W_p=125, F=1)
→ Flatten: (B, T_in, L=375, 1)
→ Expand to (B, T_in, L, embed_dim) for the imputation linear layers
```

### 3.4 Encoder: SwinLSTM-I Unit

Each encoder unit is a **SwinLSTM-I cell** that extends the vanilla SwinLSTM cell with an imputation mechanism:

#### Imputation Step

Missing entries are estimated using a linear projection of the previous cell state `c_{t-1}` and hidden state `h_{t-1}`:

```
P_hat = σ(W_p · c_{t-1} + U_p · h_{t-1} + b_p)
```

where `W_p` and `U_p` are `Linear(embed_dim, embed_dim)` layers, and `σ` is sigmoid.

#### Mask-Aware Fill

```
x_t_filled = m_t ⊙ x_t + (1 - m_t) ⊙ P_hat
```

where `⊙` is element-wise multiplication. Observed entries (mask=1) pass through unchanged; missing entries (mask=0) are replaced with the imputed estimate.

#### Simplified LSTM Gate Update

The filled map then proceeds through a simplified LSTM with a single gate (matching the SwinLSTM repo):

```
F_t = SwinTransformer(x_t_filled, h_{t-1})
gate = σ(F_t)
cell = tanh(F_t)
c_t = gate ⊙ (c_{t-1} + cell)
h_t = gate ⊙ tanh(c_t)
```

The `SwinTransformer(·)` function applies a stack of Swin Transformer blocks with alternating W-MSA and SW-MSA. The hidden state `h_{t-1}` is integrated by concatenation with the input along the feature dimension, followed by `Linear(2*dim, dim)` reduction.

### 3.5 Decoder: Vanilla SwinLSTM Unit

The decoder uses vanilla SwinLSTM units **without** the imputation mechanism. Decoder cells process only complete (imputed) token representations:

```
F_t = SwinTransformer(h_{t-1}, h_{t-1})    — input = previous hidden state
gate = σ(F_t), cell = tanh(F_t)
c_t = gate ⊙ (c_{t-1} + cell)
h_t = gate ⊙ tanh(c_t)
```

No mask or imputation is applied in decoder cells.

### 3.6 Reconstruction Layer

Maps the decoder's final hidden state back to pixel-space spectrum maps:

```
Input:  (B, L=375, embed_dim=128)
→ Linear(embed_dim → patch_h * patch_w * F)
  where patch_h=1, patch_w=2, F=1 → output channels = 2
→ Reshape to (B, F, H, W)
→ tanh activation
Output: (B, F=1, H=3, W=250)
```

The reconstruction layer predicts the padded map size and the final output is cropped back to the original spatial extent.

### 3.7 Decoder Feedback

Two feedback modes control what the decoder receives at each autoregressive step:

- **`pixel_feedback`** (default): The reconstruction output `y_hat_t` is passed back through the patch hierarchy to re-enter token space.
- During training, teacher forcing can replace `y_hat_t` with the ground-truth target frame according to `teacher_forcing_ratio`.
- During evaluation and inference, decoding is fully autoregressive.

---

## 4. Output Format

### Raw Output Tensor Shape

```
(B, T_out, F, H, W)
```

| Dimension | Meaning |
|-----------|---------|
| B | Batch size |
| T_out | Number of predicted future minutes |
| F | Channels (1 for CSV mode) |
| H | Spatial height (n_nodes in full mode, 1 in CC2 mode) |
| W | Spatial width (frequency bins per node) |

Output is in **normalized (MinMax) space** `[-1, 1]`. Predictions must be denormalized for interpretation:

```
value_dbm = (prediction + 1) / 2 × (data_max − data_min) + data_min
```

### CSV Output Format

Predicted data is written as CSV with the same format as the input:
- No header
- 6 decimal places
- 750 columns per row (CC1[0–249], CC2[250–499], LW1[500–749])
- One row per predicted minute
- Denormalized dBm values

### Visualization

- **Per-node spectrograms**: Side-by-side ground truth vs prediction for each node.
- **Error analysis**: Per-node error heatmaps showing residual patterns.
- All plots use denormalized dBm values.

---

## 5. Training Pipeline

### Data Loading Process

The data pipeline in `load_and_split()` supports both modes:

1. **CSV mode**: load CSV, select node subset columns, reshape to pseudo-spectrum-map `(T, H, W, F=1)`.
2. **Interpolated-map mode**: load prepared NPZ `(T, H, W, F)` and clean NaNs while preserving chronology.
3. Split chronologically.
4. Normalize using training-only statistics.
5. Build masked sliding windows for training/validation/testing.
4. Compute MinMax stats on training split (when `fit_on_train_only: true`):
   - A single `dmin` and `dmax` across all dimensions of the training data.
5. Normalize all splits to `[-1, 1]` using these stats.
6. Create `AERPAWDataset` instances that generate:
   - Sliding windows of `(T_in, H, W, F)` inputs and `(T_out, H, W, F)` targets.
   - Binary masks applied to input windows (not targets).

### Train / Validation / Test Split

Ratios: Default 80/10/10, configurable via `split` config section. Chronological split (recommended for time series).

### Loss Function

**Mean Squared Error (MSE)** in normalized space:

```
L = (1 / N) × Σ(predicted − target)²
```

MSE is computed on the full prediction tensor across all dimensions (batch, time, channel, height, width).

### Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.0001 (paper default) |
| Gradient clip norm | 5.0 |

### Epochs

- Default: **400** (paper default)
- Early stopping with patience of 30 epochs based on validation loss.

### Training Loop

1. Batch input `X: (B, T_in, F, H, W)` and mask `M: (B, T_in, F, H, W)` into the model.
2. **Encoder pass**: For `t = 0 .. T_in-1`:
   - Patch-embed `X[:, t]` to token space.
   - Pool mask to token space via `MaskPool`.
   - Feed through encoder SwinLSTM-I cells with mask-driven imputation.
3. **Decoder pass**: For `t = 0 .. T_out-1`:
   - Run decoder SwinLSTM cell (no masks, no imputation).
   - Reconstruct to pixel space.
   - Apply tanh activation.
   - Feed `h_t` back as next decoder input (when `decoder_feedback: hidden_state`).
4. Compute MSE loss between `pred` and `target`.
5. Backpropagate and update optimizer.

### Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| RMSE | √(mean((ŷ − y)²)) | Primary metric, same unit as dBm |
| MAE | mean(|ŷ − y|) | Robust to outliers |
| R² | 1 − (SS_res / SS_tot) | Proportion of variance explained |
| NRMSE(dB) | RMSE / (y_max − y_min) | Paper's primary metric |

Metrics are computed:
- **Overall**: Average across all dimensions.
- **Per horizon**: RMSE at each predicted time step, measuring error accumulation.
- **Per node**: RMSE for each sensor node separately.

---

## 6. Assumptions and Design Decisions

### Assumptions

1. The input CSV has `n_nodes × bins_per_node` columns (default 750) ordered as per `node_names` in config.
2. The CSV has no header and contains only comma-separated dBm values.
3. Rows (time axis) are in strict chronological order with no gaps.
4. Normalization statistics are computed globally across the training split (single `dmin`/`dmax`), not per-bin.
5. The train/val/test split is chronological, not random, because this is time-series data.
6. Masking applies only to input X; target Y remains complete (paper requirement).

### Design Decisions

1. **Pseudo-spectrum-map representation**: Sensor nodes are treated as rows and frequency bins as columns in a 2D map `(H=nodes, W=freq_bins, F=1)`. This allows 2D patch-based Swin Transformer processing despite lacking a true spatial grid.

2. **Padding-aware patch hierarchy**: arbitrary map sizes are padded to remain compatible with patch embedding, two merge stages, and the Swin window size. Final outputs are cropped back to the original extent.

3. **Hierarchical patch merging/expanding enabled**: the primary interpolated-map path follows the paper's multi-scale encoder/decoder design.

4. **MinMax normalization to `[-1, 1]`**: Matches the paper's normalization scheme. The tanh output activation is chosen to match this range.

5. **Pixel-feedback decoder with teacher forcing**: training can replace the autoregressive feedback frame with the ground-truth frame; evaluation and inference are autoregressive only.

6. **Mask pooled to token space via AvgPool2d**: The pixel-resolution mask is downsampled to match the patch-embedded token grid via average pooling, then expanded to match the embedding dimension for the imputation linear layers.

7. **Single `dmin`/`dmax` normalization**: Unlike per-bin z-score in ConvLSTM, DSwinLSTM-I uses global MinMax statistics. This preserves the paper's normalization approach.

8. **No external imputation**: Unlike STS-PredNet or ConvLSTM, the model integrates imputation inside the recurrent cell via the SwinLSTM-I mechanism.

---

## 7. Deviations from the Original Paper

| Aspect | Paper (DSwinLSTM-I) | Our Reconstruction | Reason |
|--------|---------------------|-------------------|--------|
| Dataset | Simulated spectrum maps (64×64 grid) | Prepared interpolated maps or CSV pseudo-maps | Interpolated maps are the primary paper-aligned path; CSV is an AERPAW adaptation |
| Map shape | 64 × 64 spatial grid | Arbitrary padded map sizes | Supports arbitrary prepared maps via padding/cropping |
| Patch size | 2 × 2 (square) | Configurable (default 2 × 2) | Keeps paper default while supporting arbitrary sizes |
| Patch Merging/Expanding | Used between encoder/decoder cells | Enabled in the primary path | Restored hierarchical multi-scale processing |
| Reconstruction | ConvTranspose2d / hierarchical upsampling | Linear patch reconstruction + crop | Exact-size reconstruction for arbitrary padded shapes |
| Normalization | MinMax [-1, 1] | MinMax [-1, 1] (same) | Same as paper |
| Output activation | Sigmoid | tanh | tanh matches [-1, 1] range; sigmoid matches [0, 1] |
| Optimizer | Adam | Adam | Same |
| Epochs | 400 | Configurable (default 400) | Same default |
| Batch size | 4 | Configurable (default 4) | Same default |
| Primary metric | NRMSE(dB) | NRMSE(dB) + RMSE/MAE/R² | Same primary metric, additional metrics |
| Input length | 10 | Configurable (default 10) | Same default |
| Prediction horizon | 10 | Configurable (default 10) | Same default |

---

## 8. Known Limitations

1. **AERPAW is sparse in the spatial dimension: only 3 nodes.** Treating node × frequency as a pseudo-image imposes artificial adjacency between frequency bins of the same node. The Swin attention window operates on this adjacency, which may not reflect true physical relationships.

2. **AERPAW pseudo-map lacks true spatial structure.** Unlike the paper's 64×64 grid where adjacent cells represent nearby spatial locations, our pseudo-map has no spatial meaning along the height axis — node rows are categorical, not spatial.

3. **CSV pseudo-map is still an adaptation path.** The interpolated-map path is more paper-aligned because it preserves a real spatial grid.

4. **Global normalization may not suit per-frequency power variations.** Unlike per-bin normalization (used by ConvLSTM), global MinMax applies the same transformation to all bins, potentially under-representing quieter frequency bands.

5. **Masking occurs in pixel space, not real sensor failures.** The synthetic masks (random/block/frequency/node) simulate missing data but may not reflect real-world sensor dropout patterns.

6. **CPU training is significantly slower than ConvLSTM due to self-attention.** The paper uses 400 epochs with batch size 4; a full run on CPU may take hours. GPU is strongly recommended.

---

## References

1. **DSwinLSTM-I paper:**
   *Robust Imputation SwinLSTM for Spectrum Map Prediction of Incomplete Data.* (Target architecture)

2. **SwinLSTM repo:**
   Tang, S., Li, C., Zhang, P., Tang, R. (2023). *SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin Transformer and LSTM.* ICCV 2023.
   Code: https://github.com/SongTang-x/SwinLSTM

3. **Swin Transformer paper:**
   Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
   Code: https://github.com/microsoft/Swin-Transformer

4. **AERPAW dataset:**
   AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022). DOI: [10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn).
