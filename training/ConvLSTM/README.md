# ConvLSTM Spectrum Prediction — Integrated Pipeline

> **Based on:** *Convolutional LSTM-based Long-Term Spectrum Prediction for Dynamic Spectrum Access* — Shawel, Woldegebreal, Pollin (EUSIPCO 2019)
>
> **Reference implementation:** https://github.com/ndrplz/ConvLSTM_pytorch
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

---

## Quick Start

```bash
python3 training/ConvLSTM/train_integrated.py
```

Outputs go to `training/results/ConvLSTM/` by default. Override with `--output-dir`:

```bash
python3 training/ConvLSTM/train_integrated.py \
    --output-dir /path/to/custom_output
```

Use a custom config with `--config`:

```bash
python3 training/ConvLSTM/train_integrated.py \
    --config /path/to/custom_config.yaml
```

---

## Scripts Reference

### `train_integrated.py` — Train and evaluate per chunk

The integrated runner trains one model per 200 MHz chunk defined in the shared config.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `training/common/config.yaml` | Shared config with data/model sections |
| `--output-dir` | `training/results/ConvLSTM/` | Output directory for metrics and models |

Outputs:

```
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
models/<chunk_id>_convlstm.pt
<chunk_id>_training_log.csv
```

Evaluation is built into the runner — no separate `evaluate.py`. Metrics are reported per horizon (1, 5, 15, 60 min by default) for each chunk's test split.

### `model.py` — ConvLSTM architecture

Contains `ConvLSTMCell`, `ConvLSTM` (multi-layer), and `ConvLSTMPredictor` (seq2seq encoder–decoder). Same architecture as the main branch.

---

## 1. What the Model Is Intended to Do

The model performs **long-term spectrum prediction** using the AERPAW dataset. Given a window of past per-minute PSD measurements from a single CC2 200 MHz chunk, it predicts the PSD values for multiple future time steps across all frequency bins in that chunk.

This is a **multi-step time series regression** problem:

```
⟨χ_{t-n}, ..., χ_{t-2}, χ_{t-1}⟩  ⟶  ⟨χ_t, χ_{t+1}, ..., χ_{t+m}⟩
```

where `n` = lookback (60 min), `m` = max horizon (60 min), and `χ_t` is a 1D power spectrogram of 200 frequency bins at time `t`.

**Key difference from paper:** The paper trains on 3 nodes simultaneously using a `(3, 250)` spatial-spectral map. Our integrated pipeline trains one model per chunk on a single node (CC2) with `(1, 200)` bins per chunk, then evaluates across all configured chunks (600–800, 2400–2600, 3500–3700 MHz).

---

## 2. Input Format

### 2.1 Per-Chunk Data Loading

The integrated pipeline uses `load_chunk()` from `training/common/data.py`. For each configured chunk, it:
1. Reads CC2's per-minute CSV from `evaluation/aerpaw/`
2. Selects frequency columns within `[start_mhz, end_mhz)` (e.g., 600–800 MHz = 200 bins)
3. Interpolates missing values
4. Applies z-score normalization per frequency bin
5. Splits into train and test sets chronologically

**NaN handling (``clean_interpolated_map`` in ``training/common/data.py``):**
   When the integrated pipeline eventually supports the pre-interpolated map format (``.npz``), loaded arrays are cleaned before normalization: fully-NaN timesteps are dropped, partial-NaN spatial slices are filled via nearest-neighbor interpolation, and any remaining NaNs are imputed with the per-frequency training-set mean. An assertion guarantees no NaNs enter training.

### 2.2 Expected Tensor Shapes

```
Raw chunk:           (T, 200)            — (time, frequency bins)
After normalization:  (T, 200)
Window input:        (B, 60, 1, 1, 200)  — (batch, lookback, channels, height, width)
Window target:       (B, 60, 1, 1, 200)  — (batch, horizon, channels, height, width)
```

The 2D spatial structure is `(1, 200)` — height=1 (single node), width=200 (frequency bins in the chunk).

---

## 3. Model Architecture

### 3.1 Overview

```
Input: (B, 60, 1, 1, 200)
        │
        ▼
┌────────────────────────────────────┐
│  Encoder                           │
│  ConvLSTM Layer 1: 1 → 32, k=(1,3)│
│  ConvLSTM Layer 2: 32 → 64, k=(1,1)│
└────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│  Decoder                           │
│  Regular LSTM: 64*200 → 128        │
│  ConvLSTM Layer 3: 1 → 32, k=(1,1)│
│  Conv2d(32 → 1, k=1)              │
└────────────────────────────────────┘
        │
        ▼
Output: (B, 60, 1, 1, 200)
```

### 3.2 Encoder — ConvLSTM Layers

| Parameter | Layer 1 | Layer 2 |
|-----------|---------|---------|
| Input channels | 1 | 32 |
| Hidden channels | 32 | 64 |
| Kernel size | `(1, 3)` | `(1, 1)` |
| Padding | `(0, 1)` | `(0, 0)` |
| Output shape | `(B, 60, 32, 1, 200)` | `(B, 60, 64, 1, 200)` |

**Activation:** Configurable via `convlstm.model.cell_activation` (default `relu`, per paper §III-A). Options: `relu`, `tanh`, `sigmoid`, `gelu`, `leaky_relu`, `elu`.

### 3.3 State Transfer — LSTM Bottleneck

The encoder's final hidden state `(B, 64, 1, 200)` is flattened to `(B, 64*200)`, passed through a regular LSTM with hidden size 128, and projected back to `(B, 32, 1, 200)` to initialize the decoder.

This LSTM layer is our interpretation of the paper's "LSTM hidden layer to capture memory and hidden states from the encoder output."

### 3.4 Decoder

| Component | Details |
|-----------|---------|
| ConvLSTM Layer 3 | `1 → 32`, kernel `(1, 1)`, dropout 0.3, batch norm |
| Output layer | `Conv2d(32 → 1, kernel=1)` |

The decoder generates the output sequence autoregressively. Teacher forcing (default ratio 1.0) feeds ground-truth frames during training.

When `model.fc_hidden_channels > 0`, the output head becomes a two-layer MLP (`Conv2d → Conv2d`) with configurable intermediate activation (`model.fc_intermediate_activation`, default `relu`).

### 3.5 ConvLSTM Cell Equations

```
i_t = σ(W_xi ∗ X_t + W_hi ∗ H_{t-1} + b_i)
f_t = σ(W_xf ∗ X_t + W_hf ∗ H_{t-1} + b_f)
o_t = σ(W_xo ∗ X_t + W_ho ∗ H_{t-1} + b_o)
g_t = activation(W_xg ∗ X_t + W_hg ∗ H_{t-1} + b_g)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
H_t = o_t ⊙ activation(C_t)
```

where `∗` = 2D convolution, `⊙` = element-wise product, `σ` = sigmoid, `activation` = configurable (`convlstm.model.cell_activation`, default ReLU per paper §III-A).

---

## 4. Output Format

```
Raw output:       (B, 60, 1, 1, 200)
Squeezed:         (B, 60, 200)
Per horizon h:    (B, 200)  — extracted at index h-1
Denormalized:     dBm via inverse z-score
```

Metrics are computed per horizon in denormalized dBm space using the shared `absolute_and_squared_errors_dbm()` function from `training/common/metrics.py`.

---

## 5. Training Pipeline

### 5.1 Data Loading (integrated)

```python
data = load_chunk(config, chunk)
train = data.splits[data.train_split].model_input   # (T_train, 200)
test = data.splits[data.test_split].model_input      # (T_test, 200)
```

### 5.2 Window Generation

Windows are generated on-the-fly by `ConvLSTMWindowDataset`:

- Lookback = 60 (configurable via `windowing.lookback`)
- Prediction horizon = 60 (configurable via `convlstm.prediction_horizon`)
- Origins = `[lookback-1, ..., len(data)-horizon]`
- Val split = last 10% of origins

### 5.3 Loss, Optimizer, Training

| Parameter | Value | Source |
|-----------|-------|--------|
| Loss | MSE | Standard |
| Optimizer | Adam (paper uses NADAM) | PyTorch standard |
| Learning rate | 0.0002 | Paper §III-B |
| Weight decay | 0.004 | Paper §III-B |
| Gradient clip | 5.0 | Common practice |
| Batch size | 32 | Configurable |
| Epochs | 25 | Configurable (reduced from paper's 100) |
| Teacher forcing | 1.0 (always on) | Our addition |
| Early stopping | Best val loss | No fixed patience |

### 5.4 Evaluation

For each configured horizon `h ∈ [1, 5, 15, 60]`:
1. Build target rows with `target_rows_for()`
2. Extract `h`-step-ahead prediction from the full `T_out` output
3. Denormalize and compute RMSE/MAE
4. Write to aggregate, per-frequency, and per-band metric CSVs

---

## 6. Changes from Main Branch (Standalone ConvLSTM)

| Aspect | Main Branch (Standalone) | Integrate Branch |
|--------|------------------------|------------------|
| **Data loading** | Per-model dataset.py loads 750-col CSV, reshapes to `(T, 3, 250)` | Shared `load_chunk()` reads per-site CSVs, extracts single-node chunk |
| **Nodes** | All 3 nodes (CC1, CC2, LW1) in `(H=3, W=250)` spatial-spectral map | Single node CC2 per chunk `(H=1, W=200)` |
| **Training targets** | All 3 nodes, 250 bins | Per-chunk 200 MHz band on CC2 |
| **Config** | `training/ConvLSTM/config.yaml` (standalone) | `training/common/config.yaml` (shared, `convlstm:` section) |
| **Normalization** | Per-model z-score fit on training only | Handled by `load_chunk()` / aerpaw_loader |
| **Train/val/test split** | Chronological 80/10/10 | Chronological, test = last 2 days (configurable via `test_split`) |
| **Inference** | Separate `inference.py` script | No standalone infer — evaluation is built into `train_integrated.py` |
| **Chunks** | Single full-band model | 3 separate models per 200 MHz chunk |

### Rationale for Changes

The integrated pipeline standardizes data loading and evaluation across all models. Instead of each model independently loading the 750-column merged CSV and doing its own reshaping/windowing, all models share `load_chunk()` which handles interpolation, normalization, splitting, and chunk extraction uniformly. This ensures consistent train/test splits and normalization across all compared models.

The per-chunk approach is justified because our evaluation uses 200 MHz bands on a single node — the spatial dimension is 1 (not 3), making the 2D ConvLSTM's `H` dimension trivial. If future work needs multi-node training, this can be added to the shared data pipeline.

---

## 7. Configuration Reference

All ConvLSTM settings are under `convlstm:` in `training/common/config.yaml`:

| Field | Default | Description |
|-------|---------|-------------|
| `input_sequence_length` | 60 | Past minutes (T_in) |
| `prediction_horizon` | 60 | Future minutes to predict (T_out) |
| `batch_size` | 32 | Mini-batch size |
| `epochs` | 25 | Max training epochs |
| `learning_rate` | 0.0002 | Adam learning rate |
| `weight_decay` | 0.004 | L2 weight decay |
| `val_fraction` | 0.1 | Fraction of windows for validation |
| `teacher_forcing_ratio` | 1.0 | Teacher forcing probability |
| `gradient_clip_norm` | 5.0 | Max gradient norm |
| `model.input_channels` | 1 | Input channel count |
| `model.hidden_channels` | [32, 64] | Encoder layer hidden sizes |
| `model.kernel_size` | [[1, 3], [1, 1]] | Encoder kernel sizes |
| `model.num_encoder_layers` | 2 | Encoder ConvLSTM layers |
| `model.decoder_hidden_channels` | 32 | Decoder hidden size |
| `model.decoder_kernel_size` | [1, 1] | Decoder ConvLSTM kernel |
| `model.dropout` | 0.3 | Dropout probability |
| `model.use_batch_norm` | true | Batch normalization |
| `model.decoder_lstm_hidden` | 128 | LSTM hidden size after encoder |
| `model.fc_hidden_channels` | 0 | FC intermediate (0 = single 1×1 Conv2d) |
| `model.fc_kernel_size` | [1, 3] | FC kernel (only if fc_hidden_channels > 0) |
| `model.cell_activation` | `"relu"` | ConvLSTM cell activation (g candidate and h output). Options: `relu`, `tanh`, `sigmoid`, `gelu`, `leaky_relu`, `elu` |
| `model.fc_intermediate_activation` | `"relu"` | Activation for optional FC intermediate layer (only if fc_hidden_channels > 0). Same options |
| `model.use_channel_projection` | `false` | Apply 1×1 Conv2d before encoder to reduce channel count (map mode) |
| `model.channel_projection_dim` | 16 | Target channel count after projection (only if `use_channel_projection: true`) |
| `interpolated_map.enabled` | `false` | Enable interpolated-map mode |
| `interpolated_map.map_path` | `"evaluation/...npz"` | Path to .npz with pre-interpolated spatial map |
| `interpolated_map.map_key` | `"map_db"` | Key inside the .npz |
| `interpolated_map.n_freq_bins` | 200 | Number of frequency channels (becomes ConvLSTM input channels) |
| `interpolated_map.grid_height` | 50 | Spatial grid height/rows |
| `interpolated_map.grid_width` | 50 | Spatial grid width/columns |

---

## 8. Deviations from Original Paper

| Aspect | Paper (2019) | Our Integrated Version | Reason |
|--------|-------------|------------------------|--------|
| Dataset | Electrosense (5 sensors, 450–520 MHz) | AERPAW (3 nodes, per-chunk) | Our target dataset |
| Spatial dimension | 40×40 IDW grid → 1600 locations | 1 node per chunk (height=1) | Single-node per-chunk setup |
| Input time steps | 120 (6 hours × 3 min) | 60 (1 hour × 1 min) | Configurable; 1-min resolution |
| Prediction horizon | 50 steps (150 min) | 60 steps (60 min) | Configurable via shared config |
| Encoder layers | 2 ConvLSTM | 2 ConvLSTM | Same structure |
| Decoder layers | LSTM + ConvLSTM + FC | LSTM + ConvLSTM + FC (1×1 Conv2d) | Same structure |
| Activation | ReLU (output, per §III-A) | Configurable (`cell_activation`; default ReLU) | Config-driven; matches paper at default |
| Optimizer | NADAM | Adam | NADAM not in standard PyTorch |
| Framework | R + TensorFlow | Python + PyTorch | Our stack |

---

## References

1. **Original paper:** B. S. Shawel, D. H. Woldegebreal, S. Pollin, "Convolutional LSTM-based Long-Term Spectrum Prediction for Dynamic Spectrum Access," EUSIPCO 2019.
2. **ConvLSTM paper:** X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, W.-C. Woo, "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting," NIPS 2015.
3. **Reference implementation:** [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch)
4. **AERPAW dataset:** DOI: [10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn)
