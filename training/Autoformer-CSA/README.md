# Autoformer-CSA Spectrum Prediction — Adaptation

> **Based on:** *An Autoformer-CSA Approach for Long-Term Spectrum Prediction* — Pan et al., IEEE Wireless Communications Letters, 2023.
>
> **Reference implementation:** https://github.com/Demii-7/Autoformer
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

---

## Quick Start

*Implementation scripts not yet created. This document describes the planned adaptation.*

```bash
# Planned invocation (once implemented):
python3 training/Autoformer-CSA/train.py --config training/Autoformer-CSA/config.yaml
```

Outputs will go to `training/results/Autoformer-CSA/` by default.

---

## Scripts Reference

*To be implemented — see Section 11 for planned file responsibilities.*

---

## File Structure

```
training/Autoformer-CSA/
├── README.md           # This file — adaptation plan and architecture reference
├── config.yaml         # Tunable hyperparameters and pipeline configuration
├── dataset.py          # AERPAW dataset loader with sliding windows (planned)
├── model_csa.py        # Autoformer backbone + CSAM replacement (planned)
├── train.py            # Training loop, checkpointing, evaluation (planned)
├── evaluate.py         # Standalone evaluation on test split (planned)
├── inference.py        # Forward pass on new CSV data (planned)
└── utils.py            # Metrics, visualization, helpers (planned)
```

---

## Configuration Reference

All settings live in `config.yaml`. See that file for detailed descriptions.

| Section | Key Fields |
|---------|------------|
| `data` | dataset_path, n_features, n_nodes, bins_per_node, cc2_only_smoke_test |
| `windowing` | seq_len, label_len, pred_len, train_stride, val_stride, test_stride |
| `split` | train_ratio, val_ratio, test_ratio, chronological_split |
| `preprocessing` | normalization (zscore), fit_on_train_only |
| `model` | enc_in, dec_in, c_out, d_model, d_ff, encoder_layers, decoder_layers, n_heads, moving_avg, dropout, use_csam, run_vanilla_autoformer_baseline, csam_kernel_size, csam_reduction |
| `training` | batch_size, epochs, learning_rate, optimizer, loss, early_stopping, patience |
| `evaluation` | metrics, eval_horizons, export_predictions, plot_denormalized_dbm |
| `paths` | checkpoints_dir, evaluation_dir |
| `device` | device (auto / cuda / cpu) |

---

## 1. What the Model Is Intended to Do

Autoformer-CSA performs **long-term spectrum prediction**: given historical power spectral density (PSD) measurements, it forecasts future PSD values across multiple frequency bins and time steps.

The model builds on the **Autoformer** architecture (Wu et al., 2021), retaining:

- **Series decomposition block** — moving-average-based trend/seasonal separation at each layer
- **Auto-correlation mechanism** — period-based dependency discovery replacing self-attention, with O(L log L) complexity
- **Encoder-decoder structure** — progressive trend extraction in the decoder

The key modification is replacing Autoformer's feed-forward network (FFN) with the **Series Channel-Spatial Attention Module (CSAM)**. CSAM applies channel attention followed by series spatial-position attention, both using 1D convolutions (adapted from the 2D convolutions used in image attention mechanisms). The goal is to improve feature selection across both frequency channels and temporal positions.

The paper evaluates on Electrosense 600–640 MHz data from two sensors in Madrid. Our target is the **AERPAW sub-6 GHz dataset** (three fixed nodes: CC1, CC2, LW1, Feb 2022), which presents a different spectral environment and sensor layout.

---

## 2. Input Format

### Paper Format

```
P ∈ R^(M × F)
```

Where:
- `M` = input time steps (default 96)
- `F` = number of frequency channels (Electrosense sensors)

### AERPAW Format

The raw merged CSV after combining three nodes:

```
Raw CSV shape: (6839, 750)
```

Where `750` columns correspond to:

```
Columns  0–249   = CC1 (node 1, 250 frequency bins)
Columns 250–499  = CC2 (node 2, 250 frequency bins)
Columns 500–749  = LW1 (node 3, 250 frequency bins)
```

After windowing for the model:

```
X: (T_in, 750)    — encoder input
Y: (T_out, 750)   — decoder target (and prediction)
```

### Key Design Choices

| Choice | Description |
|--------|-------------|
| **Chronological splitting** | Train = earliest samples, val = middle, test = latest. No random shuffling across time. |
| **Train-only normalization** | Z-score statistics (mean, std) computed only on the training split, applied to val/test. |
| **Sliding windows** | Overlapping windows generated with configurable stride. |
| **CC2-only smoke-test mode** | When `cc2_only_smoke_test: true`, only columns 250–499 are used (single-node 250-bin input) to verify model trains correctly before full 750-channel runs. |
| **3-node full mode** | Default — all 750 channels used as a multivariate time series. |

### Optional Analysis Reshape

For per-node per-frequency evaluation only:

```
Raw output: (T_out, 750)
Reshaped:   (T_out, 3, 250)  — for per-node plotting and metric breakdown
```

The model never sees this 3D shape during training.

---

## 3. Model Architecture

### 3.1 Autoformer Backbone

The paper preserves the full Autoformer architecture:

```
Encoder:                          Decoder:
┌──────────────┐                  ┌──────────────┐
│Embedding      │                  │Embedding      │
│(no position)  │                  │(no position)  │
└──────┬───────┘                  └──────┬───────┘
       │                                 │
       ▼                                 ▼
┌──────────────┐  K layers      ┌──────────────┐
│EncoderLayer   │                │DecoderLayer   │ D layers
│  Autocorr ──► │                │  Self-Autocorr│
│  +Decomp1     │                │  +Decomp1     │
│  FFN/CSAM ──► │                │  Cross-Autocorr│
│  +Decomp2     │                │  +Decomp2     │
└──────────────┘                 │  FFN/CSAM ──► │
                                 │  +Decomp3     │
                                 └──────────────┘
                                        │
                                        ▼
                               ┌────────────────┐
                               │ trend + seasonal│
                               │   projection    │
                               └────────────────┘
```

**Series decomposition** at each layer:

```
seasonal, trend = series_decomp(x)
  where trend = AvgPool1d(x)  (moving average)
        seasonal = x - trend
```

**Auto-correlation mechanism** replaces scaled dot-product attention:

1. FFT-based period discovery — computes correlation via `FFT → conjugate → iFFT`
2. Top-K period selection
3. Time-delay aggregation — rolls the value sequence by selected periods and aggregates

### 3.2 Decoder Initialization

The decoder input is constructed from the encoder input's decomposition:

```
Encoder input x_enc decomposed:
  seasonal_init, trend_init = decomp(x_enc)

Decoder seasonal input = seasonal_init[-label_len:]  concatenated with  zeros(pred_len)
Decoder trend input    = trend_init[-label_len:]     concatenated with  repeat(mean(x_enc), pred_len)
```

- The seasonal part receives zero placeholders for the prediction horizon (the model learns to fill these).
- The trend part receives the mean of `x_enc` as a placeholder.

This is already implemented in the reference repo at `models/Autoformer.py:77-82`.

### 3.3 CSAM — Series Channel-Spatial Attention Module

CSAM replaces the FFN in both the encoder and decoder layers. The original FFN is:

```python
y = Dropout(Activation(Conv1d(d_model → d_ff, k=1)))
y = Dropout(Conv1d(d_ff → d_model, k=1))
```

CSAM operates on the **embedded representation** — its channel dimension is `d_model` (typically 512), not the raw 750 input features. The 750-channel input affects only the embedding layer and final projection; CSAM scales with `d_model`.

CSAM replaces the FFN with a two-branch attention module:

**1. Channel Attention Branch**

```
Input: x ∈ R^(B, T, C)
  → MaxPool1D along T dimension → channel descriptor
  → AvgPool1D along T dimension → channel descriptor
  → Concatenate pooled descriptors
  → Conv1D(reduction_ratio) → ReLU → Conv1D(expand)
  → Sigmoid → channel attention weights
  → x * channel_weights
```

- MaxPool1D and AvgPool1D aggregate temporal information per channel.
- Conv1D with reduction ratio compresses then expands the channel dimension.
- Sigmoid produces per-channel importance weights.

**2. Series Spatial-Position Attention Branch**

```
Input: x ∈ R^(B, T, C)
  → Max over channels → (B, T, 1) descriptor
  → Mean over channels → (B, T, 1) descriptor
  → Concatenate along channel dim → (B, T, 2)
  → Conv1D(kernel_size) → Sigmoid
  → spatial attention weights over sequence positions
  → x * spatial_weights
```

- Max and mean pool across the channel dimension.
- 1D convolution with configurable kernel size captures local temporal context.
- Sigmoid produces per-timestep importance weights.

**3. Final Output**

```
Output = Conv1D(d_model → d_model, k=1) + Dropout
```

The paper uses 1D convolutions throughout (instead of 2D convolutions used in image-based CBAM), adapting the attention mechanism for time-series/spectrum features where the "spatial" dimension is the sequence length.

The skip connection and series decomposition remain unchanged:

```
Before:  attention → decomp1 → FFN → decomp2 → output
After:   attention → decomp1 → CSAM → decomp2 → output
```

### 3.4 CSAM Integration Points

| Location | File in Repo | Lines | Modification |
|----------|-------------|-------|--------------|
| EncoderLayer forward | `layers/Autoformer_EncDec.py` | 75–78 | Replace `conv1 → activation → dropout → conv2 → dropout` with CSAM |
| DecoderLayer forward | `layers/Autoformer_EncDec.py` | 143–146 | Same replacement |
| New module | `models/model_csa.py` (planned) | — | Define `CSAM` class with channel and spatial attention |

---

## 4. Output Format

### Model Output

```
Raw prediction: (B, T_out, 750)
```

For evaluation on the test set:

| File | Description |
|------|-------------|
| `predictions.csv` | Denormalized predictions, shape `(T_test, 750)` |
| `ground_truth.csv` | Denormalized ground truth, shape `(T_test, 750)` |
| `metrics.json` | RMSE, MAE, R² per node and overall |

### Visualization

All plots denormalized to dBm:

- **Spectrogram plots** — one per node (CC1, CC2, LW1) showing predicted vs actual PSD over time
- **Error analysis plot** — per-frequency-bin error heatmap or MAE across the spectrum
- **Optional:** per-horizon comparison plots for multi-horizon evaluation

### Optional Reshape for Analysis

```
(T_out, 750) → (T_out, 3, 250)
```

Allows slicing `[node_idx, :]` for per-node evaluation without modifying model internals.

---

## 5. Training Pipeline

### 5.1 Data Flow

```
CSV (6839, 750)
  → Chronological split (train / val / test)
  → Z-score normalization (fit on train only)
  → Sliding window generation
  → Encoder input X: (B, T_in, 750)
  → Decoder target Y: (B, T_out, 750)
```

### 5.2 Encoder/Decoder Construction for Training

```
seq_x = data[s_begin : s_end]                             # encoder input:  (seq_len, features)
seq_y = data[s_end - label_len : s_end + pred_len]        # decoder input:  (label_len + pred_len, features)
seq_x_mark = time_features[s_begin : s_end]                # encoder timestamp
seq_y_mark = time_features[s_end - label_len : s_end + pred_len]  # decoder timestamp
```

Loss is computed only on the final `pred_len` steps of the decoder output:

```
loss = MSE(model(seq_x)[:, -pred_len:, :], seq_y[:, -pred_len:, :])
```

### 5.3 Training Loop

| Component | Detail |
|-----------|--------|
| **Loss** | MSE during training |
| **Optimizer** | Adam |
| **Scheduler** | ReduceLROnPlateau (optional, per repo default) |
| **Early stopping** | Patience-based on val loss |
| **Checkpointing** | Save best model by val loss |
| **Evaluation metrics** | RMSE, MAE, R² (computed on denormalized predictions) |

### 5.4 Paper Defaults

| Parameter | Paper Default |
|-----------|--------------|
| Encoder layers | 2 |
| Decoder layers | 1 |
| Attention heads | 8 |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Epochs | 20 |
| Patience | 6 |
| Normalization | Z-score |
| Train/val/test split | 5:1:1 |
| Input length M | 96 |
| Prediction range K | {60, 120, 240, 300} |

All of these are configurable via `config.yaml`.

---

## 6. Assumptions and Design Decisions

1. **Use Autoformer repo as base** — The forked repo at `github.com/Demii-7/Autoformer` is vanilla Autoformer. CSAM must be added as a replacement for the Conv1d-based FFN.

2. **Add CSAM in place of FFN only** — The decomposition block, auto-correlation mechanism, embedding (no positional encoding), and decoder initialization remain unchanged from the paper's specification.

3. **Treat AERPAW as multivariate time series with 750 channels** — Each frequency bin is a separate channel. The model sees `(B, T, 750)` tensors.

4. **Do not reshape to image-like `(3, 250)` for training** — The 3-node, 250-bin structure is only used for evaluation/plotting, not as a 2D input. This avoids conflating node identity with spatial adjacency.

5. **Use chronological splits** — No random shuffling; the temporal order is preserved. The earliest fraction is training, middle is validation, latest is test.

6. **Fit normalization only on train split** — Train statistics are applied to val and test to avoid data leakage.

7. **Start with CC2-only smoke test** — Verify the model trains and converges on a single node (250 channels) before scaling to full 750 channels.

8. **Use `train_stride=1` and `val/test_stride=T_out`** — Dense windows for training (maximizing data); non-overlapping windows for clean evaluation without leakage.

9. **Paper-faithful decoder initialization** — The latter half of encoder seasonal+trend + zero/mean placeholders, as implemented in the reference repo.

### Implementation Decisions Already Confirmed

| Decision | Status |
|----------|--------|
| Autoformer repo is vanilla Autoformer; CSAM is not implemented | Confirmed via repo inspection — no CSA/CSAM code exists |
| FFN replacement points are in `layers/Autoformer_EncDec.py` | `EncoderLayer` lines 61–62/75–78 and `DecoderLayer` lines 122–123/143–146 |
| `seq_len`, `label_len`, `pred_len` naming matches the repo | Used consistently across `run.py`, models, datasets |
| Decoder initialization already exists | `models/Autoformer.py:77-82` — no changes needed |
| Normalization will be handled in custom `dataset.py` | Repo's `Dataset_Custom` normalizes internally; we will handle z-score in our own loader |
| Evaluation exports must use denormalized dBm | All other AERPAW pipeline models use denormalized metrics for comparability |

---

## 7. AERPAW Dataset Adaptation

### Paper vs Our Setup

| Aspect | Paper (Autoformer-CSA) | Our Pipeline (AERPAW) |
|--------|----------------------|----------------------|
| **Dataset** | Electrosense | AERPAW |
| **Frequency range** | 600–640 MHz | Multi-band (full 750-bin merged) |
| **Sensors/nodes** | 2 sensors in Madrid | 3 fixed nodes: CC1, CC2, LW1 |
| **Data points** | Not specified | 6839 rows |
| **Features** | Per-sensor PSD (unknown count) | 750 features (3 × 250 bins) |
| **Temporal resolution** | 1 minute | 1 minute |
| **Train/val/test split** | 5:1:1 | 5:1:1 (configurable) |
| **Input length M** | 96 | Configurable via `seq_len` |
| **Prediction range K** | {60, 120, 240, 300} | Configurable via `pred_len` |
| **Label length** | M/2 | Configurable via `label_len` |

### Important Note

Our results are **not** a direct reproduction of the paper's numbers. The spectral environment, sensor hardware, and geographic location differ significantly. The architecture is adopted, but the specific performance numbers will be dataset-dependent.

---

## 8. Repo-Specific Implementation Notes

### Files in the Forked Repo

```
layers/
├── AutoCorrelation.py          # Auto-correlation mechanism ✅ keep
├── Autoformer_EncDec.py        # Encoder/Decoder layers, series_decomp ✅ keep (modify FFN → CSAM)
├── Embed.py                    # Embeddings (no positional encoding) ✅ keep
├── SelfAttention_Family.py     # Other attention variants ⚠️ not used
└── Transformer_EncDec.py       # Vanilla Transformer layers ⚠️ not used

models/
├── Autoformer.py               # Main Model class ✅ keep as base
├── Informer.py / Reformer.py / Transformer.py  ⚠️ not used

data_provider/
├── data_factory.py             # Dataset dispatcher ⚠️ replace with our own
└── data_loader.py              # Dataset classes (Dataset_Custom useful reference)

exp/
├── exp_basic.py                # Base experiment class ⚠️ replace
└── exp_main.py                 # Training/val/test loop ⚠️ replace with our train.py

utils/
├── metrics.py                  # MAE, MSE, RMSE, etc. ✅ adapt
└── tools.py                    # EarlyStopping, LR scheduler, viz ✅ adapt
```

### Key Findings

| Question | Answer |
|----------|--------|
| **Does CSAM already exist?** | **No.** The repo contains only vanilla Autoformer. The PDF paper is present as a reference document, but no CSA/CSAM code exists anywhere in the Python codebase. |
| **Where does the FFN live?** | `layers/Autoformer_EncDec.py` — `EncoderLayer` (lines 61–62, forward lines 75–78) and `DecoderLayer` (lines 122–123, forward lines 143–146). Both use a two-layer `Conv1d(d_model → d_ff → d_model)` with activation and dropout. |
| **What must be modified to insert CSAM?** | Replace the `conv1`/`conv2` Conv1d FFN in both `EncoderLayer` and `DecoderLayer` with the CSAM module. The CSAM module takes `(B, T, C)` input, applies channel attention then spatial attention, and outputs `(B, T, C)`. |
| **Dataset interface** | The repo's `Dataset_Custom` class loads CSV with a `date` column + feature columns. We will write our own `dataset.py` that loads the 750-column merged CSV and returns `(B, T_in, 750)` and `(B, T_out, 750)` tensors. |
| **Standard forecasting datasets** | The repo is designed for standard time series forecasting benchmarks (ETT, ECL, Exchange, Traffic, Weather, ILI). It uses `data custom` mode for arbitrary CSV with a `date` column. |
| **Custom CSV support** | Yes — `--data custom` + `--root_path` + `--data_path`. However, our pipeline uses a pre-merged 750-column CSV rather than the per-dataset format. |
| **seq_len/label_len/pred_len naming** | Used consistently throughout. `seq_len = T_in`, `label_len = T_in/2`, `pred_len = T_out`. |
| **Decoder initialization** | Already implemented in `models/Autoformer.py:77-82`. No changes needed. |
| **Multi-channel output** | Handled via `enc_in`, `dec_in`, `c_out`. Setting `c_out=750` should work directly with the final linear projection layer. |

### Required Modifications

1. **Create `models/model_csa.py`** — Define the CSAM module (channel attention + spatial attention + final Conv1d).
2. **Modify `layers/Autoformer_EncDec.py`** — In both `EncoderLayer` and `DecoderLayer`, either:
   - Add a `use_csam` parameter to conditionally switch between FFN and CSAM, or
   - Create new `CSAMEncoderLayer` / `CSAMDecoderLayer` classes.
3. **Update `models/Autoformer.py`** — Either modify to accept `use_csam` flag or create `AutoformerCSA` subclass.
4. **Create custom `dataset.py`** — AERPAW-specific data loading with chronological splits, z-score normalization, sliding windows.
5. **Create `train.py`** — Training loop with early stopping, checkpointing, evaluation.
6. **Create `evaluate.py`** — Standalone evaluation script for trained models.
7. **Create `inference.py`** — Forward pass on new data.
8. **Create `utils.py`** — Metrics computation, plotting, helper functions.

---

## 9. Known Limitations

1. **Horizon sensitivity** — Autoformer-CSA is designed for long-horizon forecasting (K ≥ 60). Smoke-test horizons (e.g., K = 1 or K = 5) may be too short to show the benefit of the auto-correlation mechanism and CSAM.

2. **Memory footprint** — 750 input features mainly affect embedding/projection and data tensors; CSAM channel attention mainly scales with `d_model`. The channel attention bottleneck is `d_model → d_model/r → d_model`, so a typical `d_model=512` with `r=16` produces a manageable 512→32→512 Conv1d. Still, full 750-channel training may be memory-heavy for the embedding and projection layers. Consider gradient checkpointing if OOM occurs.

3. **Tensor layout** — The repo uses `(B, T, C)` layout throughout (`batch_first`). The CSAM implementation must be careful with the `transpose(-1, 1)` patterns used in the existing Conv1d FFN (which expects `(B, C, T)`). The CSAM module should accept and return `(B, T, C)` consistently.

4. **Dataset mismatch** — The paper evaluates on Electrosense data (600–640 MHz, 2 sensors, Madrid). AERPAW has different spectral characteristics, node placement, and interference patterns. Results are not directly comparable.

5. **Decoder initialization correctness** — The paper specifies decoder initialization carefully. The repo already implements it, but we must verify the `label_len` slicing aligns with our windowing logic.

6. **Cross-model fairness** — For fair comparison with ConvLSTM, TimeRAN, TSS-LCD, and STS-PredNet, the evaluation pipeline must produce the same smoke-test export format (predictions.csv, ground_truth.csv, metrics.json) and use the same denormalized plotting functions.

7. **CSAM vs original FFN comparison** — To isolate the effect of CSAM, we should also support a `use_csam: false` mode that runs the original Autoformer FFN. This allows ablation studies.

---

## 10. Items to Verify Before Scripting

Before writing any implementation code, verify the following unresolved items:

1. **Confirm exact tensor layout in the forked repo during forward pass** — The expected model input appears to be `(B, T, C)`; internal Conv1d blocks transpose to `(B, C, T)`. CSAM must accept and return `(B, T, d_model)`.

2. **Confirm the safest integration strategy** — Option A: replace FFN inside existing `EncoderLayer` and `DecoderLayer`. Option B: create separate `EncoderLayerCSA` and `DecoderLayerCSA`. Prefer option B if it avoids breaking vanilla Autoformer.

3. **Confirm decoder output slicing** — Model output should be `(B, pred_len, c_out)`; training loss should compare against `seq_y[:, -pred_len:, :]`.

4. **Confirm custom AERPAW dataset compatibility** — No date column; synthetic time marks may be needed if the repo expects `x_mark` and `y_mark`. Alternatively, disable time-feature dependence if unused by Autoformer.

5. **Confirm final projection behavior** — Verify `c_out=250` works in CC2 smoke mode and `c_out=750` works in full mode, both for the `nn.Linear(d_model, c_out)` in the model and the Conv1d projection within `DecoderLayer`.

---

## 11. Implementation Notes

*The following files are planned but not yet created.*

### `dataset.py`

Responsibility: Load AERPAW merged CSV, apply chronological split, fit z-score normalization on train, generate sliding windows.

```
class AERPAWDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, label_len, stride):
        # Store pre-loaded and normalized data
        # Generate window indices

    def __getitem__(self, idx):
        # Return (seq_x, seq_y, seq_x_mark, seq_y_mark)
        # seq_x: (seq_len, features)
        # seq_y: (label_len + pred_len, features)
```

```python
def create_dataloaders(config):
    # Load CSV → split → normalize → window → DataLoader
```

### `model_csa.py`

Responsibility: Define CSAM module and (optionally) a modified Autoformer model that uses CSAM.

```
class ChannelAttention(nn.Module):
    # MaxPool1D + AvgPool1D → Conv1D → ReLU → Conv1D → Sigmoid

class SpatialAttention(nn.Module):
    # Max + Mean over channels → Concat → Conv1D → Sigmoid

class CSAM(nn.Module):
    # ChannelAttention → element-wise multiply → SpatialAttention → element-wise multiply → Conv1D → Dropout
```

The CSAM module should accept and return `(B, T, C)` tensors.

If a new model class (`AutoformerCSA`) is created, it should inherit or wrap the existing Autoformer forward logic, replacing only the FFN with CSAM.

### `train.py`

Responsibility:

- Parse config
- Create dataloaders
- Instantiate model
- Training loop with MSE loss, Adam optimizer
- Validation loop
- Early stopping (patience-based)
- Checkpoint saving (best val loss)
- Final evaluation on test set
- Export predictions.csv, ground_truth.csv, metrics.json
- Generate spectrogram plots (denormalized to dBm)

### `evaluate.py`

Responsibility: Load a trained checkpoint and config, run on the test split, export metrics and plots.

### `inference.py`

Responsibility: Load a trained checkpoint and config, run forward pass on new CSV data, export predictions.

### `utils.py`

Responsibility:

- `compute_metrics(pred, true)` → RMSE, MAE, R²
- `denormalize(data, mean, std)` → dBm values
- `plot_spectrogram(pred, true, node_name, output_path)`
- `plot_error_analysis(errors, frequencies, output_path)`
- `EarlyStopping` class (adapt from repo's `utils/tools.py`)
- `save_config_copy(config, output_dir)` — for reproducibility

---

## References

1. **Autoformer paper:** H. Wu, J. Xu, J. Wang, M. Long, "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting," NeurIPS 2021.
2. **Autoformer-CSA paper:** Pan et al., "An Autoformer-CSA Approach for Long-Term Spectrum Prediction," IEEE Wireless Communications Letters, 2023.
3. **Reference implementation:** [Demii-7/Autoformer](https://github.com/Demii-7/Autoformer) (vanilla Autoformer fork)
4. **CBAM (inspiration for CSAM):** S. Woo, J. Park, J.-Y. Lee, I. S. Kweon, "CBAM: Convolutional Block Attention Module," ECCV 2018.
5. **AERPAW dataset:** DOI: [10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn)
6. **AERPAW paper:** D. Uvaydov et al., "AERPAW: A Dataset for Long-Term Spectrum Prediction," IEEE DySPAN 2024.
