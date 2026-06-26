# DSwinLSTM-I Spectrum Prediction — Adaptation

> **Based on:** *Robust Imputation SwinLSTM for Spectrum Map Prediction of Incomplete Data* (target architecture)
>
> **Supporting code reference (vanilla SwinLSTM only, no imputation):** https://github.com/SongTang-x/SwinLSTM — ICCV 2023 paper implementation
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

**This is a paper reconstruction, not a direct adaptation of the SwinLSTM repo.** The repo provides vanilla SwinLSTM building blocks (Swin Transformer blocks, SwinLSTM cell, patch embedding/merging/expanding, reconstruction layer). The imputation unit (SwinLSTM-I), encoder-decoder separation, and mask-aware training pipeline must be implemented from scratch following the DSwinLSTM-I paper. The repo is a supporting reference for the Swin Transformer mechanics only.

---

## Quick Start

The following commands are proposed — scripts are planned but not yet created.

### Setup

```bash
cd /home/cc/spectrum-usage
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib tqdm scikit-learn pyyaml timm einops
```

### Train the Model

```bash
# Default config (full 3-node mode)
python3 training/DSwinLSTM-I/train.py

# CC2-only smoke test
python3 training/DSwinLSTM-I/train.py --config training/DSwinLSTM-I/smoke_test/config.yaml
```

Training creates `training/DSwinLSTM-I/checkpoints/` with `best_model.pt`, `last_model.pt`, and `normalization_stats.pt`.

### Evaluate

```bash
python3 training/DSwinLSTM-I/evaluate.py \
    --checkpoint training/DSwinLSTM-I/checkpoints/best_model.pt
```

Output: per-node and per-frequency-bin RMSE/MAE/R²/NRMSE(dB), spectrogram plots, and `predictions.csv`.

### Run Inference on New Data

```bash
python3 training/DSwinLSTM-I/inference.py \
    --checkpoint training/DSwinLSTM-I/checkpoints/best_model.pt \
    --input /path/to/new_measurements.csv \
    --output predictions.csv
```

---

## Scripts Reference

| Script | Status | Responsibility |
|--------|--------|---------------|
| `dataset.py` | Planned | CSV loading, pseudo-map reshaping, masking, windowing |
| `model.py` | Planned | SwinLSTM cell, SwinLSTM-I cell, encoder-decoder, reconstruction |
| `train.py` | Planned | Training loop, checkpoints, logs |
| `evaluate.py` | Planned | Metrics/plots/exports |
| `inference.py` | Planned | Standalone prediction on new CSV |
| `utils.py` | Planned | Config, normalization, metrics, plotting, checkpointing |

---

## File Structure

```
training/DSwinLSTM-I/
├── README.md              # This file (plan)
├── config.yaml            # Proposed configuration
├── dataset.py             # Planned — data loading, reshaping, masking, windowing
├── model.py               # Planned — SwinLSTM / SwinLSTM-I / encoder-decoder
├── train.py               # Planned — training entrypoint
├── evaluate.py            # Planned — evaluation entrypoint
├── inference.py           # Planned — inference on new CSV
├── utils.py               # Planned — shared utilities
├── checkpoints/           # Created during training
├── evaluation/            # Created during evaluation
└── smoke_test/            # Planned — CC2-only smoke test config
    └── config.yaml
```

---

## Configuration Reference

See `config.yaml` for the full configuration. Key fields:

| Section | Field | Default | Description |
|---------|-------|---------|-------------|
| `data` | `n_nodes` | `3` | Sensor nodes for pseudo-map |
| `data` | `map_representation` | `node_frequency` | How to reshape data: `node_frequency`, `flat_strip`, or `artificial_grid` |
| `data` | `cc2_only_smoke_test` | `false` | CC2-only mode (`H=1, W=250`) |
| `windowing` | `input_sequence_length` | `10` | Historical time steps (paper: `T_in=10`) |
| `windowing` | `prediction_horizon` | `10` | Future time steps to predict (paper: `T_out=10`) |
| `preprocessing` | `normalization` | `minmax` | Normalization method |
| `preprocessing` | `missing_rate` | `0.3` | Fraction of input masked as missing |
| `preprocessing` | `missing_strategy` | `random` | Mask generation strategy |
| `model` | `patch_shape` | `[1, 2]` | Rectangular patch size for AERPAW adaptation |
| `model` | `embed_dim` | `128` | Embedding dimension |
| `model` | `encoder_units` | `2` | Encoder SwinLSTM-I units (paper: 2) |
| `model` | `decoder_units` | `2` | Decoder SwinLSTM units (paper: 2) |
| `model` | `swin_depths` | `[2, 6, 6, 2]` | Swin blocks per cell (paper: 2,6,6,2) |
| `model` | `num_heads` | `[4, 8, 8, 4]` | Attention heads per layer |
| `model` | `window_size` | `4` | Swin attention window size |
| `model` | `use_imputation_unit` | `true` | Enable SwinLSTM-I imputation (encoder only) |
| `model` | `mask_as_input_channel` | `false` | Concatenate mask as extra channel |
| `model` | `padding_mode` | `reflect` | Padding for non-divisible dimensions |
| `training` | `batch_size` | `4` | Paper default for spectrum experiments |
| `training` | `epochs` | `400` | Paper default |
| `training` | `learning_rate` | `0.0001` | Paper default |
| `evaluation` | `metrics` | `rmse, mae, r2, nrmse_db` | Including paper's primary metric NRMSE(dB) |

---

## 1. What the Model Is Intended to Do

DSwinLSTM-I (Deep SwinLSTM with Imputation) performs **joint imputation and prediction** of future spectrum maps from incomplete or corrupted historical maps.

Key properties:

- **Joint imputation-prediction.** Unlike a two-stage pipeline (impute missing data first, then predict), DSwinLSTM-I integrates missing-value estimation inside the recurrent cell. Missing entries are inferred from prior hidden and cell states, not from an external imputation method.
- **Mask-aware.** A binary mask `M_t` indicates which entries of the input `P_t` are observed (1) and which are missing/corrupted (0). Only the input is masked; the target remains complete.
- **Swin Transformer backbone.** The model uses Swin Transformer blocks (shifted-window multi-head self-attention) inside an LSTM-like recurrent structure. This replaces the convolutional operations in ConvLSTM with self-attention for capturing global spatial dependencies.
- **Encoder-decoder architecture.** The encoder uses **SwinLSTM-I units** (with imputation). The decoder uses **vanilla SwinLSTM units** (prediction only). This separation lets the encoder focus on reconstructing missing entries while the decoder focuses on forecasting.

The paper's visual architecture (Fig. 1):

- **Fig. 1(a):** SwinLSTM-I block — shows the imputation unit that estimates `P_hat_t` from `C_{t-1}` and `H_{t-1}`, then fills missing entries with `P_t = M_t ⊙ P_t + (1 - M_t) ⊙ P_hat_t` before proceeding to the SwinLSTM update.
- **Fig. 1(b):** Vanilla SwinLSTM block — standard LSTM gating with Swin Transformer replacing convolutions.
- **Fig. 1(c):** Full DSwinLSTM-I encoder-decoder — encoder with SwinLSTM-I cells and patch merging, decoder with SwinLSTM cells and patch expanding, finishing with a reconstruction layer.

---

## 2. Input Format

### Paper Format

```text
P_T ∈ R^(T × H × W × F)      — Incomplete/corrupted historical spectrum maps
M_T ∈ R^(T × H × W × F)      — Binary mask (1 = observed, 0 = missing/corrupted)
Y ∈ R^(T_out × H × W × F)    — Complete future spectrum maps
```

Paper simulation: `H=64, W=64, F=1, T_in=10, T_out=10`.

### AERPAW Format

```text
Raw CSV:                    (6839, 750)
Reshape full mode:          (T, 3, 250, 1)
Reshape CC2 smoke mode:     (T, 1, 250, 1)
```

### Window Format

```text
X:      (T_in, H, W, F)     — Incomplete input maps
Mask:   (T_in, H, W, F)     — Binary mask for X
Y:      (T_out, H, W, F)    — Complete target maps
```

Missing/anomalous entries are represented with a binary mask:

```text
1 = observed and normal
0 = missing/corrupted
```

**Masking applies only to input X; target Y remains complete. This is a paper requirement.**

---

## 3. Model Architecture

### 3.1 Overview

```text
Input sequence:  {P_1, P_2, ..., P_T_in}
Masks:           {M_1, M_2, ..., M_T_in}
     │
     ▼
┌─────────────────────────────────────────────┐
│           Patch Embedding                    │
│  (Conv2d: patch_size × patch_size stride)    │
│  Input: (B, C, H, W) → (B, L, embed_dim)     │
└─────────────────────┬───────────────────────┘
                      │
┌─────────────────────────────────────────────┐
│         Encoder (SwinLSTM-I units)           │
│                                              │
│  For each time step t:                        │
│    1. Imputation unit:                        │
│       P_hat_t = σ(Wp*C_{t-1} + Up*H_{t-1}+bp)│
│       P_t = M_t ⊙ P_t + (1-M_t) ⊙ P_hat_t    │
│    2. SwinLSTM update:                        │
│       Ft = Swin(P_t, H_{t-1})                │
│       gate = σ(Ft), cell = tanh(Ft)          │
│       C_t = gate ⊙ (C_{t-1} + cell)          │
│       H_t = gate ⊙ tanh(C_t)                  │
│    3. Patch Merging (downsampling)           │
│                                              │
│  Units: 2 (paper)                             │
└─────────────────────┬───────────────────────┘
                      │  (encoded states)
                      ▼
┌─────────────────────────────────────────────┐
│         Decoder (vanilla SwinLSTM units)      │
│                                              │
│  Same LSTM gating without imputation step.    │
│  Patch Expanding (upsampling) between units.  │
│                                              │
│  Units: 2 (paper)                             │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         Reconstruction Layer                  │
│   (exact-size projection → tanh)             │
│   Output: (Y_1, ..., Y_T_out)                │
│   Each: (B, F, H, W)                         │
└─────────────────────────────────────────────┘
```

### 3.2 Patch Embedding

Each input map is divided into non-overlapping patches of size `patch_size × patch_size` (paper: `patch_size = 2`).

A Conv2d with `kernel_size = patch_size`, `stride = patch_size` projects each patch to `embed_dim` features:

```text
Input:  (B, F, H, W)
→ Conv2d projection
→ Flatten spatial dims
→ LayerNorm
Output: (B, L, embed_dim)  where L = (H/p) × (W/p)
```

**AERPAW note:** With `H=3` and paper patch_size=2, `3 ÷ 2 = 1.5`, so H is not evenly divisible. Rectangular patches (e.g., `[1, 2]`) or padding are required. See §6 and §10.

### 3.3 Encoder: SwinLSTM-I Unit

Each encoder unit is a **SwinLSTM-I cell** that extends the vanilla SwinLSTM cell with an imputation mechanism.

#### Imputation Step

The missing entries are estimated using a linear projection of the previous cell state `C_{t-1}` and hidden state `H_{t-1}`:

```text
P_hat_t = σ(W_p * C_{t-1} + U_p * H_{t-1} + b_p)
```

where `W_p` and `U_p` are linear layers, and `σ` is the sigmoid activation producing outputs in the same normalized range.

#### Imputation Fill

The estimated values replace only the missing entries:

```text
P_t_filled = M_t ⊙ P_t + (1 - M_t) ⊙ P_hat_t
```

where `⊙` is element-wise multiplication. Observed entries pass through unchanged.

#### SwinLSTM Gate Update

The filled map then proceeds through the standard SwinLSTM gating mechanism:

```text
F_t = Swin(P_t_filled, H_{t-1})
gate = σ(F_t)
cell = tanh(F_t)
C_t = gate ⊙ (C_{t-1} + cell)
H_t = gate ⊙ tanh(C_t)
```

The `Swin(·)` function applies a stack of Swin Transformer blocks. Each block alternates between W-MSA (window multi-head self-attention) and SW-MSA (shifted window multi-head self-attention). The hidden state `H_{t-1}` is integrated by concatenation with the input along the feature dimension, followed by a linear reduction (`nn.Linear(2*dim, dim)`).

#### Simplified LSTM Gating

Unlike standard LSTM with separate input, forget, and output gates, the paper uses a **simplified LSTM** with a single gate:

```text
F_t = Swin(P_t_filled, H_{t-1})
gate = sigmoid(F_t)
cell = tanh(F_t)
C_t = gate * (C_{t-1} + cell)
H_t = gate * tanh(C_t)
```

This matches the SwinLSTM repo's `SwinLSTMCell` implementation at `SwinLSTM_D.py:401-420`.

### 3.4 Decoder: Vanilla SwinLSTM Unit

The decoder uses vanilla SwinLSTM units **without** the imputation mechanism. The decoder operates on the encoded states and generates future predictions autoregressively.

The forward pass for decoder cells:

```text
F_t = Swin(H_t_enc, H_{t-1})
gate = σ(F_t), cell = tanh(F_t)
C_t = gate ⊙ (C_{t-1} + cell)
H_t = gate ⊙ tanh(C_t)
```

No imputation needed because the decoder processes only complete (imputed) representations from the encoder or its own previous predictions.

### 3.5 Patch Merging / Patch Expanding

**Patch Merging** (downsampling, from `SwinLSTM_D.py:257-288`):

- Operates on embedded tokens `(B, L, C)` where `L = H × W`.
- Reshapes to `(B, H, W, C)` and concatenates 2×2 neighboring patches.
- Outputs `(B, H/2, W/2, 4*C)` then reduces to `(B, H/2, W/2, 2*C)`.
- Requires `H` and `W` to be even (enforced by assertion).

**Patch Expanding** (upsampling, from `SwinLSTM_D.py:294-321`):

- Expands feature dimension from `C` to `2*C` via linear layer.
- Rearranges `(B, H, W, 2*C)` → `(B, 2*H, 2*W, C/2)` using `einops.rearrange`.

These layers create a hierarchical multi-scale representation in the encoder and progressively restore resolution in the decoder.

### 3.6 Reconstruction Layer

Maps the hidden features back to the original spectrum map dimensions:

- `PatchInflated` (from `SwinLSTM_D.py:357-387`): ConvTranspose2d with kernel 3×3, stride 2, padding 1.
- Applied to the decoder's final hidden state.
- Sigmoid activation produces normalized output in the same range as paper normalization (`[-1, 1]` after rescaling, or `[0, 1]` depending on implementation).

Output shape per time step:

```text
Y_hat_t ∈ R^(B, F, H, W)    →    (B, 1, 3, 250) for AERPAW full mode
```

---

## 4. Output Format

### Raw Tensor

```text
Full mode output:   Y_hat: (T_out, 3, 250, 1)
CC2 smoke output:   Y_hat: (T_out, 1, 250, 1)
```

### CSV Export

```text
Full mode:
    Flattened export: (T_out, 750)
    Columns: CC1[0:250], CC2[250:500], LW1[500:750]

CC2 smoke:
    Flattened export: (T_out, 250)
    Columns: CC2[0:250]
```

### Evaluation Outputs

Must match the project's output convention:

```text
metrics.json              — All computed metrics
predictions.csv           — Denormalized predictions (dBm)
ground_truth.csv          — Denormalized ground truth (dBm)
spectrogram_<node>.png    — Per-node spectrogram comparison
error_analysis.png        — Error distribution and residual plots
```

All plots must use denormalized dBm values.

---

## 5. Training Pipeline

### 5.1 Data Loading

1. Load CSV with `numpy.loadtxt()` or `pandas.read_csv()` — shape `(6839, 750)`.
2. Select node subset if configured (e.g., `["CC2"]` for smoke test).
3. Reshape to pseudo-spectrum-map:
   - Full: `(T, 3, 250, 1)`
   - CC2 smoke: `(T, 1, 250, 1)`
4. Chronological split into train/val/test.
5. Fit normalization statistics on **training split only**.
6. Generate sliding windows:
   - Input: `(T_in, H, W, F)`
   - Target: `(T_out, H, W, F)`

### 5.2 Mask Generation

For each training input window, generate a binary mask:

```text
M_t[i,j] = 1 if observed, 0 if missing/corrupted
```

Strategies (configurable via `missing_strategy`):

- `random`: Each element independently masked with probability `missing_rate`.
- `block`: Contiguous blocks masked (more realistic for spectrum dropout).
- `frequency`: Entire frequency bins masked.
- `node`: Entire sensor nodes masked.

### 5.3 Mask Application

```text
X_masked = M ⊙ X + (1 - M) ⊙ placeholder
```

where `placeholder` can be 0, the normalization minimum, or random noise (configurable). The mask `M` is also passed to the model for the imputation unit.

**Target Y is never masked.**

### 5.4 Training Loop

1. Batch input `X: (B, T_in, H, W, F)` and mask `M: (B, T_in, H, W, F)`.
2. Roll through time steps:
   - For `t = 0 .. T_in-1`: feed `X[:, t]` and `M[:, t]` through encoder.
   - For `t = 0 .. T_out-1`: autoregressive prediction through decoder (no mask).
3. Compute loss between predictions and targets.
4. Backpropagate and update.

### 5.5 Loss Function

Default: **MSE** (mean squared error) — paper default.

Configurable alternatives: `mae`, `huber`, or combined loss.

### 5.6 Paper Default Hyperparameters

```text
T_in               = 10
T_out              = 10
patch_size         = 2
encoder units      = 2
decoder units      = 2
Swin depths        = (2, 6, 6, 2)
epochs             = 400
batch_size         = 4
learning_rate      = 0.0001
optimizer          = Adam
normalization      = MinMax [-1, 1]
train/test split   = 10:2
missing rates      = 0.2 to 0.8
primary metric     = NRMSE(dB)
```

### 5.7 Early Stopping

Monitor validation loss. Configurable patience (default: 30 epochs). Restore best checkpoint.

---

## 6. Assumptions and Design Decisions

1. **SwinLSTM repo is supporting code only.** The repo at `SongTang-x/SwinLSTM` implements vanilla SwinLSTM (B and D variants). It contains `SwinLSTMCell`, `PatchEmbed`, `PatchMerging`, `PatchExpanding`, `PatchInflated`, and Swin Transformer blocks. The imputation unit, encoder-decoder separation with different cell types, mask handling, and masking pipeline must be implemented from scratch.

2. **Paper is the architecture target.** The DSwinLSTM-I paper defines the model. Differences between the repo and the paper are resolved in favor of the paper.

3. **AERPAW is not a true 64×64 spatial grid.** The pseudo-map shape `(nodes, frequency_bins)` is an adaptation, not a direct reproduction. This has implications for patch-based processing (see §7).

4. **Default full mode:** `H=3, W=250, F=1`.

5. **CC2 smoke test:** `H=1, W=250, F=1` for quick CPU verification.

6. **Masking only on input X.** Target Y is always complete. This follows the paper's experiment design.

7. **Train stride and test stride are configurable.** Default: train stride = 1 (maximal overlap for limited data), test stride = prediction_horizon (minimal overlap for independent evaluation).

8. **Evaluation export format** must match all other models in the project (`metrics.json`, `predictions.csv`, `ground_truth.csv`, `spectrogram_*.png`, `error_analysis.png`).

9. **NRMSE(dB) as primary metric.** The paper evaluates on NRMSE in decibel space. This metric is included alongside standard RMSE, MAE, and R².

---

## 7. AERPAW Dataset Adaptation

### Paper vs. Our Data

| Aspect | Paper (DSwinLSTM-I) | AERPAW Adaptation |
|--------|---------------------|-------------------|
| Data source | Simulated spectrum maps | AERPAW measured RSS/PSD |
| Map shape | `64 × 64 × 1` | `3 × 250 × 1` (pseudo-map) |
| Spatial semantics | True 2D spatial grid | Nodes × frequency bins |
| Frequency | Single frequency point | 250 bins per node |
| Time points | 9200 maps | 6839 time steps |
| Time span | ~153 hours | ~114 hours (1-min avg) |
| Normalization | MinMax [-1, 1] | MinMax [-1, 1] |
| Missing rate | 20%–80% | Configurable (default 30%) |
| Train/test | 10:2 (≈83%/17%) | Configurable (default 80/10/10) |
| Input length | 10 | Configurable (default 10) |
| Output length | 10 | Configurable (default 10) |
| Primary metric | NRMSE(dB) | NRMSE(dB) + RMSE/MAE/R² |

### Adaptation Statement

> Our adaptation tests whether the DSwinLSTM-I architecture can operate on AERPAW node-frequency maps, not whether we reproduce the original 64×64 simulated-grid experiment. The node-frequency representation treats each sensor node as a row in a pseudo-image and each frequency bin as a column. This is a deliberate simplification — the model's spatial attention operates on node-frequency adjacency, not physical spatial adjacency.

### Implications of Non-Square Pseudo-Maps

| Issue | Impact |
|-------|--------|
| `H=3` not divisible by `patch_size=2` | Need rectangular patches (e.g., `[1, 2]`) or padding |
| `H=1` (CC2 mode) with 2×2 patches | Single row cannot be divided into 2-row patches |
| `W=250` divisible by 2 | 250 ÷ 2 = 125 patches — fine for width dimension |
| Window size 4 on 3-height maps | Window may exceed spatial dimension; SwinTransformerBlock auto-adjusts when `H ≤ window_size` |
| Patch merging requires even H/W | Patch merging fails when H is odd (H=3) or H=1 |

---

## 8. Repo-Specific Implementation Notes

### Files in `SongTang-x/SwinLSTM`

| File | Contents | Reuse? |
|------|----------|--------|
| `SwinLSTM_B.py` | Single-cell SwinLSTM (`STconvert`, one `SwinLSTMCell`), PatchEmbed, PatchInflated | Partial — cell design reference |
| `SwinLSTM_D.py` | Multi-cell SwinLSTM (`DownSample`, `UpSample`, `PatchMerging`, `PatchExpanding`), multiple `SwinLSTMCell` layers | Primary code reference |
| `configs.py` | Argparse config | Not reusable (use YAML) |
| `dataset.py` | Moving MNIST data loader | Not reusable |
| `functions.py` | Train/eval loops, teacher forcing | Reference for forward pass design |
| `train.py` | Training orchestration | Reference only |
| `test.py` | Evaluation with pretrained weights | Reference only |
| `utils.py` | Metrics (MSE, SSIM), visualization | SSIM reference possible |

### Key Architecture Components in Repo

**Swin Transformer Block** (`SwinLSTM_D.py:142-254`):

- Alternating W-MSA and SW-MSA per layer.
- Hidden state integration: concatenation `[x, hx]` along feature dim → `Linear(2*dim, dim)`.
- Window partition/reverse, cyclic shift, relative position bias.
- **Reusable as-is** for both SwinLSTM and SwinLSTM-I cells.

**SwinLSTMCell** (`SwinLSTM_D.py:390-420`):

- Simplified LSTM with single gate.
- `Ft = Swin(xt, hx)` → `gate = sigmoid(Ft)`, `cell = tanh(Ft)`.
- `cy = gate * (cx + cell)`, `hy = gate * tanh(cy)`.
- **This is the vanilla SwinLSTM cell (no imputation). Requires extension for SwinLSTM-I.**

**PatchEmbed** (`SwinLSTM_D.py:324-354`):

- Conv2d projection `(B, C, H, W)` → `(B, embed_dim, H/p, W/p)` → flatten → LayerNorm → `(B, L, embed_dim)`.
- Uses `to_2tuple(patch_size)` — patch_size converted to 2-element tuple.
- Requires `H` and `W` to match `img_size` exactly (assertion).
- **Reusable, but may need modification for rectangular patches or padding.**

**PatchMerging** (`SwinLSTM_D.py:257-288`):

- Concatenates 2×2 neighboring patches; requires `H` and `W` even.
- `Linear(4*dim, 2*dim)` reduction.
- **Will fail for H=3 (odd) or H=1. Need workaround or skip.**

**PatchExpanding** (`SwinLSTM_D.py:294-321`):

- Linear expand `dim → 2*dim`, then `rearrange` to double spatial dims.
- Output shape `(B, 2*H, 2*W, dim/2)`.
- **Requires input resolution to match; no odd-size assertion but will break if upstream merging was skipped.**

**PatchInflated** (Reconstruction, `SwinLSTM_D.py:357-387`):

- ConvTranspose2d `(B, embed_dim, H, W)` → `(B, in_chans, 2*H, 2*W)`.
- **Requires H and W even (assertion).**

### What Must Be Added

1. **SwinLSTM-I cell** — Extends `SwinLSTMCell` with imputation unit:
   - Linear layers `W_p`, `U_p`, `b_p` for `P_hat = σ(W_p * C_{t-1} + U_p * H_{t-1} + b_p)`.
   - Mask-aware filling: `P_t = M_t ⊙ P_t + (1 - M_t) ⊙ P_hat_t`.
   - The mask `M_t` must be in token space (same `L` dimension as patches) or in pixel space (with matching down-projection).

2. **Encoder module** — Stacks SwinLSTM-I cells with Patch Merging between them.

3. **Decoder module** — Stacks vanilla SwinLSTM cells with Patch Expanding between them.

4. **Full DSwinLSTM-I model** — Combines encoder + decoder + reconstruction layer.

5. **Mask injection pipeline** — How the mask enters the model:
   - Option A (paper): mask feeds into imputation unit only.
   - Option B: mask concatenated as extra input channel.
   - Option C: both.

### Tensor Layouts in Repo

- Input to model: `(B, C, H, W)` — single frame, C == in_chans.
- Internal patch tokens: `(B, L, C_feat)` where `L = H_p * W_p`.
- States `(hx, cx)`: each `(B, L, C_feat)`.
- PatchEmbed output: `(B, L, C_feat)` after flatten and LayerNorm.
- PatchMerging output: `(B, L/4, 2*C_feat)`.
- PatchExpanding output: `(B, L*4, C_feat/2)`.
- Reconstruction output: `(B, C, 2*H_p, 2*W_p)` after ConvTranspose2d.

---

## 9. Known Limitations

1. **AERPAW is sparse in the spatial dimension: only 3 nodes.** Treating node × frequency as a pseudo-image imposes artificial adjacency between frequency bins of the same node. The Swin attention window operates on this adjacency, which may not reflect true physical relationships.

2. **Patch size 2 does not divide H=3 evenly.** The paper uses patch_size=2 for 64×64 maps. With `H=3`, `3 ÷ 2 = 1.5`. Configurable padding or rectangular patches are required. Even with rectangular `[1, 2]` patches, the resulting `H_p = 3, W_p = 125` may cause issues with PatchMerging (which requires even dimensions).

3. **CC2-only mode with `H=1` also does not naturally support 2×2 patches.** A single spatial row cannot be divided into 2-row patches. `[1, 2]` rectangular patches may work, but PatchMerging would still fail on the `H=1` dimension.

4. **PatchMerging and PatchInflated require even `H` and `W`.** Both have assertions `H % 2 == 0 and W % 2 == 0`. The AERPAW pseudo-map `H=3` (odd) will violate these. Options:
   - Pad `H=3` to `H=4` with zeros/reflect padding and crop output.
   - Skip PatchMerging for the first layer (reduce encoder/decoder units).
   - Use `flat_strip` representation `(1, 750)` where `H=1, W=750`, but `W=750` is divisible by 2.

5. **This architecture may be more suitable for true spatial grid maps** (like the paper's 64×64) than for sparse node-frequency tensors. Consider whether SimTSC or TSS-LCD may be a better fit.

6. **CPU training may be very slow.** Swin Transformer self-attention is computationally heavier than ConvLSTM convolutions. The paper uses 400 epochs with batch size 4; CPU training may take hours.

7. **Missing data experiments require standardized masking** to compare fairly with TSS-LCD. Define the same missing rates and strategies.

8. **The repo's PatchInflated layer uses ConvTranspose2d (3×3 kernel, stride 2)**. This upsamples the patch-embedded resolution back to the original. It assumes the patch-embedded resolution is smaller than the original. For `H=3, W=250` with patch `[1,2]`, the embedded resolution is `(3, 125)`. ConvTranspose2d with stride 2 would produce `(6, 250)`, which does not match the original `(3, 250)`. The reconstruction layer needs modification.

---

## 10. Implementation Decisions Already Confirmed

The following design decisions have been resolved and will be hardcoded in the implementation. These are no longer open for debate.

### 10.1 Map Representation

```yaml
map_representation: node_frequency
```

AERPAW data is reshaped to `(T, 3, 250, 1)`, treating the 3 sensor nodes as rows and 250 frequency bins as columns in a pseudo-spectrum map. No alternative representations (`flat_strip`, `artificial_grid`, `padded`) will be implemented.

### 10.2 Patch Merging / Expanding

```yaml
use_patch_merging: false
use_patch_expanding: false
```

Patch merging and expanding are disabled because:

- `H=3` is odd — `PatchMerging` asserts both H and W are even.
- `H=3` is already minimal; downsampling the height dimension would collapse it to `H=2` or `H=1`, losing the 3-node structure.
- The encoder and decoder each stack two SwinLSTMCell layers operating at the same patch-embedded resolution `(3, 125)`.
- Only `PatchEmbed` (at entry) and the reconstruction layer (at exit) change resolution.

This deviates from the paper's DSwinLSTM which uses hierarchical down/up sampling, but is necessary for the AERPAW pseudo-map shape.

### 10.3 Reconstruction Head Design

```yaml
reconstruction: exact_size_linear_or_conv
```

The repo's `PatchInflated` (ConvTranspose2d stride 2) is unsuitable because the embedded resolution `(3, 125)` upsampled by 2× would produce `(6, 250)`, not `(3, 250)`.

The reconstruction head will use **exact-size projection** — either an `nn.Linear` mapping `(B, L, C_feat)` → `(B, L, F)` then reshape to `(B, F, H, W)`, or a Conv2d with `kernel_size=1, stride=1` that preserves spatial dimensions. This guarantees the output matches the target shape `(B, 1, 3, 250)` without cropping or padding.

### 10.4 Mask Handling

```yaml
mask_as_input_channel: false
```

The binary mask is **not** concatenated as an extra input channel. Instead it feeds into the imputation unit of each SwinLSTM-I cell in token space:

1. Input mask `M_t` (pixel space) is aggregated to patches via average pooling to match the patch-embedded token grid `(p_H, p_W)`.
2. The patch-level mask is flattened to `(B, L, 1)` and broadcast to `(B, L, C_feat)` for the imputation linear layers.
3. Inside the cell: `P_hat_t = σ(W_p · C_{t-1} + U_p · H_{t-1} + b_p)`, then `P_t = M_t ⊙ P_t + (1 - M_t) ⊙ P_hat_t`.
4. The filled `P_t` proceeds to the standard SwinLSTM gate update.

Decoder cells receive `mask = None` and skip the imputation step entirely.

### 10.5 Output Activation

```yaml
output_activation: tanh
```

The final reconstruction layer applies `tanh` activation, matching the paper's MinMax normalization to `[-1, 1]`. The loss (MSE) is computed in this normalized space. Denormalization to dBm happens only during evaluation.

### 10.6 Patch Shape

```yaml
patch_shape: [1, 2]
```

Rectangular patches of height 1, width 2 are used because:

- Height 1 is compatible with `H=3` (no remainder, no padding needed).
- Width 2 is compatible with `W=250` (125 patches exactly, no remainder).
- No `reflect` padding required for the height dimension.
- The resulting token grid is `(3, 125)` — both dimensions usable by the Swin Transformer blocks and WindowAttention.

This deviates from the paper's `patch_size=2` (square 2×2) but is the only option that divides both `H=3` and `W=250` evenly.

### 10.7 Sequence Handling

- The model is **sequence-to-sequence**: processes all `T_in` input time steps, then autoregressively predicts `T_out` future time steps.
- Target is the next `T_out` complete maps, not a single map.
- Teacher forcing is used during training (loss computed on all output frames).

---

## 11. Implementation Notes

### `dataset.py` — Data loading, reshaping, masking, windowing

Responsibilities:

- Load CSV via `numpy.loadtxt()` or `pandas.read_csv()`.
- Select node subset (filter columns by configured node names).
- Reshape to pseudo-spectrum-map format `(T, H, W, F)`.
- Chronological train/val/test split.
- Fit normalization statistics on training split only.
- Generate sliding windows of `(T_in, H, W, F)` inputs and `(T_out, H, W, F)` targets.
- Apply missing/corruption mask to input windows (not targets).
- Batch `(X, M, Y)` tuples.

Mask generation functions:

- `random_mask(T, H, W, rate)` → element-wise Bernoulli mask.
- `block_mask(T, H, W, rate)` → contiguous block mask.
- `frequency_mask(T, H, W, rate)` → mask whole frequency bins.
- `node_mask(T, H, W, rate)` → mask whole sensor nodes.

### `model.py` — SwinLSTM, SwinLSTM-I, encoder-decoder, patch layers, reconstruction

Classes to implement (in order of dependency):

1. `SwinTransformerBlock` — Copy from repo `SwinLSTM_D.py` (WindowAttention, Mlp, window_partition, window_reverse, SwinTransformerBlock). These are well-tested components.

2. `SwinLSTMCell` — Based on repo's `SwinLSTMCell` at `SwinLSTM_D.py:390-420`. Simplified LSTM with Swin Transformer.

3. `SwinLSTMCellI` — **The imputation-augmented version.** Extends SwinLSTMCell:
   - Extra linear layers `W_p`, `U_p`, `b_p` for imputation.
   - Forward: `(xt, mask, hidden_states)` → fill missing → SwinLSTM gate → `(hy, (hy, cy))`.

4. `PatchEmbed` — Copy from repo `SwinLSTM_D.py:324-354` with `patch_size=(1, 2)` for rectangular patches.

5. `Encoder` — Stacks 2 `SwinLSTMCellI` cells at the same resolution. No PatchMerging between them.

6. `Decoder` — Stacks 2 vanilla `SwinLSTMCell` cells at the same resolution. No PatchExpanding between them.

7. `Reconstruction` — Exact-size projection (`nn.Linear` or `Conv2d 1×1`) mapping `(B, L, C_feat)` → `(B, F, H, W)` with `tanh` activation.

8. `DSwinLSTM_I` — Full model: PatchEmbed → Encoder → Decoder → Reconstruction → tanh. Supports two decoder feedback modes:
   - `hidden_state` (default): decoder cell's own `hx` is fed as the next input token. No pixel round-trip.
   - `pixel_feedback`: reconstruct `y_hat`, then `patch_embed(y_hat)` back to token space. More expensive but may improve stability.

### `train.py` — Training loop

- Parse config.
- Initialize dataset, mask generator, model, optimizer, criterion.
- Training loop with teacher forcing.
- Early stopping based on validation loss.
- Save checkpoints.

Forward pass logic with `decoder_feedback: hidden_state`:

```text
# Encoder: impute and encode each input time step
for t in range(T_in):
    hx, cx = encoder_cell(x_t, mask_t, (hx, cx))

# Decoder: autoregressive prediction (no masks)
pred = []
hx, cx = hx, cx     # carry last encoder state to decoder
for t in range(T_out):
    hx, cx = decoder_cell(hx, (hx, cx))   # no mask, feedback = hidden state
    y_hat = reconstruction(hx)
    pred.append(y_hat)
```

**Shape rationale:** `hx` is token-space `(B, L, C_feat)` throughout — the decoder cell consumes and produces the same token-space representation. The reconstruction layer only runs to produce the final pixel-space output `y_hat` for the loss/metrics; `y_hat` is never fed back into the decoder. This avoids a redundant patch-embed round-trip at every autoregressive step.

With `decoder_feedback: pixel_feedback` (alternative):

```text
    ...
    y_hat = reconstruction(hx)
    pred.append(y_hat)
    hx = patch_embed(y_hat)   # re-embed pixel output back to token space
```

### `evaluate.py` — Metrics/plots/exports

- Load checkpoint and normalization stats.
- Run evaluation on test split.
- Compute metrics: RMSE, MAE, R², NRMSE(dB).
- Denormalize predictions and ground truth to dBm.
- Export `metrics.json`, `predictions.csv`, `ground_truth.csv`.
- Generate spectrogram and error-analysis plots.

### `inference.py` — Standalone prediction

- Load checkpoint, normalization stats, and new CSV.
- Apply same preprocessing as training.
- Run model forward.
- Export denormalized predictions as CSV.

### `utils.py` — Shared utilities

- `load_config()` — Load and validate YAML config.
- `set_seed()` — Set random seeds for reproducibility.
- `normalize()` / `denormalize()` — MinMax [-1, 1] transform.
- `compute_metrics()` — Compute RMSE, MAE, R², NRMSE(dB) per-node and per-frequency.
- `plot_spectrogram()` — Generate spectrogram comparison plots.
- `save_checkpoint()` / `load_checkpoint()` — Model persistence.

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
