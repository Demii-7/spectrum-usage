# STS-PredNet Spectrum Prediction — Integrated Pipeline

> **Based on:** *Deep Learning for Spectrum Prediction From Spatial–Temporal–Spectral Data* — Li, Liu, Chen, Xu, Song (IEEE Communications Letters, 2020)
>
> **Conceptual reference:** https://github.com/Demii-7/pred-rnn (incomplete, treat as rough reference only)
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

**This is a paper reconstruction, not an adaptation of a pretrained model.** The architecture is rebuilt from scratch in PyTorch based on the STS-PredNet and PredRNN paper equations.

---

## Quick Start

```bash
python3 training/STS-PredNet/train_integrated.py
```

Outputs go to `training/results/STS-PredNet/` by default. Override with `--output-dir`:

```bash
python3 training/STS-PredNet/train_integrated.py \
    --output-dir /path/to/custom_output
```

---

## Scripts Reference

### `train_integrated.py` — Train and evaluate per chunk

The integrated runner trains one model per 200 MHz chunk. It uses closeness + period branches (trend disabled) with recursive multi-step prediction.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `training/common/config.yaml` | Shared config |
| `--output-dir` | `training/results/STS-PredNet/` | Output directory |

Outputs:

```
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
models/<chunk_id>_stsprednet.pt
<chunk_id>_training_log.csv
```

### `stsprednet.py` — Main STS-PredNet model

Combines closeness and period PredRNN branches with learnable weighted fusion.

### `predrnn.py` — Single PredRNN branch

Stacks STS-ConvLSTM cells with zigzag memory flow across layers and time steps.

The `output_proj` 1×1 convolution projects the top-layer hidden state to `output_channels` (default 1 for single-channel CSV mode; configurable for multi-channel map mode).

### `sts_convlstm_cell.py` — STS-ConvLSTM (ST-LSTM) cell

Implements the SpatioTemporal LSTM cell with three states: `H` (hidden), `C` (cell), `M` (unified spatiotemporal memory).

---

## 1. What the Model Is Intended to Do

STS-PredNet performs spectrum prediction from spatial-temporal-spectral data. The model predicts future spectrum maps by capturing three temporal properties:

- **Closeness** — recent consecutive observations
- **Daily period** — same time from previous days (1-day interval)
- **Weekly trend** — same time from previous weeks (7-day interval, disabled in our setup)

For the integrated pipeline, each spectrum map is a single-node 200 MHz chunk `(1, 200)`.

**Key difference from paper:** The paper predicts a single future step from closeness + period + trend branches using a `(100, 60)` spatial-spectral map with 4 sensors. Our integrated version:
- Uses closeness + period only (trend unavailable — only 4.75 days of data)
- Predicts recursively for multi-step horizons
- Trains per 200 MHz chunk on a single node (CC2)

---

## 2. Input Format

### 2.1 Per-Chunk Data Loading

The integrated pipeline uses `load_chunk()` to extract a single-node 200 MHz chunk from CC2's per-minute CSV:

```python
data = load_chunk(config, chunk)
train_input = data.splits[data.train_split].model_input   # (T_train, 200)
```

### 2.2 Temporal Branch Inputs

#### Closeness branch (`lc = 36`)

Recent consecutive observations:

```text
S_c = {X_{t-(lc-1)}, ..., X_t}   shape (lc, 200)
```

#### Period branch (`lp = 3`, `period_interval = 1440`)

Daily-period observations at the same time from previous days:

```text
S_p = {X_{t+Δt-lp·1440}, ..., X_{t+Δt-1440}}   shape (lp, 200)
```

#### Trend branch (disabled)

Configurable with `lq = 4`, `trend_interval = 10080`. Not feasible because the AERPAW CSV has only 6839 minutes (< 1 week).

### 2.3 Normalization

Applied by `load_chunk()` before training. The shared pipeline uses z-score normalization per frequency bin, not the paper's min-max scaling to `[-1, 1]`. Metrics are reported in denormalized dBm.

---

## 3. Model Architecture

### 3.1 Overview

```
Chunk data: (T, 200)
       │
       ▼
Two temporal branches:
    Closeness PredRNN ─→ X_c  (B, 1, 1, 200)
    Period PredRNN     ─→ X_p  (B, 1, 1, 200)
       │
       ▼
Learnable weighted fusion:
    W_c ⊙ X_c + W_p ⊙ X_p
       │
       ▼
Output: (B, 1, 200)  →  single-step prediction
```

### 3.2 STS-ConvLSTM Cell

Each PredRNN branch uses stacked STS-ConvLSTM (ST-LSTM) units with three states:

| State | Description |
|-------|-------------|
| `H` | Hidden state |
| `C` | Standard LSTM cell state |
| `M` | Unified spatiotemporal memory |

**Cell equations (reconstructed from PredRNN):**

```
g  = tanh(W_xg ∗ X_t + W_hg ∗ H_{t-1} + b_g)
i  = sigmoid(W_xi ∗ X_t + W_hi ∗ H_{t-1} + b_i)
f  = sigmoid(W_xf ∗ X_t + W_hf ∗ H_{t-1} + b_f)
C_t = f ⊙ C_{t-1} + i ⊙ g

g' = tanh(W_xg' ∗ X_t + W_mg ∗ M_{t-1} + b_g')
i' = sigmoid(W_xi' ∗ X_t + W_mi ∗ M_{t-1} + b_i')
f' = sigmoid(W_xf' ∗ X_t + W_mf ∗ M_{t-1} + b_f')
M_t = f' ⊙ M_{t-1} + i' ⊙ g'

o  = sigmoid(W_xo ∗ X_t + W_ho ∗ H_{t-1} + W_co ∗ C_t + W_mo ∗ M_t + b_o)
H_t = o ⊙ tanh(W_{1×1} ∗ [C_t, M_t])
```

where `∗` = convolution, `⊙` = element-wise multiplication, `[, ]` = concatenation.

### 3.3 PredRNN Branch

| Parameter | Value |
|-----------|-------|
| `num_layers` | 4 |
| `hidden_dim` | 128 per layer |
| `kernel_size` | `[3, 3]` |
| `output_channels` | 1 (default; configurable for multi-channel maps) |

Memory flows in a zigzag pattern:
- Layer 0 receives `M` from the last layer of the *previous* time step
- Layers 1..L-1 receive `M` from the previous layer of the *current* time step
- Output is the hidden state of the top layer after all frames
- The 1×1 `output_proj` convolution maps this to `(B, output_channels, H, W)`

### 3.4 Fusion Layer

```text
Y_pred = W_c ⊙ out_c + W_p ⊙ out_p
```

where `W_c, W_p` are learnable parameters of shape `(1, 1, 1, 200)` broadcastable over batch and channel.

### 3.5 Recursive Prediction (Integrated Extension)

Since our shared pipeline evaluates multiple horizons (1, 5, 15, 60 min), the integrated runner performs recursive single-step prediction:

1. Predict `Y_{t+1}` from closeness + period
2. Append `Y_{t+1}` to running history
3. Shift closeness window forward
4. Repeat for `h` steps

This is not described in the original paper (which predicts a single future step), but is necessary for multi-horizon evaluation.

---

## 4. Output Format

```
Per step:   (B, 1, 200)    — single predicted spectrum map
Squeezed:   (B, 200)       — frequency bin predictions
Denormalized: dBm via inverse z-score
```

---

## 5. Training Pipeline

### 5.1 Data Loading

The `STSPredNetDataset` constructs closeness and period sequences on-the-fly:

- Closeness: last `lc` consecutive maps ending at `target_idx - 1`
- Period: `lp` maps at `period_interval` steps back from `target_idx`
- Target: the map at `target_idx`

### 5.2 Loss, Optimizer, Training

| Parameter | Value | Source |
|-----------|-------|--------|
| Loss | MSE | Same as paper |
| Optimizer | Adam | Same as paper |
| Learning rate | 0.0002 | Same as paper |
| Batch size | 32 | Same as paper (configurable) |
| Epochs | 25 | Configurable (paper uses 500) |
| Early stopping | Patience 30 on val loss | Our addition |
| Weight decay | 0.0 | Configurable |

### 5.3 Evaluation

For each horizon `h`:
1. Identify target rows with sufficient history
2. Recursively predict `h` steps using the trained model
3. Compare with ground truth (denormalized dBm)
4. Write per-horizon, per-frequency, per-band metrics

---

## 6. Changes from Main Branch (Standalone STS-PredNet)

| Aspect | Main Branch | Integrate Branch |
|--------|------------|------------------|
| **Data loading** | Custom dataset.py, 750-col CSV → `(T, 3, 250)` maps | Shared `load_chunk()`, single CC2 node, `(T, 200)` per chunk |
| **Branches** | closeness + period + trend (all three) | closeness + period only (trend disabled) |
| **Prediction** | Single-step (paper-faithful) | Recursive multi-step for horizons 1/5/15/60 |
| **Normalization** | Min-max `[-1, 1]` (paper-faithful) | Z-score via `load_chunk()` |
| **Output activation** | tanh (paper-faithful) | tanh (unchanged) |
| **Config** | Standalone `config.yaml` | Shared `config.yaml`, `stsprednet:` section |
| **Epochs** | 500 (paper-faithful) | 25 (reduced; converges faster on single-node chunks) |
| **Output channels** | 1 (always) | 1 (default); `output_channels` parameter available for multi-channel maps |

### Rationale for Changes

- **Trend disabled:** The AERPAW dataset has only 6839 minutes (< 1 week). Trend requires 4 × 10080 = 40320 minutes of history.
- **Reduced `lp` from 7 to 3:** Paper default `lp = 7` needs 7 × 1440 = 10080 minutes of history (unavailable). `lp = 3` needs 4320 minutes, which fits.
- **Recursive prediction:** The shared evaluation framework reports per-horizon metrics. STS-PredNet natively predicts one step ahead, so we extend it recursively.
- **Z-score normalization:** Using the shared pipeline's normalization ensures fair comparison across models. The paper's min-max `[-1, 1]` can be added back as a config option if needed.

---

## 7. Configuration Reference

All STS-PredNet settings are under `stsprednet:` in `training/common/config.yaml`:

| Field | Default | Description |
|-------|---------|-------------|
| `lc` | 36 | Closeness sequence length |
| `lp` | 3 | Period sequence length (reduced from paper's 7) |
| `lq` | 4 | Trend sequence length (paper default; disabled) |
| `period_interval` | 1440 | Minutes between period samples (1 day) |
| `trend_interval` | 10080 | Minutes between trend samples (1 week; unused) |
| `batch_size` | 32 | Training batch size |
| `epochs` | 25 | Max epochs |
| `learning_rate` | 0.0002 | Adam learning rate |
| `weight_decay` | 0.0 | L2 weight decay |
| `gradient_clip_norm` | 5.0 | Max gradient norm |
| `patience` | 30 | Early stopping patience |
| `model.input_channels` | 1 | Input channel count |
| `model.map_height` | 1 | Spatial height (single node) |
| `model.hidden_dim` | 128 | Hidden dimension per layer |
| `model.num_layers` | 4 | STS-ConvLSTM layers |
| `model.kernel_size` | [1, 3] | Convolution kernel size |
| `model.output_activation` | tanh | Output activation |
| `model.fusion_weight_shape` | per_location | Fusion weight shape |
| `model.output_channels` | 1 | PredRNN output channels (1 for single-channel CSV, >1 for multi-channel maps) |
| `interpolated_map.enabled` | false | Enable interpolated .npz map input mode |
| `interpolated_map.map_path` | — | Path to .npz file with grid map data |
| `interpolated_map.map_key` | map_db | Key inside .npz for the 4D array |
| `interpolated_map.temporal_overrides` | — | Override lc/lp/period_interval for short map datasets |

---

## 8. Known Limitations

1. **Weekly trend branch unavailable** — requires > 1 week of data (40320 min)
2. **Period branch requires reduced `lp`** — paper's `lp = 7` would need 10080 min of history; reduced to `lp = 3` (needs 4320 min)
3. **Recursive prediction accumulates error** — multi-step evaluation uses recursive roll-out; errors compound at longer horizons
4. **Small spatial dimension** — with height=1, `3×3` kernels are effectively `1×3` in practice
5. **Computational cost** — two independent PredRNN branches ≈ 2× the cost of a single branch

---

## References

1. **STS-PredNet paper:** Li, X., Liu, Z., Chen, G., Xu, Y., & Song, T. (2020). *Deep Learning for Spectrum Prediction From Spatial–Temporal–Spectral Data.* IEEE Communications Letters, 25(4), 1216–1220. DOI: [10.1109/LCOMM.2020.3045205](https://doi.org/10.1109/LCOMM.2020.3045205)
2. **PredRNN paper:** Wang, Y., Long, M., Wang, J., Gao, Z., & Yu, P. S. (2017). *PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs.* NIPS 2017.
3. **AERPAW dataset:** DOI: [10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn)
