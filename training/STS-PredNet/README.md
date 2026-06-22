# STS-PredNet Spectrum Prediction — Reconstructed Model

> **Based on:** *Deep Learning for Spectrum Prediction From Spatial–Temporal–Spectral Data* — Li, Liu, Chen, Xu, Song (IEEE Communications Letters, 2020)
>
> **Conceptual reference:** https://github.com/Demii-7/pred-rnn (incomplete, treat as rough reference only)
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

**This is a paper reconstruction, not an adaptation of a pretrained model.** The architecture is rebuilt from scratch in PyTorch based on the STS-PredNet and PredRNN paper equations. The provided `pred-rnn` repo is incomplete and should be treated only as a conceptual reference for ST-LSTM cell design.

---

## Quick Start

The following commands are proposed — scripts do not exist yet and will be created in a future step.

### Setup

```bash
cd /home/cc/spectrum-usage
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib pyyaml scikit-learn tqdm
```

### Train the Model

```bash
# Default config (closeness + period branches, 80/10/10 split)
python3 training/STS-PredNet/train.py

# With overrides
python3 training/STS-PredNet/train.py \
    --batch-size 64 \
    --epochs 300 \
    --lr 0.0001 \
    --lp 5
```

Training creates `training/STS-PredNet/checkpoints/` with `best_model.pt`, `last_model.pt`, and `normalization_stats.pt`.

### Evaluate

```bash
python3 training/STS-PredNet/evaluate.py \
    --checkpoint training/STS-PredNet/checkpoints/best_model.pt
```

Output: per-node and per-frequency-bin RMSE/MAE/R², spectrogram plots, and `predictions.csv`.

### Run Inference on New Data

```bash
python3 training/STS-PredNet/inference.py \
    --checkpoint training/STS-PredNet/checkpoints/best_model.pt \
    --input /path/to/new_measurements.csv \
    --output predictions.csv
```

---

## Scripts Reference

### `dataset.py` — Load and prepare AERPAW spectrum data

| Responsibility | Description |
|----------------|-------------|
| Load CSV | Read `(T, 750)` raw dBm values |
| Reshape | Convert to `(T, 3, 250)` spectrum maps |
| Normalize | Min-max normalization to `[-1, 1]` (fit on training split) |
| Split | Chronological train/val/test split |
| Generate branches | Sliding-window construction of closeness, period, trend sequences |
| Batch | Yield `(S_c, S_p, S_q, target)` tuples with configurable branch usage |

### `sts_convlstm_cell.py` — STS-ConvLSTM cell

Implements the SpatioTemporal LSTM (ST-LSTM) cell from the PredRNN paper.

State: `(H, C, M)` — hidden, cell, and unified spatiotemporal memory.

Equations based on:

* Standard cell gates for `C` (input, forget, cell update)
* Memory gates for `M` (memory input, memory forget)
* Output gate using `X_t`, `H_{t-1}`, `C_t`, `M_t`
* Concatenation of `[C_t, M_t]` → `1×1` conv → `tanh` → multiplied by output gate

### `predrnn.py` — PredRNN branch

| Config | Default |
|--------|---------|
| `num_layers` | 4 |
| `hidden_dim` | 128 |
| `kernel_size` | `[3, 3]` |

Each branch stacks STS-ConvLSTM cells with zigzag memory flow:

* Layer 0 receives `M` from last layer of previous time step
* Subsequent layers receive `M` from previous layer of *this* time step
* Output is the hidden state of the top layer at the final time step

### `stsprednet.py` — Main STS-PredNet model

Combines three PredRNN branches (closeness, period, trend) with a learnable weighted fusion layer.

Fusion equation:

```text
X_fused = W_c ⊙ X_c + W_p ⊙ X_p + W_q ⊙ X_q
```

where `W_c`, `W_p`, `W_q` are learnable parameters broadcastable over `(H, W)`.

Output activation: `tanh` (default) or `linear`.

### `train.py` — Training entrypoint

```bash
python3 training/STS-PredNet/train.py [--config CONFIG] [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration |
| `--batch-size` | from config | Override batch size |
| `--epochs` | from config | Override max epochs |
| `--lr` | from config | Override learning rate |
| `--use-trend` | from config | Enable weekly trend branch |

Output: `checkpoints/best_model.pt`, `checkpoints/last_model.pt`, `checkpoints/normalization_stats.pt`, `checkpoints/training_log.json`.

### `evaluate.py` — Evaluate a trained model

```bash
python3 training/STS-PredNet/evaluate.py --checkpoint CHECKPOINT [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` checkpoint (required) |
| `--config` | from checkpoint | Path to config (overrides embedded config) |
| `--output` | `evaluation/` | Output directory for metrics, plots, and CSVs |

Output: `evaluation/metrics.json`, `evaluation/predictions.csv`, `evaluation/ground_truth.csv`, `evaluation/spectrogram_*.png`, `evaluation/per_node_rmse.png`, `evaluation/per_frequency_rmse.png`.

### `inference.py` — Predict on new CSV data

```bash
python3 training/STS-PredNet/inference.py \
    --checkpoint CHECKPOINT --input CSV [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` checkpoint (required) |
| `--input` | — | Path to new CSV file (required) |
| `--output` | `predictions.csv` | Output CSV path |
| `--config` | from checkpoint | Path to config |

Output: CSV with same `750` column layout as input (CC1[0:250], CC2[250:500], LW1[500:750]).

### `utils.py` — Shared utilities

| Function | Description |
|----------|-------------|
| `load_config()` | Load and validate YAML config |
| `set_seed()` | Set random seeds for reproducibility |
| `denormalize()` | Convert `[-1, 1]` predictions back to dBm |
| `compute_metrics()` | Compute RMSE, MAE, R² per-node and per-frequency |
| `plot_spectrogram()` | Generate spectrogram comparison plots |

---

## File Structure

```
training/STS-PredNet/
├── README.md              # This file
├── config.yaml            # Proposed configuration
├── dataset.py             # Data loading and branch generation
├── sts_convlstm_cell.py   # STS-ConvLSTM cell (ST-LSTM)
├── predrnn.py             # Single PredRNN branch
├── stsprednet.py          # Main STS-PredNet model with fusion
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Inference script
├── utils.py               # Shared utilities
├── checkpoints/           # Created during training
└── evaluation/            # Created during evaluation
```

---

## Configuration Reference

See `config.yaml` for the full configuration. Key fields:

| Section | Field | Default | Description |
|---------|-------|---------|-------------|
| `branches` | `use_closeness` | `true` | Enable closeness branch |
| `branches` | `use_period` | `true` | Enable daily period branch |
| `branches` | `use_trend` | `false` | Enable weekly trend branch (requires >1 week of data) |
| `branches` | `lc` | `36` | Closeness sequence length |
| `branches` | `lp` | `3` | Period sequence length (paper default 7; reduced to 3 for AERPAW) |
| `branches` | `lq` | `4` | Trend sequence length |
| `branches` | `period_interval` | `1440` | Minutes between period samples (1 day at 1-min resolution) |
| `branches` | `trend_interval` | `10080` | Minutes between trend samples (1 week at 1-min resolution) |
| `branches` | `share_branch_weights` | `false` | Share PredRNN weights across branches |
| `model` | `num_layers` | `4` | STS-ConvLSTM layers per branch |
| `model` | `hidden_dim` | `128` | Hidden dimension per layer |
| `model` | `kernel_size` | `[3, 3]` | Convolution kernel size |
| `model` | `output_activation` | `tanh` | Output activation (`tanh` or `linear`) |
| `training` | `batch_size` | `32` | Training batch size |
| `training` | `epochs` | `500` | Maximum epochs |
| `training` | `learning_rate` | `0.0002` | Adam learning rate |
| `training` | `loss` | `mse` | Loss function |

---

## 1. What the Model Is Intended to Do

STS-PredNet performs spectrum prediction from spatial-temporal-spectral data. The task is signal-power-level prediction: given historical spectrum measurements across multiple sensor nodes and frequency bins, the model predicts future spectrum maps.

The original paper (Li et al., 2020) proposes an end-to-end deep learning model that predicts the future spectrum map `X_{T+Δt}` from historical maps by capturing three temporal properties:

* **Closeness** — recent observations
* **Daily period** — same time from previous days
* **Weekly trend** — same time from previous weeks

For AERPAW, each spectrum map is represented as `(3 nodes, 250 frequency bins)` corresponding to the three fixed nodes CC1, CC2, and LW1.

---

## 2. Input Format

### 2.1 Raw CSV Format

The AERPAW CSV has shape:

```text
(6839, 750)
```

where:

```text
750 = 3 nodes × 250 frequency bins
```

Columns 0–249 correspond to CC1, 250–499 to CC2, and 500–749 to LW1.

### 2.2 CSV → Spectrum Map Conversion

```text
(6839, 750)
↓
(6839, 3, 250)
```

Each time step becomes a spectrum map:

```text
X_t ∈ R^(N×F)   where N = 3, F = 250
```

For model input, add a channel dimension:

```text
(B, T_branch, C, H, W)
```

where:

| Dim | Value | Meaning |
|-----|-------|---------|
| B | batch size | Number of samples |
| T_branch | lc / lp / lq | Branch sequence length |
| C | 1 | Input channel (power) |
| H | 3 | Spatial nodes |
| W | 250 | Frequency bins |

### 2.3 Temporal Branch Inputs

#### Closeness branch

Recent consecutive observations:

```text
S_c = {X_{t-(lc-1)}, ..., X_t}
```

Default: `lc = 36`

#### Period branch

Daily-period observations at the same time from previous days:

```text
S_p = {X_{t+Δt-lp·p}, ..., X_{t+Δt-p}}
```

Default: `lp = 3` (reduced from paper's `lp = 7` because AERPAW has only `6839` minutes). With `period_interval = 1440`, `lp = 3` needs `3 × 1440 = 4320` minutes of history, which fits within `6839`. The paper's default `lp = 7` would require `10080` minutes (> 1 week) and is not feasible with the current dataset.

#### Trend branch

Weekly-trend observations from previous weeks:

```text
S_q = {X_{t+Δt-lq·q}, ..., X_{t+Δt-q}}
```

Default: `lq = 4`, `trend_interval = 10080` (1 week at 1-minute resolution)

**Important AERPAW limitation:** The available CSV has only `6839` minutes. This is less than one week (`10080` minutes), so the weekly trend branch is not feasible. It also limits the period branch: the paper's default `lp = 7` requires `10080` minutes of history, which is unavailable. The period branch is feasible only with a reduced `lp` (e.g., `lp = 3` requires `4320` minutes). Branch usage is configurable:

```yaml
use_closeness: true
use_period: true    # feasible with reduced lp (from paper's lp=7)
use_trend: false    # not feasible (< 1 week of data)
```

---

## 3. Model Architecture

### 3.1 Overview

```text
AERPAW CSV
↓
Spectrum maps X_t = (3, 250)
↓
Three temporal branches:
    Closeness PredRNN ─→ X_c
    Period PredRNN     ─→ X_p
    Trend PredRNN      ─→ X_q
↓
Learnable weighted fusion:
    W_c ⊙ X_c + W_p ⊙ X_p + W_q ⊙ X_q
↓
Output activation (tanh or linear)
↓
Predicted spectrum map  X_{t+Δt} ∈ (3, 250)
```

### 3.2 STS-ConvLSTM Cell

Each PredRNN branch uses stacked STS-ConvLSTM (ST-LSTM) units. Unlike standard ConvLSTM which carries only `(H, C)`, STS-ConvLSTM carries three states:

| State | Description |
|-------|-------------|
| `H` | Hidden state |
| `C` | Standard LSTM cell state |
| `M` | Unified spatiotemporal memory state |

The unified memory `M` flows in a zigzag pattern: vertically across layers at each time step, and horizontally across time steps for the first layer.

#### Equations

The cell implements the following (reconstructed from the PredRNN paper):

**Standard cell gates for C:**

```text
g  = tanh(W_xg * X_t + W_hg * H_{t-1} + b_g)
i  = sigmoid(W_xi * X_t + W_hi * H_{t-1} + b_i)
f  = sigmoid(W_xf * X_t + W_hf * H_{t-1} + b_f)
C_t = f ⊙ C_{t-1} + i ⊙ g
```

**Memory gates for M:**

```text
g' = tanh(W_xg' * X_t + W_mg * M_{t-1} + b_g')
i' = sigmoid(W_xi' * X_t + W_mi * M_{t-1} + b_i')
f' = sigmoid(W_xf' * X_t + W_mf * M_{t-1} + b_f')
M_t = f' ⊙ M_{t-1} + i' ⊙ g'
```

**Output gate:**

```text
o  = sigmoid(W_xo * X_t + W_ho * H_{t-1} + W_co * C_t + W_mo * M_t + b_o)
H_t = o ⊙ tanh(W_{1×1} * [C_t, M_t])
```

where `*` denotes convolution, `⊙` denotes element-wise multiplication, and `[, ]` denotes concatenation along the channel dimension.

### 3.3 PredRNN Branch

Each branch is a PredRNN with:

| Parameter | Value |
|-----------|-------|
| `num_layers` | 4 |
| `hidden_dim` | 128 |
| `kernel_size` | `[3, 3]` |

The zigzag memory flow works as follows:

* At the first time step: all states are initialized to zeros
* Layer 0: receives `M` from the last layer of the *previous* time step (horizontal zigzag)
* Layers 1..L-1: receive `M` from the previous layer of the *current* time step (vertical flow)
* The output is the hidden state `H` of the top layer after processing all input frames
* The final hidden state is passed through a `1×1` convolution to produce the branch output

The three branches (closeness, period, trend) have the same architecture but independent weights by default. Set `share_branch_weights: true` to share weights across branches.

### 3.4 Fusion Layer

The paper fuses branch outputs with learnable weights:

```text
X_fused = W_c ⊙ X_c + W_p ⊙ X_p + W_q ⊙ X_q
```

where `W_c`, `W_p`, and `W_q` are learnable parameters of shape `(1, 1, H, W)` (broadcastable over batch and channel). This allows the model to learn different importance weights for closeness, period, and trend at different node-frequency locations.

If a branch is disabled (e.g., trend), its term is excluded from the fusion.

### 3.5 Output Activation

The paper normalizes values to `[-1, 1]` and applies `tanh` activation. Default:

```yaml
output_activation: tanh
```

For regression experiments where `tanh` may limit performance, use:

```yaml
output_activation: linear
```

### 3.6 Prediction Target

The paper predicts a single future spectrum map `X_{T+Δt}` for a configurable prediction range. Target shape:

```text
(B, 1, 3, 250)
```

or squeezed:

```text
(B, 3, 250)
```

Multi-step prediction (predicting multiple future time steps) may be added as an implementation extension but is not part of the paper-default setup.

---

## 4. Output Format

### Raw Tensor

```text
Shape: (B, 1, 3, 250)  or  (B, 3, 250)
Values: normalized to [-1, 1] (if tanh output) or raw (if linear output)
```

### Denormalization

Convert from `[-1, 1]` back to dBm using the min/max statistics saved during training:

```python
X_dBm = 0.5 * (X_norm + 1) * (max_val - min_val) + min_val
```

### CSV Output

Same layout as input CSV — 750 columns:

```text
CC1[0:250], CC2[250:500], LW1[500:750]
```

### Visualization

Spectrogram comparison plots showing predicted vs. ground truth for each node and frequency bin.

---

## 5. Training Pipeline

### 5.1 Data Loading

Load the CSV file with `numpy.loadtxt()` or `pandas.read_csv()`. The data has shape `(T, 750)` and is reshaped to `(T, 3, 250)`.

### 5.2 Normalization

The paper uses min-max normalization to `[-1, 1]`:

```python
X_norm = 2.0 * (X - min_val) / (max_val - min_val) - 1.0
```

Min/max are computed from the training split only to avoid data leakage. Configurable alternatives:

| Method | Description |
|--------|-------------|
| `minmax_neg1_pos1` | Paper default, scales to `[-1, 1]` |
| `zscore` | Zero-mean unit-variance |
| `none` | Raw dBm values |

### 5.3 Train / Validation / Test Split

Chronological split (required for time series):

| Split | Ratio | Usage |
|-------|-------|-------|
| Training | 80% | Model training |
| Validation | 10% | Early stopping and checkpoint selection |
| Test | 10% | Final evaluation |

Split is applied before branch sample generation to prevent look-ahead leakage.

### 5.4 Training Instance Generation

For each target time index in the training set, construct:

* `S_c`: closeness sequence — last `lc` consecutive maps
* `S_p`: period sequence — `lp` maps at `period_interval` steps back
* `S_q`: trend sequence — `lq` maps at `trend_interval` steps back
* `Y`: target map — the map at `t + Δt` ahead

A branch is only constructed if its history exists (i.e., there are enough preceding time steps). The target must also fall within the available data range.

For the AERPAW dataset with `6839` time steps:

* Closeness branch: always feasible (requires only `lc` recent steps)
* Period branch: feasible with reduced `lp`. Paper default `lp = 7` needs `7 × 1440 = 10080` min (unavailable). With `lp = 3` needs `3 × 1440 = 4320` min (feasible for targets with ≥ 4320 min of history).
* Trend branch: **not feasible** (requires ≥ `lq * trend_interval = 4 * 10080 = 40320` steps of history)

### 5.5 Loss Function

Default: **MSE** (mean squared error)

```yaml
loss: mse
```

### 5.6 Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Betas | (0.9, 0.999) |

### 5.7 Batch Size

Default: `32`. Configurable because the AERPAW shape `(3, 250)` with hidden dimension 128 may differ in GPU memory requirements from the paper's original `(100, 60)` maps.

### 5.8 Epochs

Default: `500` with early stopping (patience: 30 epochs based on validation loss).

### 5.9 Evaluation Metrics

| Metric | Description | Notes |
|--------|-------------|-------|
| RMSE | Root mean squared error | Primary metric |
| MAE | Mean absolute error | |
| R² | Coefficient of determination | |
| Per-node RMSE | RMSE for each of 3 nodes | Diagonal of spatial structure |
| Per-frequency-bin RMSE | RMSE for each of 250 bins | Frequency-wise analysis |

MAPE (mean absolute percentage error) is documented as optional because it is unstable for near-zero or negative dBm values.

---

## 6. Assumptions and Design Decisions

### Assumptions

1. The AERPAW CSV has exactly `750` columns (3 nodes × 250 frequency bins).
2. AERPAW rows are chronological with no missing timestamps.
3. Each row can be reshaped to `(3, 250)` — the first 250 columns are CC1, next 250 are CC2, last 250 are LW1.
4. Weekly trend is unavailable and daily period requires reduced `lp` — the AERPAW CSV has only `6839` minutes, which is less than one week.
5. The three temporal branches capture complementary predictive information.
6. Learnable weighted fusion improves over equal-weight averaging.
7. The paper's PredRNN hyperparameters (4 layers, 128 hidden, 3×3 kernels) transfer reasonably to the AERPAW setup.

### Design Decisions

1. **Clean PyTorch rebuild** — The implementation is a fresh PyTorch reconstruction from the paper equations, not a port of the reference repo.
2. **Conceptual reference only** — The `pred-rnn` repo is incomplete; its ST-LSTM cell design is used as one reference for the cell equations.
3. **Configurable branches** — Each branch can be enabled or disabled independently, since the AERPAW dataset does not support the weekly trend branch.
4. **Independent branch weights by default** — The three PredRNN branches have independent weights unless `share_branch_weights: true` is set.
5. **Tanh output with min-max normalization** — Paper-faithful default.
6. **Linear output as optional experiment** — For cases where tanh limits regression performance.
7. **Single-step prediction** — The paper predicts one future map; multi-step is left as an extension.
8. **Learnable fusion weights per location** — Shape `(1, 1, H, W)` allows different importance weights per node-frequency position.

---

## 7. Deviations from Original STS-PredNet Setup

| Aspect | Paper (Original) | AERPAW Reconstruction | Reason |
|--------|------------------|-----------------------|--------|
| Dataset | Electrosense | AERPAW sub-6 GHz | Different measurement campaign |
| Sensors | 4 | 3 | Available AERPAW fixed nodes |
| Spatial locations | 100 | 3 | Electrosense scans 100 locations; AERPAW has 3 fixed nodes |
| Frequency bands | 60 | 250 per node | Different spectrum analyzer configurations |
| Map size | `(100, 60)` | `(3, 250)` | Different spatial × frequency dimensions |
| Time resolution | 10 minutes | 1 minute | AERPAW logs every minute |
| Period branch (`lp`) | 7 | 3 (reduced) | `7 × 1440 = 10080` min exceeds 6839 available; `3 × 1440 = 4320` fits |
| Trend branch | Available | Unavailable (without more data) | `6839` rows < one week of 1-min data |
| Normalization | Min-max `[-1, 1]` | Min-max `[-1, 1]` (same) | Paper-faithful default |
| `lc` | 36 | 36 | Same as paper |
| `lp` | 7 | 3 | Reduced to fit AERPAW (7 × 1440 = 10080 > 6839) |
| `lq` | 4 | 4 | Same as paper (but disabled — trend unavailable) |
| PredRNN layers | 4 | 4 | Same as paper |
| Hidden states | 128 | 128 | Same as paper |
| Kernel size | `3 × 3` | `3 × 3` | Same as paper |
| Optimizer | Adam | Adam | Same as paper |
| Learning rate | 0.0002 | 0.0002 | Same as paper |
| Batch size | 32 | 32 | Same as paper (but configurable) |
| Epochs | 500 | 500 | Same as paper (but configurable with early stopping) |
| Loss | MSE | MSE | Same as paper |
| Output activation | tanh | tanh | Same as paper (linear available) |

---

## 8. Known Limitations

1. **Weekly trend branch unavailable; daily period branch requires reduced `lp`.** The trend branch needs at least one week of data (`10080` minutes). The period branch with paper default `lp = 7` also needs `10080` minutes. With only `6839` minutes available, trend is disabled and period `lp` is reduced to `3` (needs `4320` minutes). These branches remain unavailable at full paper settings unless more data is collected.

2. **Daily period branch requires data history.** The period branch samples from `lp` previous days. The paper default `lp = 7` requires `10080` minutes of history, which exceeds the available `6839` minutes. A reduced `lp` (e.g., `lp = 3` needing `4320` minutes) is required. Even with reduced `lp`, early time steps in the dataset may not generate period branch samples.

3. **Small spatial dimension (3 rows).** With only 3 spatial nodes, the `3 × 3` convolution kernels span the full height of the spatial dimension. This means the model has limited ability to learn localized spatial patterns within individual nodes beyond what the kernel width captures.

4. **MAPE is unsuitable for negative dBm.** Mean absolute percentage error is not well-defined when true values are near zero or negative, which is common in dBm power measurements. MAPE is documented as optional and should be used with caution.

5. **Computational cost of three branches.** STS-PredNet is heavier than ConvLSTM because it runs three independent PredRNN branches (closeness, period, trend). Even with trend disabled, the two-branch model is roughly twice the cost of a single PredRNN.

6. **Reference repo is incomplete.** The `pred-rnn` repository (`git@github.com:Demii-7/pred-rnn.git`) is an incomplete implementation with syntax errors and mismatched shapes. It should not be used as a runnable baseline or copied directly.

---

## 9. Implementation Notes

1. **Paper equations are sufficient.** The original paper provides enough detail for a faithful reconstruction of the cell equations, fusion mechanism, and training procedure.

2. **Original code is unavailable.** The authors' original STS-PredNet implementation is not publicly available. The provided `pred-rnn` repo is a third-party PredRNN implementation that is incomplete and buggy.

3. **ST-LSTM cell from scratch.** The STS-ConvLSTM (ST-LSTM) cell must be implemented based on the PredRNN paper equations, not copied from the repo. The cell uses `Conv2d` for all gates with `kernel_size` and `padding` parameters.

4. **Zigzag memory flow.** The unified memory `M` must be passed between layers and across time steps as described in the PredRNN paper: horizontally from the last layer of time `t-1` to the first layer of time `t`, and vertically from layer `l` to layer `l+1` within the same time step.

5. **Fusion weights.** `W_c`, `W_p`, and `W_q` should be `nn.Parameter` objects of shape `(1, 1, H, W)` so they are broadcastable over batch and channel dimensions. They can be initialized to `1/n` (equal weight) where `n` is the number of enabled branches.

6. **Branch output shape.** Each branch outputs a tensor of shape `(B, 1, H, W)` that predicts the target spectrum map. All branch outputs must have the same spatial shape as the target.

7. **Disabled branches.** If `use_trend: false`, the trend branch is not constructed and its term is excluded from the fusion equation.

8. **Configurability.** All model and training parameters should be configurable via `config.yaml`. The config file documents the default from the paper with notes on why each field is configurable.

---

## References

1. **STS-PredNet paper:**
   Li, X., Liu, Z., Chen, G., Xu, Y., & Song, T. (2020). *Deep Learning for Spectrum Prediction From Spatial–Temporal–Spectral Data.* IEEE Communications Letters, 25(4), 1216–1220.
   DOI: [10.1109/LCOMM.2020.3045205](https://doi.org/10.1109/LCOMM.2020.3045205)

2. **PredRNN paper:**
   Wang, Y., Long, M., Wang, J., Gao, Z., & Yu, P. S. (2017). *PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs.* Advances in Neural Information Processing Systems, 30.
   [Link](https://papers.nips.cc/paper/6689-predrnn-recurrent-neural-networks-for-predictive-learning-using-spatiotemporal-lstms)

3. **Provided conceptual repo (incomplete):**
   `git@github.com:Demii-7/pred-rnn.git`
   A third-party PredRNN implementation with ST-LSTM cells. Contains syntax errors, mismatched shapes, and is not a runnable implementation. Use as conceptual reference only.
