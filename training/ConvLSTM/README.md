# ConvLSTM Spectrum Prediction — Reconstructed Model

> **Based on:** *Convolutional LSTM-based Long-Term Spectrum Prediction for Dynamic Spectrum Access* — Shawel, Woldegebreal, Pollin (EUSIPCO 2019)
>
> **Reference implementation:** https://github.com/ndrplz/ConvLSTM_pytorch (cloned locally as reference)
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

---

## 1. What the Model Is Intended to Do

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

## 2. Input Format

### 2.1 Raw CSV Format

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

### 2.2 CSV → Tensor Conversion

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

### 2.3 Expected Tensor Shape

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

---

## 3. Model Architecture

The architecture follows the paper's **sequence-to-sequence** design with an encoder–decoder structure.

### 3.1 Overview

```
Input: (B, T_in, 1, 3, 250)
        │
        ▼
┌─────────────────────────────────────────────┐
│              ENCODER                        │
│  ┌──────────────────────────────────┐       │
│  │ ConvLSTM Layer 1                 │       │
│  │  input_dim=1, hidden=32          │       │
│  │  kernel=(3,3), padding=1         │       │
│  │  activation=ReLU                 │       │
│  │     ↓ (B, T_in, 32, 3, 250)     │       │
│  │ ConvLSTM Layer 2                 │       │
│  │  input_dim=32, hidden=64         │       │
│  │  kernel=(1,1), padding=0         │       │
│  │  activation=ReLU                 │       │
│  │     ↓ (B, T_in, 64, 3, 250)     │       │
│  └──────────────────────────────────┘       │
│  Last states: (h_enc, c_enc)                │
│  each (B, 64, 3, 250)                       │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│         DECODER                             │
│  ┌──────────────────────────────────┐       │
│  │ LSTM (regular)                   │       │
│  │  Flatten encoder states          │       │
│  │  → (B, 64*3*250)                │       │
│  │  LSTM(hidden=128)                │       │
│  │  → (B, 128)                      │       │
│  │  Unflatten → (B, 32, 3, 250)    │       │
│  └──────────────────────────────────┘       │
│        │                                     │
│        ▼                                     │
│  ┌──────────────────────────────────┐       │
│  │ ConvLSTM Layer 3                 │       │
│  │  input_dim=64, hidden=32         │       │
│  │  kernel=(1,1), padding=0         │       │
│  │  activation=ReLU                 │       │
│  │  Dropout(p=0.3)                  │       │
│  │  BatchNorm(32)                   │       │
│  │     ↓ (B, T_out, 32, 3, 250)    │       │
│  └──────────────────────────────────┘       │
│        │                                     │
│        ▼                                     │
│  ┌──────────────────────────────────┐       │
│  │ FC (Dense) Output Layer          │       │
│  │  Conv2d(32 → 16, k=3, pad=1)     │       │
│  │  ReLU                             │       │
│  │  Conv2d(16 → 1, k=1)             │       │
│  │     ↓ (B, T_out, 1, 3, 250)     │       │
│  └──────────────────────────────────┘       │
└─────────────────────────────────────────────┘
        │
        ▼
Output: (B, T_out, 1, 3, 250)
        Squeeze channels → (B, T_out, 3, 250)
```

### 3.2 Layer-by-Layer Specification

#### Encoder — ConvLSTM Layer 1

| Parameter | Value |
|-----------|-------|
| Type | `ConvLSTMCell` (2D convolutional LSTM cell) |
| Input channels | 1 (normalized dBm PSD) |
| Hidden channels | 32 |
| Kernel size | `(3, 3)` |
| Padding | `(1, 1)` (same convolution, preserves H×W) |
| Bias | True |
| Activation (gates) | Sigmoid |
| Activation (cell candidate g) | ReLU (per paper §III-A, replaces tanh) |
| Activation (output H) | ReLU (per paper §III-A, replaces tanh) |
| Output shape | `(B, T_in, 32, 3, 250)` |

**Role**: Learns low-level spatial-spectral features. The `3×3` kernel captures local correlations between adjacent nodes and adjacent frequency bins simultaneously. Padding of 1 preserves the `3×250` spatial dimensions.

#### Encoder — ConvLSTM Layer 2

| Parameter | Value |
|-----------|-------|
| Type | `ConvLSTMCell` (2D convolutional LSTM cell) |
| Input channels | 32 (from Layer 1) |
| Hidden channels | 64 |
| Kernel size | `(1, 1)` |
| Padding | `(0, 0)` |
| Bias | True |
| Output shape | `(B, T_in, 64, 3, 250)` |

**Role**: Compresses the learned features into a higher-dimensional latent space. The `1×1` kernel performs pointwise convolution — it mixes information across channels at each spatial location without mixing adjacent spatial positions. This acts as a learned feature aggregation layer.

#### State Transfer (Encoder → Decoder) — Implementation Choice

> **Note:** The paper states there is *"an LSTM hidden layer that is used to capture memory and hidden states from the encoder output"* but does not specify its exact dimensionality or connectivity. The design below is our interpretation.

The encoder's final hidden and cell states from ConvLSTM Layer 2 (`h_enc`, `c_enc` — each `(B, 64, 3, 250)`) are fed into a regular LSTM layer at the decoder:

1. **Flatten**: Each state tensor is flattened from `(B, 64, 3, 250)` to `(B, 64 × 3 × 250) = (B, 48000)`.
2. **LSTM**: The flattened state passes through a regular LSTM cell with hidden size 128, producing a context vector `(B, 128)`.
3. **Unflatten → ConvLSTM init**: The context vector is projected back to a 4D tensor:
   - `Linear(128 → 32 × 3 × 250)` → `(B, 32, 3, 250)`
   - This projected tensor initializes the decoder ConvLSTM's hidden state `h_dec_0` and cell state `c_dec_0`.

This LSTM layer acts as the memory-capture mechanism referenced in the paper: it compresses the encoder's spatial-spectral representation into a compact code and initializes the decoder's generative process. Sizing (128, 48000) is empirical; tune as needed.

#### Decoder — ConvLSTM Layer 3

| Parameter | Value |
|-----------|-------|
| Type | `ConvLSTMCell` (2D convolutional LSTM cell) |
| Input channels | 64 (learned "start" embedding projected from LSTM context) |
| Hidden channels | 32 |
| Kernel size | `(1, 1)` |
| Padding | `(0, 0)` |
| Bias | True |
| Activation (gates) | Sigmoid |
| Activation (cell candidate g) | ReLU (per paper §III-A, replaces tanh) |
| Activation (output H) | ReLU (per paper §III-A, replaces tanh) |
| Dropout | `p = 0.3` (after hidden state, before output) |
| Batch norm | Layer-wise batch normalization on hidden states |
| Output shape | `(B, T_out, 32, 3, 250)` |

**Role**: Generates the output sequence step by step. At each decoder time step:
- **t=1**: Input is a learned "start" embedding (projected from the LSTM context vector), and states are initialized from the encoder via the LSTM.
- **t>1**: Input is the previous step's hidden state (autoregressive decoding). Optionally, ground-truth frames can be fed instead during training (teacher forcing — see §5.9; this is our implementation choice, not in the paper).

#### Decoder — Fully Connected (Dense) Output Layer

This implements the "fully connected (dense) layer" from the paper §III-A. The default is a single `1×1` convolution (equivalent to a per-location dense layer):

| Layer | Type | Input → Output | Kernel | Padding |
|-------|------|----------------|--------|---------|
| FC | `Conv2d` | 32 → 1 | (1, 1) | (0, 0) |

Input: `(B, 32, 3, 250)` — single time step of decoder ConvLSTM hidden state.

Output: `(B, 1, 3, 250)` — squeezed to `(B, 3, 250)`.

**Role**: Linearly projects the 32-channel hidden representation to a single normalized PSD value per (node, frequency bin) location. The `1×1` kernel ensures each spatial location is projected independently, exactly matching a fully-connected layer applied per position.

**Optional alternative** (not in the paper): Replace the single `1×1` Conv2d with a small two-layer MLP convolution to add non-linearity:

| Layer | Type | Input → Output | Kernel | Padding |
|-------|------|----------------|--------|---------|
| FC1 | `Conv2d` | 32 → 16 | (3, 3) | (1, 1) |
| Activation | ReLU | — | — | — |
| FC2 | `Conv2d` | 16 → 1 | (1, 1) | (0, 0) |

This version incorporates local spatial-spectral context (via the 3×3 kernel) before the final projection. Enabled by setting `fc_hidden_channels > 0` in config.yaml.

### 3.3 ConvLSTMCell Equations

Each ConvLSTMCell follows the formulation from Shi et al. (2015), modified per paper §III-A to use ReLU instead of tanh for the output and cell candidate activations:

```
i_t = σ(W_xi ∗ X_t + W_hi ∗ H_{t-1} + b_i)
f_t = σ(W_xf ∗ X_t + W_hf ∗ H_{t-1} + b_f)
o_t = σ(W_xo ∗ X_t + W_ho ∗ H_{t-1} + b_o)
g_t = ReLU(W_xg ∗ X_t + W_hg ∗ H_{t-1} + b_g)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
H_t = o_t ⊙ ReLU(C_t)
```

where:
- `∗` = 2D convolution
- `⊙` = Hadamard (element-wise) product
- `σ` = sigmoid gate activation
- `ReLU` = rectified linear unit (output and cell candidate, per paper §III-A)
- `i, f, o, g` = input gate, forget gate, output gate, cell candidate
- `C, H` = cell state, hidden state

### 3.4 Prediction Target

The model predicts **the next `T_out` consecutive minutes** of PSD values across all 3 nodes and 250 frequency bins. Specifically:

- **Single-frame prediction**: The immediate next minute `(3 × 250)` slice
- **Multi-frame prediction**: `T_out` consecutive minutes `(T_out × 3 × 250)`

The output is in **normalized (z-score) dBm space**. Predictions must be un-normalized for interpretation:
```
dBm = prediction × σ_bin + μ_bin
```

---

## 4. Output Format

### 4.1 Raw Output Tensor Shape

```
(B, T_out, 1, 3, 250)
```

| Dimension | Meaning |
|-----------|---------|
| B | Batch size |
| T_out | Number of predicted future minutes |
| 1 | Channel dimension (squeezed in post-processing) |
| 3 | Sensor node index (0=CC1, 1=CC2, 2=LW1) |
| 250 | Frequency bin index |

### 4.2 Meaning of Each Output Element

Each output element `y_{b,t,0,n,f}` represents the **predicted normalized power spectral density** for:
- Batch sample `b`
- Future time step `t` (minutes from now)
- Node `n` (CC1, CC2, or LW1)
- Frequency bin `f`

### 4.3 Post-Processing: Normalization → dBm

```python
# Z-score denormalization
pred_dbm = pred_normalized * freq_std[freq_idx] + freq_mean[freq_idx]

# Array reshape: (B, T_out, 3, 250) → (B, T_out, 750)
pred_flat = pred_dbm.reshape(B, T_out, -1)
```

### 4.4 CSV Output Format

Predicted data can be written as CSV with the same format as the input:
- No header
- 6 decimal places
- 750 columns per row (CC1[0–249], CC2[250–499], LW1[500–749])
- One row per predicted minute

### 4.5 Visualization

To visualize predictions as spectrograms:
1. Select a specific node: `pred_dbm[:, :, node_idx, :]` → shape `(B, T_out, 250)`
2. Transpose to `(250, T_out)` for a frequency-vs-time spectrogram
3. Plot with `matplotlib.pyplot.imshow` using a `'viridis'` or `'jet'` colormap
4. Overlay ground truth for comparison (side-by-side or difference plot)

---

## 5. Training Pipeline

### 5.1 Data Loading Process

```python
# Pseudocode
dataset = SpectrumDataset(csv_path, T_in=12, T_out=6, stride=1)
# Returns: (X, y) where
#   X shape: (num_samples, T_in, 3, 250)
#   y shape: (num_samples, T_out, 3, 250)
#   Both in normalized dBm
```

The `SpectrumDataset` class:
1. Loads CSV as `numpy.ndarray` of shape `(6839, 750)`
2. Reshapes to `(6839, 3, 250)`
3. Computes normalization statistics (frequency-bin-wise mean/std)
4. Normalizes the data
5. Generates sliding windows of `(T_in + T_out)` consecutive time steps
6. Returns `(X, y)` tuples for each window

### 5.2 Normalization

**Method**: Z-score (standard score) normalization, applied **per frequency bin** across the time dimension.

For each frequency bin `f` in `{0..249}` and each node `n` in `{0..2}`:

```
μ_{n,f} = mean( X_train[:, n, f] )
σ_{n,f} = std( X_train[:, n, f] )
X_normalized[:, n, f] = (X[:, n, f] − μ_{n,f}) / σ_{n,f}
```

- Statistics are computed **only from the training set** to prevent data leakage.
- The same `μ` and `σ` are applied to validation and test sets.
- Saved alongside the model checkpoint for inference-time denormalization.

### 5.3 Train / Validation / Test Split

**Ratios**: Configurable in `config.yaml` under `split`. Default is 80/10/10, but `val_ratio` can be set to 0 for train/test only. Ratios can sum to less than 1.0 (remaining data unused) or to 1.0 (all data used).

**Strategy**: Default is **chronological split** (recommended for time series). If `chronological_split: false`, splits are random.

With the default config (80/10/10 chronological):

- Train: first 80% of temporal window centers
- Validation: next 10%
- Test: final 10%

This respects temporal ordering and avoids look-ahead bias.

### 5.4 Windowing / Sequence Generation

With the default config (`T_in=12`, `T_out=6`, `stride=1`):

```
Total time steps: 6839
Windows per sequence: 6839 − (12 + 6) + 1 = 6822

Training windows:  floor(6822 × 0.80) = 5457
Validation windows: floor(6822 × 0.10) =  682
Test windows:        6822 − 5457 − 682 =  683
```

Each window `(X, y)`:
- `X`: 12 consecutive time steps → `(12, 3, 250)`
- `y`: 6 consecutive time steps immediately following `X` → `(6, 3, 250)`
- DataLoader adds batch dimension → `(B, 12, 3, 250)` and `(B, 6, 3, 250)`
- Channel dimension added → `(B, 12, 1, 3, 250)` and `(B, 6, 1, 3, 250)`

### 5.5 Loss Function

**Mean Squared Error (MSE)**:

```
L = (1 / (B × T_out × 3 × 250)) × Σ(predicted − target)²
```

MSE penalizes larger errors more heavily, which is appropriate for PSD values where outliers (e.g., unexpected interference) matter.

**Alternative**: MAE (Mean Absolute Error) can be used for robustness to outliers.

### 5.6 Optimizer

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | Adam (default) | Paper uses NADAM |
| Learning rate | 0.0002 | Paper §III-B |
| β₁ | 0.9 | Paper §III-B |
| β₂ | 0.999 | Paper §III-B |
| ε | 1×10⁻⁸ | Paper §III-B |
| Weight decay (λ) | 0.004 | Paper §III-B |
| Gradient clip norm | 5.0 | Common practice |

NADAM (Nesterov-accelerated Adam) is preferred per the paper but requires a third-party implementation. Adam provides a close approximation.

### 5.7 Batch Size

- Default: **32**
- Tune based on GPU memory. Reduce to 16 or 8 for smaller GPUs.

### 5.8 Number of Epochs

- Default: **100**
- Early stopping with patience of 20 epochs based on validation loss
- Learning rate reduction on plateau (factor 0.5, patience 10)

### 5.9 Teacher Forcing (Implementation Choice, Not in Paper)

Optionally, the decoder can receive ground-truth frames as input during training with probability `teacher_forcing_ratio` (default 0.5). This:
- Speeds up convergence (model sees real data during decoding)
- Acts as regularization (randomly switches between ground-truth and model's own predictions)

Set `teacher_forcing_ratio: 0` in config.yaml for pure autoregressive decoding (closer to the paper's described setup).

### 5.10 Regularization Techniques

| Technique | Value | Paper reference |
|-----------|-------|-----------------|
| Dropout (encoder & decoder) | p = 0.3 | §III-B |
| Batch normalization | After ConvLSTM layers | §III-B |
| Gaussian input noise | σ = 0.2 | §III-B |
| Weight decay (L2) | λ = 0.004 | §III-B (NADAM parameter) |

### 5.11 Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| RMSE | √(mean((ŷ − y)²)) | Primary metric, same unit as dBm |
| MAE | mean(|ŷ − y|) | Robust to outliers |
| R² | 1 − (SS_res / SS_tot) | Proportion of variance explained |

Metrics are computed:
- **Per horizon**: RMSE at t=1, t=3, t=6 (to measure error accumulation)
- **Per node**: RMSE for CC1, CC2, LW1 separately
- **Overall**: Average across all nodes and time steps

---

## 6. Modular Design — Recommended Code Structure

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

### 6.1 Module Responsibilities

| File | Contents |
|------|----------|
| `dataset.py` | `SpectrumDataset` (torch `Dataset`), CSV loading, z-score normalization, sliding windows, train/val/test splitting |
| `model.py` | `ConvLSTMCell`, `ConvLSTM` (multi-layer, from reference), `ConvLSTMPredictor` (seq2seq encoder–decoder) |
| `train.py` | Training loop, teacher forcing, gradient clipping, LR scheduling, early stopping, checkpoint saving |
| `evaluate.py` | Test set evaluation, RMSE/MAE/R² per horizon and per node, spectrogram visualization, prediction CSV export |
| `utils.py` | Normalization statistics, metrics, seed setting, device detection, denormalization |
| `inference.py` | Load checkpoint + normalization stats, predict on arbitrary CSV input, save predictions as CSV |
| `config.yaml` | All hyperparameters (see Section 7) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

### 6.2 State Flow Between Modules

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

## 7. Configuration (`config.yaml`)

All hyperparameters are in `config.yaml` (located in this directory). Key settings:

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| Data | `dataset_path` | `training/data/merged_power_data_sub6GHz_avg_per_minute.csv` | Input CSV path |
| Data | `n_bins_per_node` | 250 | Frequency bins per node |
| Data | `n_nodes` | 3 | Number of sensor nodes |
| Preprocessing | `normalization` | `zscore` | Normalization method |
| Windowing | `input_sequence_length` | 12 | Past minutes (T_in) |
| Windowing | `prediction_horizon` | 6 | Future minutes (T_out) |
| Windowing | `stride` | 1 | Window stride |
| Split | `train_ratio` | 0.8 | Training set fraction |
| Split | `val_ratio` | 0.1 | Validation set fraction (set to 0 for train/test only) |
| Split | `chronological_split` | true | Chronological (true) or random (false) split |
| Model | `input_channels` | 1 | Input channel count |
| Model | `hidden_channels` | [32, 64] | Encoder layer hidden sizes |
| Model | `kernel_size` | [[3,3], [1,1]] | Encoder kernel sizes |
| Model | `decoder_hidden_channels` | 32 | Decoder hidden size |
| Model | `decoder_kernel_size` | [1,1] | Decoder ConvLSTM kernel size (per paper §III-A) |
| Model | `dropout` | 0.3 | Dropout probability |
| Model | `use_batch_norm` | true | Use batch normalization |
| Model | `decoder_lstm_hidden` | 128 | Regular LSTM hidden size in decoder |
| Model | `fc_hidden_channels` | 0 | FC intermediate channels (0 = single 1×1 Conv2d) |
| Model | `fc_kernel_size` | [3,3] | FC intermediate kernel (only if fc_hidden_channels > 0) |
| Training | `batch_size` | 32 | Mini-batch size |
| Training | `epochs` | 100 | Max training epochs |
| Training | `learning_rate` | 0.0002 | Initial learning rate |
| Training | `optimizer` | adam | Optimizer choice |
| Training | `teacher_forcing_ratio` | 0.5 | Teacher forcing probability |
| Training | `noise_std` | 0.2 | Gaussian noise std for input |
| Device | `device` | auto | cuda / cpu / auto |

---

## 8. Usage Instructions

> **Note**: The scripts below are **proposed commands** that will work once the corresponding Python files are implemented. They assume you are running from the repository root (`/home/cc/spectrum-usage`).

### 8.1 Setup

```bash
cd /home/cc/spectrum-usage

# Create virtual environment (one time)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch numpy matplotlib pyyaml scikit-learn tqdm
```

### 8.2 Train the Model

```bash
# Train with default config
python3 training/ConvLSTM/train.py

# Train with custom config
python3 training/ConvLSTM/train.py --config training/ConvLSTM/config.yaml

# Train with overrides
python3 training/ConvLSTM/train.py \
    --batch-size 64 \
    --epochs 150 \
    --lr 0.0001 \
    --input-len 24 \
    --pred-horizon 12
```

Expected output:
```
Training/ConvLSTM/
├── checkpoints/
│   ├── best_model.pt          # Lowest validation loss checkpoint
│   ├── last_model.pt          # Final epoch checkpoint
│   └── normalization_stats.pt # μ and σ for denormalization
├── logs/
│   └── training_YYYYMMDD_HHMMSS.json
└── config.yaml
```

### 8.3 Evaluate the Model

```bash
# Evaluate best checkpoint
python3 training/ConvLSTM/evaluate.py \
    --checkpoint training/ConvLSTM/checkpoints/best_model.pt

# Evaluate with specific horizons
python3 training/ConvLSTM/evaluate.py \
    --checkpoint training/ConvLSTM/checkpoints/best_model.pt \
    --horizons 1 3 6 12
```

Expected output:
```
=== Evaluation Report ===
Overall RMSE: 0.8423

Per-horizon RMSE:
  t=1:  0.6123
  t=3:  0.8911
  t=6:  1.0234

Per-node RMSE:
  CC1:  0.7892
  CC2:  0.9123
  LW1:  0.8254

Outputs: training/ConvLSTM/evaluation/
├── predictions.csv
├── metrics.json
├── spectrogram_comparison_*.png
└── error_analysis.png
```

### 8.4 Run Inference on New Data

```bash
# Predict on new CSV data
python3 training/ConvLSTM/inference.py \
    --checkpoint training/ConvLSTM/checkpoints/best_model.pt \
    --input data/some_new_measurements.csv \
    --output predictions/predicted_spectrum.csv \
    --t-in 12 \
    --t-out 6
```

### 8.5 Monitor Training with TensorBoard

```bash
tensorboard --logdir training/ConvLSTM/logs
```

---

## Assumptions and Design Decisions

### Assumptions

1. **The input CSV has exactly 750 columns** (3 nodes × 250 bins) in the order CC1, CC2, LW1.
2. **The CSV has no header row** and contains only comma-separated numeric dBm values.
3. **Rows are in strict chronological order** with no gaps (the existing data has 6,839 contiguous minutes).
4. **Normalization statistics are computed per frequency bin** across the time dimension, not globally.
5. **The train/val/test split is chronological**, not random, because this is time-series data.
6. **The per-node-offset CSV is the target dataset** (`merged_power_data_sub6GHz_avg_per_minute.csv`), produced by `training/build_training_csv.py` using per-node raw SigMF bin offsets (CC1=21000, CC2=33250, LW1=27500). Each node's 250 bins are from a different quiet L-band / lower S-band region, not a shared 87–336 MHz band.

### Design Decisions

1. **2D ConvLSTM with (3, 250) spatial map**: The 3 nodes are treated as the "height" dimension and 250 frequency bins as the "width" dimension. This allows the 2D kernels to learn cross-node and cross-frequency patterns simultaneously.

2. **Single-channel input**: Power values are in dBm (scalar per node×frequency point). The channel dimension is 1.

3. **Seq2seq with optional teacher forcing**: The decoder generates outputs autoregressively. Teacher forcing (default 0.5) is our addition, not in the paper; set to 0 for the paper's pure autoregressive setup.

4. **No spatial interpolation**: The paper used IDW interpolation across 1600 grid points from 5 sensors. Our AERPAW dataset has 3 fixed nodes without a regular spatial grid. We preserve the raw per-node data as independent rows in the 2D map.

5. **Per-bin z-score normalization**: Spectrum data has different power levels across frequency bands. Normalizing per bin ensures each frequency contributes equally to the loss.

### Deviations from the Original Paper

| Aspect | Paper (2019) | Our Reconstruction | Reason |
|--------|-------------|-------------------|--------|
| Dataset | Electrosense (5 sensors, 450–520 MHz) | AERPAW (3 nodes, per-node L-band offsets) | Our target dataset |
| Spatial dimension | 40×40 IDW grid → 1600 locations | 3 fixed nodes | AERPAW has sparse, fixed nodes |
| Input time steps | 120 (6 hours × 3 min resolution) | Configurable (e.g. 12) | Set in config; 1-min vs 3-min resolution |
| Prediction horizon | 50 steps (150 min) | Configurable (e.g. 6) | Set in config.yaml |
| Encoder layers | 2 ConvLSTM | 2 ConvLSTM | Same |
| Decoder layers | LSTM + ConvLSTM + FC | LSTM + ConvLSTM + FC (1×1 Conv2d; 2-layer MLP optional) | Same structure; FC details are our interpretation |
| Activation | ReLU (output, per §III-A) | ReLU (output) | Same |
| Optimizer | NADAM | Adam | NADAM not in standard PyTorch |
| Framework | R + TensorFlow | Python + PyTorch | Our stack |

### Things That Still Need Clarification

1. **Optimal T_in and T_out** for the AERPAW dataset. The paper used 120-step input and 50-step output (3-min resolution). Our 1-min resolution may need different values; set in `config.yaml`.

2. **Spatial dimension treatment**. The paper had a regular 2D spatial grid (40×40). We have 3 discrete nodes. Treating nodes as rows in a 2D map vs. separate input channels vs. independent per-node models should be validated empirically.

3. **Regular LSTM layer in the decoder**. The paper describes it as "used to capture memory and hidden states from the encoder output." Our reconstruction flattens the encoder states, passes them through an LSTM, then unflattens to initialize the decoder ConvLSTM. The exact dimensionality and connectivity (whether the LSTM processes the full sequence or just the final state) is underspecified and may need tuning.
