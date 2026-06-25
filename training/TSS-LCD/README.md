# TSS-LCD Spectrum Prediction — Adaptation

Adaptation plan for integrating TSS-LCD (Temporal–Spectral–Spatial Guided Latent Conditional Diffusion Model) into the AERPAW spectrum prediction pipeline. This document covers dataset adaptation, architecture porting, training pipeline design, and known constraints before implementation.

## Quick Start

```bash
# 1. Activate environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install torch numpy pandas scipy scikit-learn pyyaml matplotlib

# 3. Place CSV at expected path
#    training/data/merged_power_data_sub6GHz_avg_per_minute.csv

# 4. Run training (three stages)
python3 training/TSS-LCD/train_autoencoder.py    # Stage 1
python3 training/TSS-LCD/train_tss_condition.py  # Stage 2
python3 training/TSS-LCD/train_diffusion.py      # Stage 3

# 5. Evaluate
python3 training/TSS-LCD/evaluate.py
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `dataset.py` | Dataset class, windowing, masking, reshaping |
| `model.py` | All model components (TSS-CC, LSE, LSD, NEN) |
| `train_autoencoder.py` | Stage 1: latent autoencoder pretraining |
| `train_tss_condition.py` | Stage 2: TSS-CC condition training |
| `train_diffusion.py` | Stage 3: LCD noise-estimation training |
| `evaluate.py` | Evaluation on test split |
| `inference.py` | Online inference from incomplete observations |
| `utils.py` | Normalization, metrics, checkpointing, config loading |

## File Structure

```text
training/TSS-LCD/
├── README.md              # This file
├── config.yaml            # Configuration
├── dataset.py             # Dataset and dataloading
├── model.py               # TSS-LCD model definition
├── train_autoencoder.py   # Stage 1 training script
├── train_tss_condition.py # Stage 2 training script
├── train_diffusion.py     # Stage 3 training script
├── evaluate.py            # Evaluation script
├── inference.py           # Inference script
├── utils.py               # Utilities
├── checkpoints/           # Saved model weights (created during training)
└── results/               # Metric CSVs, plots (created during evaluation)
```

## Configuration Reference

All configurable options live in `config.yaml`. Key fields:

| Field | Default | Note |
|-------|---------|------|
| `data.dataset_path` | — | Path to the merged CSV |
| `data.n_nodes` | 3 | Number of sensor nodes (CC1, CC2, LW1) |
| `data.n_bins_per_node` | 250 | Frequency bins per node |
| `data.node_names` | [CC1, CC2, LW1] | Node labels matching CSV column order |
| `windowing.input_sequence_length` | 50 | Historical time steps T_in |
| `windowing.prediction_horizon` | 10 | Future time steps T_out |
| `windowing.train_stride` | 1 | Sliding window stride for training |
| `windowing.val_stride` | 1 | Stride for validation windows |
| `windowing.test_stride` | 1 | Stride for test windows |
| `split.train_ratio` | 0.7 | Fraction for training |
| `split.val_ratio` | 0.15 | Fraction for validation |
| `split.test_ratio` | 0.15 | Fraction for testing |
| `split.chronological_split` | true | Chronological (not random) split |
| `preprocessing.normalization` | minmax | Min-max or z-score normalization |
| `preprocessing.fit_on_train_only` | true | Fit scaler only on training data |
| `preprocessing.missing_rate` | 0.25 | Fraction of historical entries masked |
| `preprocessing.masking_strategy` | random | 'random' or 'continuous' missing pattern |
| `preprocessing.zero_pad_missing` | true | Replace missing entries with 0 |
| `model.latent_dim` | 32 | Latent space dimension |
| `model.hidden_dim` | 256 | Hidden dimension in attention/FFN layers |
| `model.attention_heads` | 4 | Multi-head attention heads per branch |
| `model.diffusion_steps` | 1000 | Number of diffusion timesteps N |
| `model.noise_schedule` | cosine | Cosine or linear noise schedule |
| `model.use_temporal_branch` | true | Enable TemFE branch |
| `model.use_spectral_branch` | true | Enable SpeFE branch |
| `model.use_spatial_branch` | true | Enable SpaFE branch |
| `training.autoencoder_epochs` | 300 | Epochs for latent autoencoder |
| `training.tss_epochs` | 200 | Epochs for TSS condition stage |
| `training.diffusion_epochs` | 1000 | Epochs for diffusion training |
| `training.batch_size` | 32 | Batch size |
| `training.learning_rate` | 1e-4 | Adam learning rate |
| `training.optimizer` | adam | Optimizer choice |
| `training.gradient_clip` | 5.0 | Max gradient norm for clipping |
| `evaluation.metrics` | [rmse, mae, r2] | Evaluation metrics |
| `evaluation.eval_horizons` | [1, 5, 10] | Prediction horizons to evaluate |
| `device.device` | auto | 'cuda', 'cpu', or 'auto' |

## 1. What the Model Is Intended to Do

TSS-LCD is a two-stage generative model for spectrum prediction under incomplete historical observations:

- **Predict future spectrum RSS/PSD** across multiple nodes and frequency bins given a window of past measurements
- **Handle incomplete historical observations** where entries are missing (zero-padded) due to partial-band scanning, hardware constraints, or scheduling policies
- **Preserve fine-grained RSS variations** (2–5 dBm fluctuations) that regression-based models typically over-smooth
- **Model temporal, spectral, and spatial dependencies jointly** using dedicated self-attention branches per dimension

The paper presents TSS-LCD as superior to ConvLSTM, CSMA, STS-PredNet, DSwinLSTM, and 3D-SwinSTB under missing rates from 15% to 35%, for both random and continuous missing patterns.

## 2. Input Format

### Raw CSV

```text
Path:    training/data/merged_power_data_sub6GHz_avg_per_minute.csv
Shape:   (6839, 750)
Layout:  [CC1_cols_0_249 | CC2_cols_250_499 | LW1_cols_500_749]
```

The CSV has **no header row**. Each row is per-minute averaged RSS in dBm across three fixed AERPAW nodes. Column ordering (confirmed by `build_training_csv.py` and reverse-engineering):

| Columns | Node | Approx. frequency range |
|---------|------|------------------------|
| 0–249 | CC1 | ~1347–1362 MHz |
| 250–499 | CC2 | ~2082–2097 MHz |
| 500–749 | LW1 | ~1737–1752 MHz |

### Reshaped Tensor

```text
Raw:  (T, D)         = (6839, 750)
Step: reshape to     = (T, L, F)  = (6839, 3, 250)
```

Where `L = 3` (CC1, CC2, LW1) and `F = 250` frequency bins per node.

### Windowing

- Input window `X: (T_in, L, F)` — historical observations
- Target window `Y: (T_out, L, F)` — future ground truth
- Total windows: `(6839 - T_in - T_out + 1)` with stride 1
- Chronological train/val/test split (7:1.5:1.5 by default)

### Incomplete Observation Simulation

Missing entries are simulated on input `X` only (target `Y` is always complete):

- **Random missing**: Each entry has probability `missing_rate` of being zero-padded
- **Continuous missing**: A contiguous block of length proportional to `missing_rate` is zero-padded
- A configurable binary observation mask tensor is returned alongside `X` and `Y` to support masked-loss training if needed

## 3. Model Architecture

TSS-LCD is a two-stage network (see paper Fig. 3):

### Stage 1: TSS-CC — Condition Construction

Three parallel multi-head self-attention branches extract dimension-specific features from the incomplete input tensor `X ∈ R^(T_in × L × F)`:

#### Temporal Feature Extraction (TemFE)

- Flattens spatial and spectral dims: `X → X_temp ∈ R^(T_in × (L·F))`
- Adds sinusoidal positional encoding
- Multi-head self-attention along the time axis
- Output: `H_temp ∈ R^(T_in × (L·F))`

#### Spectral Feature Extraction (SpeFE)

- Permutes tensor: `X → X_spec ∈ R^(F × (L·T_in))`
- Adds sinusoidal positional encoding
- Multi-head self-attention along the frequency axis
- Output: `H_spec ∈ R^(F × (L·T_in))`

#### Spatial Feature Extraction (SpaFE)

- Permutes tensor: `X → X_spat ∈ R^(L × (F·T_in))`
- Adds sinusoidal positional encoding
- Multi-head self-attention along the location axis
- Output: `H_spat ∈ R^(L × (F·T_in))`

#### Feature Fusion Module (FFM)

- Cross-attention mechanism that uses temporal feature as query, spectral as key, and (reshaped) spatial as value
- Residual connections + layer norm + FFN
- Output: `H_fusion ∈ R^(T_in × (L·F))` — the unified conditional representation

### Stage 2: LCD-SP — Latent Diffusion Prediction

#### Latent Space Encoder (LSE)

- 2D convolutional encoder with downsampling blocks (Conv2d + BN + ReLU)
- Maps ground-truth `Y ∈ R^(T_out × L × F)` → latent `z0 ∈ R^(1 × d_latent)`

#### Forward Diffusion Process

- Adds Gaussian noise progressively over `N` steps: `zn = sqrt(alpha_bar_n) * z0 + sqrt(1 - alpha_bar_n) * eps`
- Cosine noise schedule (default)

#### Noise Estimation Network (NEN)

- Takes concatenation of `[zn, cond_proj(H_fusion), time_embedding(n)]`
- Conv1D-based U-Net with downsampling, bottleneck, upsampling, and skip connections
- Estimates the noise `eps_hat_n` added at step `n`

#### Reverse Denoising Process

- Starts from pure Gaussian noise `zN`
- Iteratively denoises using NEN: `zn_minus_1 = mu_n + sigma_n * eta`
- `N` steps total, guided by `H_fusion` at each step
- Recovers denoised latent `z0_hat`

#### Latent Space Decoder (LSD)

- Mirrors LSE structure: FC layer + ConvTranspose2d upsampling blocks
- Maps `z0_hat` → `Y_hat ∈ R^(T_out × L × F)`

## 4. Output Format

```text
Shape:   (T_out, L, F)   e.g. (10, 3, 250)
Flatten: (T_out, 750)
```

Output is the predicted future RSS/PSD in the same scale as the preprocessed input (normalized [0,1] or dBm depending on config). During evaluation, predictions are inverse-transformed to dBm for metric computation.

## 5. Training Pipeline

Three separate training stages, each with its own script and checkpoint:

### Stage 1: Latent Autoencoder Pretraining

```text
Script:  train_autoencoder.py
Input:   Y (complete future ground truth)
Loss:    MSE reconstruction: ||Y - LSD(LSE(Y))||^2
Freezes: none
Output:  checkpoints/ae_lse.pth, checkpoints/ae_lsd.pth
```

The LSE and LSD are trained end-to-end to compress and reconstruct future spectrum data in a compact latent space.

### Stage 2: TSS Condition Stage Training

```text
Script:  train_tss_condition.py
Input:   X (incomplete historical) + Y (complete future)
Loss:    MSE between predicted and true latent: ||LSE(Y) - TSS-CC(X)||^2
Freezes: LSE, LSD (from Stage 1)
Output:  checkpoints/tss_cc.pth
```

The TSS-CC modules (TemFE, SpeFE, SpaFE, FFM) are trained to produce a conditional representation that aligns with the latent encoding of the future spectrum.

### Stage 3: LCD Diffusion Training

```text
Script:  train_diffusion.py
Input:   X (incomplete historical) + Y (complete future)
Loss:    MSE between true and predicted noise: ||eps - NEN(zn, n, H_fusion)||^2
Freezes: LSE, LSD, TSS-CC (all previous stages)
Output:  checkpoints/diffusion.pth
```

Only the NEN is trained. The frozen TSS-CC produces the condition, the frozen LSE encodes the target latent, and the NEN learns to predict the injected noise.

### Evaluation / Inference

```text
Script:   evaluate.py / inference.py
Process:  X → TSS-CC → H_fusion → (reverse diffusion) → Y_hat
Metrics:  RMSE, MAE, R² per horizon, per node, per frequency bin
```

## 6. Assumptions and Design Decisions

- **CSV has no header**: The `dataset.py` loader will read raw values directly (no header skipping needed for our CSV, but the original repo CSV also has no header)
- **Chronological split**: Always chronological for time series; no random shuffle across time
- **Window stride is configurable**: Default stride=1 for maximum training samples; larger strides for smoke tests
- **Complete-observation baseline first**: Train with `missing_rate=0` first to verify the pipeline before introducing masking
- **Masking applied only to input `X`**: The target `Y` is always fully observed (matching the paper setup)
- **LSE/LSD operate on future data only**: They compress `Y`, never `X`; during inference they decode the denoised latent
- **TSS-CC outputs a condition, not a direct prediction**: The condition guides diffusion rather than producing a direct forecast
- **Normalization fit on training split only**: Avoids data leakage from test/val splits

## 7. Deviations from Original TSS-LCD Setup

| Aspect | Original Paper | Our Adaptation | Rationale |
|--------|---------------|----------------|-----------|
| Frequency range | 85–335 MHz (same 250 bins for all nodes) | Per-node offset (1347, 2082, 1737 MHz) | Reverse-engineered; our CSV uses different bands per node |
| Time steps | 10,080 (1 week) | 6,839 (~4.75 days) | Our processed CSV has fewer common-minutes across nodes |
| T_in / T_out | 50 / 10 | 50 / 10 (default, configurable) | Matches paper default; configurable for experiments |
| CSV format | `dataset/trainDataset.csv` (headerless, 750 cols) | `training/data/merged_power_data_sub6GHz_avg_per_minute.csv` (headerless, 750 cols) | Same column count, different path |
| Data split | 7:3 train/test | 7:1.5:1.5 train/val/test | We need a held-out validation set for early stopping |
| Batch size | 32 (paper) / 256 (repo config) | 32 (default) | Uses paper value; configurable |
| AE training | Context AE (50→latent) + Future AE (10→latent) separately | Single LSE/LSD pair for future only | We skip the context AE; TSS-CC replaces it |
| Diffusion steps | Not explicitly stated in paper text; repo default 1000 | 1000 (default) | Configurable via `diffusion_steps` |
| Noise schedule | Not specified in paper; repo uses cosine | cosine | Matches the NoiseNet.py implementation |

### Key Design Difference: Context Autoencoder

The original repo (`Context2CondNew.py`) trains a separate context autoencoder that reconstructs `X` and produces a latent `z_ctx`. In the paper's actual architecture (Section IV), the TSS-CC stage directly produces the conditional representation `H_fusion` without an intermediate context autoencoder. The original repo's `ContextTransformerAE` implements an approximation of the TSS-CC with a simpler structure. Our adaptation follows the paper more faithfully: the TSS-CC stage (TemFE + SpeFE + SpaFE + FFM) produces `H_fusion` directly from incomplete `X`, and only the future autoencoder (LSE/LSD) is used for latent space encoding/decoding.

## 8. Known Limitations

- **Complex and slow training**: Diffusion training requires 1000+ epochs (or 1000+ steps per epoch) and is significantly slower than direct regression models
- **Multi-stage checkpoint management**: Three separate training stages with careful freeze/thaw logic; a failed mid-stage requires partial restart
- **Incomplete-observation masking must be standardized**: Masking strategy (random vs continuous) and missing rate must be consistent across all experiments for fair comparison
- **CPU training is impractical**: The Conv1D U-Net in NEN and multi-head attention in TSS-CC are compute-intensive; a GPU (>= 8 GB VRAM) is strongly recommended
- **Evaluation must distinguish two scenarios**: (a) Complete-observation prediction baseline, (b) Incomplete-observation recovery; the same model is evaluated under both
- **Diffusion sampling is slow at inference time**: 1000 reverse steps per prediction; techniques like DDIM or step distillation could be added later
- **Data mismatch**: Our CSV uses different frequency bands per node (reverse-engineered offsets), not the unified 85–335 MHz band described in the paper; performance may differ from published results
- **One-week claim vs 6839 rows**: The paper claims 10,080 time points (one week), but our CSV has only 6,839 rows after intersecting common minutes across nodes; the discrepancy should be documented in results

## 9. Implementation Notes

### Proposed file structure (future implementation):

| File | Responsibility |
|------|---------------|
| `dataset.py` | `TSSLCDDataset(tensor, t_in, t_out, missing_rate, mask_strategy)` — loads CSV, reshapes to (T, L, F), applies sliding window, simulates missing entries, returns (X, Y, mask) tuples |
| `model.py` | `PositionalEncoding`, `MultiHeadSelfAttention`, `TransformerEncoderLayer`, `TemporalFE`, `SpectralFE`, `SpatialFE`, `CrossAttentionFFM`, `TSSEncoder`, `LSE`, `LSD`, `NEN`, `DiffusionModel` |
| `train_autoencoder.py` | Loads config, creates dataset/dataloader, instantiates LSE+LSD, trains with MSE reconstruction, saves `ae_checkpoint.pth` |
| `train_tss_condition.py` | Loads pretrained LSE/LSD, instantiates TSS-CC, freezes autoencoder, extracts condition from X, minimizes `||LSE(Y) - H_fusion||^2`, saves `tss_checkpoint.pth` |
| `train_diffusion.py` | Loads pretrained LSE/LSD + TSS-CC, instantiates NEN+DiffusionModel, freezes everything else, runs forward diffusion + noise prediction, saves `diffusion_checkpoint.pth` |
| `evaluate.py` | Chronological test split, loads all checkpoints, runs full inference, computes RMSE/MAE/R² per horizon/node/bin, writes metrics CSV + plots |
| `inference.py` | Standalone inference entry point; takes CSV path + checkpoint dir, outputs predicted CSV |
| `utils.py` | Config loading (`load_config`), normalization (min-max or z-score), metric computation, checkpoint save/load |

### Smoke-test config proposal

For rapid validation before full training:

```yaml
# smoke_config.yaml
data:
  dataset_path: training/data/merged_power_data_sub6GHz_avg_per_minute.csv
  n_nodes: 3
  n_bins_per_node: 250
windowing:
  input_sequence_length: 10    # shorter window
  prediction_horizon: 3        # fewer future steps
  train_stride: 5              # fewer windows
  val_stride: 10
  test_stride: 10
split:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
preprocessing:
  missing_rate: 0.0            # complete obs first for debugging
model:
  latent_dim: 8                # smaller latent
  hidden_dim: 64               # smaller network
  attention_heads: 2
  diffusion_steps: 50          # few diffusion steps
training:
  autoencoder_epochs: 5
  tss_epochs: 5
  diffusion_epochs: 5
  batch_size: 16
```

A CC2-only smoke mode can be enabled by setting `n_nodes: 1` and slicing only columns 250–499 from the CSV.

## References

- S. Cheng, X. Li, X. Lin, H. Ding, and Y. Sun, "TSS-LCD: A Temporal–Spectral–Spatial-Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations," *IEEE Trans. Cogn. Commun. Netw.*, vol. 12, 2026.
- Original repository: [https://github.com/Demii-7/TSS-LCD](https://github.com/Demii-7/TSS-LCD) (cloned to `/tmp/opencode/TSS-LCD` during adaptation)
- Dataset: AERPAW sub-6 GHz spectrum monitoring dataset, [DOI: 10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn)
- AERPAW: [https://aerpaw.org/dataset/february-2022-cc1-cc2-lw1-spectrum-measurements/](https://aerpaw.org/dataset/february-2022-cc1-cc2-lw1-spectrum-measurements/)
