# TSS-LCD Spectrum Prediction — Adaptation

Adaptation plan for integrating TSS-LCD (Temporal–Spectral–Spatial Guided Latent Conditional Diffusion Model) into the AERPAW spectrum prediction pipeline. The **paper** (IEEE TCCN 2026) is the primary architecture reference. Where the original repository differs from the paper, those differences are documented separately under *Repo-Specific Implementation Notes*.

## Quick Start

```bash
# 1. Activate environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install torch numpy pandas scipy scikit-learn pyyaml matplotlib

# 3. Place CSV at expected path
#    training/data/merged_power_data_sub6GHz_avg_per_minute.csv

# -------------------------------------------------------------------
# 4. Run training (three stages)
#    NOTE: These are future commands only. The scripts do not yet
#    exist. They are documented here to define the intended
#    workflow before implementation begins.
# -------------------------------------------------------------------
python3 training/TSS-LCD/train_autoencoder.py    # Stage 1 — latent autoencoder
python3 training/TSS-LCD/train_tss_condition.py  # Stage 2 — TSS condition stage
python3 training/TSS-LCD/train_diffusion.py      # Stage 3 — LCD noise-estimation

# 5. Evaluate
python3 training/TSS-LCD/evaluate.py
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `dataset.py` | Dataset class, windowing, masking, reshaping |
| `model.py` | All model components (TSS-CC, LSE, LSD, NEN) |
| `train_autoencoder.py` | Stage 1: latent autoencoder pretraining (LSE + LSD) |
| `train_tss_condition.py` | Stage 2: TSS-CC condition training (TemFE, SpeFE, SpaFE, FFM) |
| `train_diffusion.py` | Stage 3: LCD noise-estimation training (NEN only) |
| `evaluate.py` | Evaluation on test split with metric computation |
| `inference.py` | Online inference from incomplete observations |
| `utils.py` | Normalization, metrics, checkpointing, config loading |

## File Structure

```text
training/TSS-LCD/
├── README.md              # This file
├── config.yaml            # Full configuration (all tunable parameters)
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

All configurable options live in `config.yaml`. Every tunable hyperparameter is exposed there; no major value should be hardcoded in scripts unless it is a fixed architectural constraint from the paper or repo. See `config.yaml` for the full field list with inline YAML comments.

Key configurable groups:

- **Data**: dataset path, number of nodes, frequency bins per node, node names, selected node subset, CC2-only smoke mode
- **Windowing**: input sequence length, prediction horizon, train/val/test strides
- **Split**: train/val/test ratios, chronological split flag, random seed
- **Preprocessing**: normalization method, missing rate, masking strategy, zero-padding toggle, complete-observation baseline mode
- **Model architecture**: latent dimension, hidden dimension, attention heads, FFN dimension, number of attention layers, diffusion steps, noise schedule, per-branch toggles, autoencoder channels/depth, NEN U-Net channels/depth
- **Training**: per-stage epochs, learning rates, batch size, optimizer, gradient clipping, weight decay, checkpoint path, Stage 2 objective strategy
- **Evaluation**: metrics list, evaluation horizons, output directory
- **Device**: auto / cuda / cpu

## 1. What the Model Is Intended to Do

TSS-LCD is a **two-stage generative model** for spectrum prediction under incomplete historical observations, as described in the paper:

- **Predict future spectrum RSS/PSD** across multiple nodes and frequency bins given a window of past measurements
- **Handle incomplete historical observations** where entries are missing (zero-padded) due to partial-band scanning, hardware constraints, or scheduling policies
- **Preserve fine-grained RSS variations** (2–5 dBm fluctuations) that regression-based models typically over-smooth
- **Model temporal, spectral, and spatial dependencies jointly** using dedicated self-attention branches per dimension, then guide a latent diffusion process

The paper reports TSS-LCD outperforming ConvLSTM, CSMA, STS-PredNet, DSwinLSTM, and 3D-SwinSTB under missing rates from 15% to 35%, for both random and continuous missing patterns.

### Target Architecture (Paper-Faithful)

The implementation targets the architecture described in the paper (Section IV, Figs. 3–7):

1. **TSS-CC Stage** — Three parallel multi-head self-attention branches (TemFE, SpeFE, SpaFE) extract temporal, spectral, and spatial dependencies from the incomplete input `X`. A cross-attention Feature Fusion Module (FFM) integrates them into a unified conditional representation `H_fusion`.
2. **LCD-SP Stage** — A Latent Space Encoder (LSE) compresses the ground-truth future `Y` into a latent `z0`. A forward diffusion process adds noise to `z0`. A Noise Estimation Network (NEN) — conditioned on `H_fusion` and the diffusion step — predicts the added noise. The reverse denoising process reconstructs `z0_hat`, and a Latent Space Decoder (LSD) maps it back to spectrum domain `Y_hat`.

No intermediate context autoencoder is used in the paper-faithful design. The condition comes directly from TSS-CC.

## 2. Input Format

### Raw CSV — AERPAW-Specific Layout

```text
Path:    training/data/merged_power_data_sub6GHz_avg_per_minute.csv
Shape:   (6839, 750)
Layout:  [CC1_cols_0_249 | CC2_cols_250_499 | LW1_cols_500_749]
```

The CSV has **no header row**. Each row is per-minute averaged RSS in dBm across three fixed AERPAW nodes. Column ordering:

| Columns | Node | Approx. frequency range | Note |
|---------|------|------------------------|------|
| 0–249 | CC1 | ~1347–1362 MHz | Per-node selected 250-bin offset (bin 21000) |
| 250–499 | CC2 | ~2082–2097 MHz | Per-node selected 250-bin offset (bin 33250) |
| 500–749 | LW1 | ~1737–1752 MHz | Per-node selected 250-bin offset (bin 27500) |

**Important:** The paper describes a unified 85–335 MHz band with 250 uniformly discretized bins for all three nodes. Our processed CSV uses **different frequency ranges per node**, determined by reverse-engineering the original TSS-LCD repository CSV (see `training/build_training_csv.py`). This is an **AERPAW-specific adaptation** — do not claim exact reproduction of the paper's frequency band.

### Paper Setting vs Our Setting

| Property | Paper | Our CSV | Impact |
|----------|-------|---------|--------|
| Time points | 10,080 (1 week) | 6,839 (~4.75 days) | Fewer total windows; window count = 6839 − T_in − T_out + 1 |
| Frequency coverage | 85–335 MHz (unified) | Per-node offsets (L-band / lower S-band) | Model still receives 250 bins per node; spatial branch captures per-location differences |
| Sensors | 3 fixed nodes (CC1, CC2, LW1) | 3 fixed nodes (CC1, CC2, LW1) | Matches paper |

The implementation must use the actual processed CSV shape (6839 rows), not assume 10080.

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
- Chronological train/val/test split (ratios configurable)

### Incomplete Observation Simulation

Missing entries are simulated on input `X` only (target `Y` is always complete):

- **Random missing**: Each entry has probability `missing_rate` of being zero-padded
- **Continuous missing**: A contiguous block of length proportional to `missing_rate` is zero-padded
- A configurable binary observation mask tensor is returned alongside `X` and `Y` to support masked-loss training if needed
- A `complete_observation_baseline` mode (`missing_rate=0`) should be supported first for pipeline debugging

## 3. Model Architecture (Paper-Faithful Target)

TSS-LCD is a two-stage network (paper Fig. 3). This section describes the **paper architecture** as the implementation target. See *Repo-Specific Implementation Notes* for differences in the original repository.

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

- Cross-attention: temporal feature as query, spectral as key, reshaped spatial as value
- Residual connections + layer norm + FFN
- Output: `H_fusion ∈ R^(T_in × (L·F))` — the unified conditional representation

### Stage 2: LCD-SP — Latent Diffusion Prediction

#### Latent Space Encoder (LSE)

- 2D convolutional encoder with stacked downsampling blocks (Conv2d + BN + ReLU)
- Number of blocks and channel progression are configurable
- Maps ground-truth `Y ∈ R^(T_out × L × F)` → latent `z0 ∈ R^(1 × d_latent)`

#### Forward Diffusion Process

- Adds Gaussian noise progressively over `N` steps: `zn = sqrt(ᾱ_n) · z0 + sqrt(1 − ᾱ_n) · ε`
- Noise schedule is configurable (cosine default, linear alternative)

#### Noise Estimation Network (NEN)

- Takes concatenation of `[zn, W_cond(H_fusion), time_embedding(n)]`
- Conv1D-based U-Net with downsampling, bottleneck, upsampling, and skip connections
- Number of encoder/decoder channels and bottleneck depth are configurable
- **Note:** The exact Conv1D U-Net skip-connection implementation should be verified against the repository (`NoiseNet.py`) before scripting

#### Reverse Denoising Process

- Starts from pure Gaussian noise `zN`
- Iteratively denoises using NEN: `z_{n-1} = µ_n + σ_n · η`
- `N` steps total, guided by `H_fusion` at each step
- Recovers denoised latent `ẑ_0`

#### Latent Space Decoder (LSD)

- Mirrors LSE structure: FC layer + ConvTranspose2d upsampling blocks
- Maps `ẑ_0` → `Ŷ ∈ R^(T_out × L × F)`

## 4. Output Format

```text
Shape:   (T_out, L, F)   e.g. (10, 3, 250)
Flatten: (T_out, 750)
```

Output is the predicted future RSS/PSD in the same scale as the preprocessed input (normalized or raw dBm depending on config). During evaluation, predictions are inverse-transformed to dBm for metric computation.

## 5. Training Pipeline

Three separate training stages, each with its own script and checkpoint. All stage-specific hyperparameters (epochs, learning rate, etc.) are independently configurable.

### Stage 1: Latent Autoencoder Pretraining

```text
Script:  train_autoencoder.py
Input:   Y (complete future ground truth)
Loss:    MSE reconstruction: ||Y - LSD(LSE(Y))||²
Freezes: none
Output:  checkpoints/ae_lse.pth, checkpoints/ae_lsd.pth
```

The LSE and LSD are trained end-to-end to compress and reconstruct future spectrum data in a compact latent space. The number of downsampling blocks and channels are configurable.

### Stage 2: TSS Condition Stage Training

```text
Script:  train_tss_condition.py
Input:   X (incomplete historical) + Y (complete future)
Freezes: LSE, LSD (from Stage 1)
Output:  checkpoints/tss_cc.pth
```

The paper states that the TSS modules are trained to provide reliable conditional information for the diffusion stage while LSE/LSD remain frozen. However, the exact supervised objective for `H_fusion` requires verification — the paper does not specify a direct loss between `H_fusion ∈ R^(T_in × L·F)` and `z0 ∈ R^(1 × d_latent)`, and those shapes do not naturally align.

Implementation options (set via `config.yaml` → `training.tss_condition_objective`):

1. **`projection_to_latent`** (default target): Add a learnable projection head (e.g., pooling + FC) that maps `H_fusion` → `z_pred ∈ R^(d_latent)`, then train with MSE against `z0 = LSE(Y)`.
   * **Note:** This projection head is **our implementation bridge**, not something explicitly specified in the paper. It exists to reconcile the dimensional mismatch between `H_fusion ∈ R^(T_in × L·F)` and the latent representation `z0 ∈ R^(d_latent)`.
2. **`joint_with_diffusion`**: Skip standalone Stage 2. Train TSS-CC jointly with NEN during Stage 3 (the condition is only supervised by the diffusion loss).
3. **`repo_context_ae`**: Fall back to the repository's context-autoencoder strategy if the paper-faithful approach proves unstable.

The final Stage 2 objective must be confirmed before scripting. Individual TSS branches can be toggled via config for ablation.

### Stage 3: LCD Diffusion Training

The freezing behavior during Stage 3 depends on `training.tss_condition_objective`:

- **`projection_to_latent`** (default): TSS-CC is frozen after Stage 2. Only the NEN is trained.
- **`joint_with_diffusion`**: No separate Stage 2 runs. TSS-CC is trained jointly with the NEN in Stage 3. LSE and LSD remain frozen in both cases.

```text
Script:  train_diffusion.py
Input:   X (incomplete historical) + Y (complete future)
Loss:    MSE between true and predicted noise: ||ε - NEN(z_n, n, H_fusion)||²
Freezes: LSE, LSD (always); TSS-CC only when objective = projection_to_latent
Output:  checkpoints/diffusion.pth
```

The frozen LSE encodes the target latent, and the NEN (and optionally TSS-CC) learns to predict the injected noise at each diffusion step.

### Evaluation / Inference

```text
Script:   evaluate.py / inference.py
Process:  X → TSS-CC → H_fusion → (reverse diffusion) → Ŷ
Metrics:  RMSE, MAE, R² per evaluation horizon, per node, per frequency bin
```

## 6. Assumptions and Design Decisions

- **CSV has no header**: The loader reads raw float32 values directly
- **Chronological split**: Always chronological for time series; no random shuffle across time
- **Masking applied only to input X**: Target Y is always fully observed (matching the paper setup)
- **Complete-observation baseline first**: Train with `missing_rate=0` first to verify the pipeline before introducing masking
- **LSE/LSD operate on future data only**: They compress Y, never X; during inference they decode the denoised latent
- **TSS-CC outputs a condition, not a direct prediction**: The condition guides diffusion rather than producing a direct forecast
- **Normalization fit on training split only**: Avoids data leakage from test/val splits
- **Batch size = 32 by default** (paper value), configurable because memory and training speed depend on GPU availability
- **All tunable hyperparameters are in config.yaml**: No values hardcoded in scripts unless they are fixed architectural constraints

## 7. AERPAW Dataset Adaptation

### Row-Count Mismatch

| Source | Time Points | Duration |
|--------|-------------|----------|
| Paper | 10,080 | One week of continuous collection per node |
| Our CSV | 6,839 | Common minutes across CC1, CC2, LW1 intersections |

The paper's 10,080 value assumes individual per-node time series. Our CSV contains only the intersection of minute buckets that exist in all three nodes simultaneously, which reduces the count to 6,839. The implementation uses the actual CSV shape — it does not assume 10080.

### Frequency-Band Mismatch

| Source | Frequency Coverage | Bins | Per-Node |
|--------|-------------------|------|----------|
| Paper | 85–335 MHz (unified) | 250 uniform bins | Same 250 bins for all nodes |
| Our CSV | Per-node offsets (CC1: ~1347, CC2: ~2082, LW1: ~1737 MHz) | 250 bins each | Different frequency ranges per node |

This is an **AERPAW-specific adaptation**. The per-node offsets were reverse-engineered from the original TSS-LCD repository CSV (see `training/build_training_csv.py` for details). The model architecture itself is agnostic to the actual frequency values — it receives `F=250` bins per node regardless. However, this means results are not directly comparable to the paper's published numbers.

### What Remains the Same

- 3 nodes (CC1, CC2, LW1) — matches paper
- 250 frequency bins per node — matches paper
- 750 total feature dimension — matches paper
- Per-minute sampling interval — matches paper

## 8. Repo-Specific Implementation Notes

The original repository at `github.com/Xlab2024/TSS-LCD` contains implementations that differ from the paper in several ways. These are **not** part of the paper-faithful target architecture, but are documented here for reference since they may influence implementation decisions. (A local working copy was cloned to `/tmp/opencode/TSS-LCD` during adaptation.)

### Context Autoencoder (Not in Paper)

The repo's `Context2CondNew.py` implements a `ContextTransformerAE` — a Transformer-based autoencoder that reconstructs `X` and produces a latent `z_ctx`. This module is **not described in the paper**. The paper's TSS-CC produces `H_fusion` directly from `X` without a reconstruction objective.

- **Paper**: TemFE + SpeFE + SpaFE → FFM → `H_fusion` (no reconstruction loss on X)
- **Repo**: `ContextTransformerAE` encodes `X` into latent `z_ctx` with MSE reconstruction of `X`, then `z_ctx` is used as condition
- **Our target**: Paper-faithful TSS-CC (direct condition, no context AE)

### Future Autoencoder (Conv1D vs Paper's Conv2D)

The repo's `F2Cnet.py` implements `FutureAutoencoder` using **Conv1d** layers (treating frequency bins as channels and time as sequence length). The paper describes the LSE/LSD using **Conv2d** layers operating on the `(T_out, L, F)` tensor directly.

- **Paper**: LSE uses stacked Conv2d downsampling blocks (Fig. 5); LSD uses ConvTranspose2d upsampling blocks (Fig. 7)
- **Repo**: `FutureAutoencoder` uses Conv1d with `kernel_size=4, stride=2` across the time dimension
- **Our target**: Paper-faithful Conv2d-based LSE/LSD

### Noise Schedule Default

The repo's `NoiseNet.py` implements a **cosine** beta schedule. The paper does not explicitly specify the noise schedule type. We adopt the repo's cosine schedule as default, with a linear schedule available as a configurable alternative.

### NEN Architecture

The repo's `EnhancedNoiseNet` uses a Conv1D U-Net with:
- Encoder: Conv1d(1→64), MaxPool, Conv1d(64→128), MaxPool
- Bottleneck: Conv1d(128→256)
- Decoder: ConvTranspose1d(256→128), Conv1d(128→64), Linear(64*T → latent_dim)
- Skip connections from encoder to decoder at each level

The paper's Fig. 6 also describes a Conv1D U-Net but specific channel counts and skip-connection topology may differ. **The exact skip-connection implementation should be verified against the paper's Fig. 6 before scripting.**

### Diffusion Model

The repo's `DiffusionModel` uses:
- A learned `cond_proj` (Linear) to project `H_fusion` to `latent_dim`
- Sinusoidal time embedding (dim=32)
- Concatenation of `[z_t, cond_proj(cond), time_emb]` as input to NEN
- Cosine beta schedule with cumulative product computation

This is largely consistent with the paper description and can be used as reference.

## 9. Known Limitations

- **Complex and slow training**: Diffusion training can be substantially slower because each training step samples diffusion noise, and inference may require many reverse denoising steps (e.g., 1000). This is inherent to iterative generative models and differs from one-shot regression.
- **Multi-stage checkpoint management**: Three separate training stages with careful freeze/thaw logic; a failed mid-stage requires partial restart
- **Incomplete-observation masking must be standardized**: Masking strategy (random vs continuous) and missing rate must be consistent across all experiments for fair comparison
- **CPU training is impractical**: The Conv1D U-Net in NEN and multi-head attention in TSS-CC are compute-intensive; a GPU (>= 8 GB VRAM) is strongly recommended
- **Evaluation must distinguish two scenarios**: (a) Complete-observation prediction baseline, (b) Incomplete-observation recovery; the same model is evaluated under both
- **Diffusion sampling is slow at inference time**: 1000 reverse steps per prediction; techniques like DDIM or step distillation could be added later
- **Frequency-band mismatch vs paper**: Our CSV uses different frequency bands per node (reverse-engineered offsets), not the unified 85–335 MHz band described in the paper; performance may differ from published results
- **One-week claim vs 6839 rows**: The paper claims 10,080 time points (one week), but our CSV has only 6,839 rows after intersecting common minutes across nodes; this affects total window count and should be documented in results

## 10. Items to Verify Before Scripting

1. **TSS-CC paper description vs implementation**: Confirm from paper Section IV-B that TemFE, SpeFE, SpaFE all use identical multi-head self-attention (just permuted axes) and that FFM uses temporal→query, spectral→key, spatial→value cross-attention
2. **NEN skip-connection details**: Verify the exact U-Net skip wiring against the paper's Fig. 6 and repo's `NoiseNet.py`. The repo uses simple concatenation skip connections — confirm this matches the paper.
3. **LSE/LSD Conv2D parameterization**: Determine the number of downsampling blocks, kernel sizes, stride, and channel progression from the paper's Fig. 5/7 (the paper shows "multiple stacked blocks" without exact counts)
4. **Conditioning mechanism in NEN**: Verify whether `H_fusion` is projected and concatenated directly with `z_t` (as in the repo) or if there is a different conditioning mechanism (e.g., adaptive layer norm) described in the paper
5. **FFM cross-attention dimensionality**: Confirm the exact reshaping of spatial features before cross-attention (paper says `H_spat` is reshaped from `L×(T·F)` to `F×(T·L)`)
6. **Stage 2 objective**: The paper does not specify an explicit loss between `H_fusion` and `z0`; their shapes differ. Decide which implementation option to adopt from the three listed in the Stage 2 section (`projection_to_latent`, `joint_with_diffusion`, or `repo_context_ae`) before scripting.
7. **Diffusion step count in paper**: The paper text mentions 500 epochs total training (line 1007) but does not explicitly state the number of diffusion steps N. The repo defaults to 1000. Verify if the paper provides a specific N value.
8. **Normalization details**: The paper mentions min-max normalization to [0,1] (implied by the min-max scaler in `DataSetPrepare.py`) but does not specify this explicitly. Confirm whether z-score or min-max is more appropriate.

## 11. Implementation Notes

### Proposed file responsibilities

| File | Responsibility |
|------|---------------|
| `dataset.py` | `TSSLCDDataset` — loads CSV, reshapes to (T, L, F), applies sliding window with configurable strides, simulates missing entries, returns (X, Y, mask) tensors |
| `model.py` | `PositionalEncoding`, `MultiHeadSelfAttention`, `TransformerEncoderLayer` (for TemFE/SpeFE/SpaFE), `TemporalFE`, `SpectralFE`, `SpatialFE`, `CrossAttentionFFM`, `TSSEncoder`, `LSE` (Conv2d), `LSD` (ConvTranspose2d), `SinusoidalTimeEmbedding`, `NEN` (Conv1D U-Net), `DiffusionModel` |
| `train_autoencoder.py` | Loads config, creates dataset, instantiates LSE+LSD, trains with MSE reconstruction, saves checkpoints |
| `train_tss_condition.py` | Loads pretrained LSE/LSD, instantiates TSS-CC components, freezes autoencoder, trains condition mapping |
| `train_diffusion.py` | Loads LSE/LSD + TSS-CC, instantiates NEN + DiffusionModel, freezes prior stages, trains noise prediction |
| `evaluate.py` | Chronological test split, loads all checkpoints, full inference, computes metrics per horizon/node/bin, writes CSV + plots |
| `inference.py` | Standalone entry point for online inference from incomplete observations |
| `utils.py` | Config loading, normalization (min-max / z-score), metric computation (RMSE/MAE/R²), checkpoint save/load, seed setting |

### Smoke-test config

For rapid validation before full training, use a reduced config (example in the smoke test section). A CC2-only mode can be enabled by setting `n_nodes: 1` in config and slicing only columns 250–499 from the CSV.

### Reproducibility

Set a fixed random seed in config for Python, NumPy, and PyTorch to ensure reproducible data splits and masking patterns across runs.

## References

- S. Cheng, X. Li, X. Lin, H. Ding, and Y. Sun, "TSS-LCD: A Temporal–Spectral–Spatial-Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations," *IEEE Trans. Cogn. Commun. Netw.*, vol. 12, 2026.
- Original repository (paper authors): [https://github.com/Xlab2024/TSS-LCD](https://github.com/Xlab2024/TSS-LCD)
- Local working clone (during adaptation): `/tmp/opencode/TSS-LCD`
- Dataset: AERPAW sub-6 GHz spectrum monitoring dataset, [DOI: 10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn)
- AERPAW: [https://aerpaw.org/dataset/february-2022-cc1-cc2-lw1-spectrum-measurements/](https://aerpaw.org/dataset/february-2022-cc1-cc2-lw1-spectrum-measurements/)
