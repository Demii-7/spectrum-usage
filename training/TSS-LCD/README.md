# TSS-LCD Spectrum Prediction — Integrated Pipeline

> **Based on:** *TSS-LCD: A Temporal–Spectral–Spatial-Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations* — Cheng, Li, Lin, Ding, Sun (IEEE TCCN 2026)
>
> **Original repository:** https://github.com/Xlab2024/TSS-LCD
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

---

## Quick Start

```bash
python3 training/TSS-LCD/train_integrated.py
```

Outputs go to `training/results/TSS-LCD/` by default:

```
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
models/<chunk_id>_tss_lcd_autoencoder.pt
models/<chunk_id>_tss_lcd_tss.pt
models/<chunk_id>_tss_lcd_diffusion.pt
<chunk_id>_training_log.csv
```

---

## Scripts Reference

### `train_integrated.py` — 3-stage training and evaluation per chunk

The integrated runner trains all three stages sequentially per 200 MHz chunk:

1. **Autoencoder** — LSE + LSD reconstruct future windows
2. **TSS-CC** — Condition constructor predicts latent from lookback
3. **Diffusion** — Noise estimation network with conditioned denoising

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `training/common/config.yaml` | Shared config |
| `--output-dir` | `training/results/TSS-LCD/` | Output directory |

### `model.py` — All model components

| Component | Description |
|-----------|-------------|
| `TemporalFE` | Multi-head self-attention along time axis |
| `SpectralFE` | Multi-head self-attention along frequency axis |
| `SpatialFE` | Multi-head self-attention along spatial (node) axis |
| `TSSConditionConstructor` | All three TSS branches + cross-attention fusion + latent projection |
| `LatentSpaceEncoder` | Conv2D encoder compressing `(T_out, F)` → latent |
| `LatentSpaceDecoder` | ConvTranspose2D decoder reconstructing latent → `(T_out, F)` |
| `DiffusionModel` | Forward/reverse diffusion with cosine/linear schedule |
| `EnhancedNoiseNet` | Conv1D U-Net for noise prediction |

### `dataset.py` / `utils.py` — Library modules

Used by the standalone evaluation scripts. The integrated runner uses `load_chunk()` from the shared pipeline instead.

---

## 1. What the Model Is Intended to Do

TSS-LCD is a two-stage generative model for spectrum prediction:

- **Predict future spectrum PSD** across frequency bins given a window of past measurements
- **Handle incomplete observations** via configurable input masking
- **Preserve fine-grained RSS variations** (2–5 dBm) that regression models over-smooth
- **Model temporal, spectral, and spatial dependencies jointly** via dedicated self-attention branches

The integrated version trains per 200 MHz chunk on a single node (CC2) with `L=1, F=200`.

### Architecture Overview

```
Stage 1 (Autoencoder):  Y → LSE → z → LSD → Y_hat    (MSE reconstruction)
Stage 2 (TSS-CC):       X → TSS-CC → z_pred, LSE(Y) → z_target    (MSE in latent space)
Stage 3 (Diffusion):    z_target + noise + cond_z → NEN → noise_pred    (MSE noise)
Evaluation:             X → TSS-CC → cond_z → p_sample_loop → z_sample → LSD → Y_hat
```

---

## 2. Input Format

### 2.1 Per-Chunk Data Loading

```python
data = load_chunk(config, chunk)
train_input = data.splits[data.train_split].model_input   # (T_train, 200)
```

Each window pair `(X, Y)`:
- `X = data[i : i + T_in]` — lookback of shape `(T_in, 200)`, L=1
- `Y = data[i + T_in : i + T_in + T_out]` — future of shape `(T_out, 200)`, L=1

### 2.2 TSS Branch Input

The TSS-CC expects input of shape `(B, T_in, L, F)` where:
- `B` = batch size
- `T_in` = lookback (60)
- `L` = 1 (single node)
- `F` = 200 (frequency bins per chunk)

### 2.3 Autoencoder Input

The LSE/LSD expects input of shape `(B, T_out, D)` where `D = L * F = 200`.

Internally, the encoder reshapes to `(B, 1, T_out, D)` as a 2D image, applies Conv2D downsampling, pools to `(1, 1)`, and projects to latent_dim.

---

## 3. Model Architecture

### 3.1 TSS-CC — Condition Constructor

Three parallel multi-head self-attention branches (paper Fig. 3, Section IV-B):

#### Temporal Feature Extraction (TemFE)

```
Input: (B, T_in, L, F) → reshape (B, L*F, T_in)
                         → proj → pos_enc → TransformerEncoder
                         → output (B, L*F, hidden_dim)
```

#### Spectral Feature Extraction (SpeFE)

```
Input: (B, T_in, L, F) → reshape (B, T_in, L*F)
                         → proj → pos_enc → TransformerEncoder
                         → output (B, T_in, hidden_dim)
```

#### Spatial Feature Extraction (SpaFE)

```
Input: (B, T_in, L, F) → permute (B, F, T_in*L)
                         → proj → pos_enc → TransformerEncoder
                         → output (B, F, hidden_dim)
```

#### Feature Fusion Module (FFM)

Cross-attention between branches (paper Fig. 4):
- Query = spectral features (default dominant branch)
- Key/value = concatenation of temporal + spatial
- Output: `H_fusion` → pooled → projected to `latent_dim`

#### ConditionToLatentProjection

```python
z_pred = Linear(mean_pool(H_fusion))   # (B, latent_dim)
```

This projection head is our implementation bridge — the paper describes TSS-CC producing `H_fusion` but does not specify how it aligns with the diffusion latent `z0`.

### 3.2 Latent Space Encoder / Decoder (Conv2D)

Based on paper Figs. 5 and 7, using Conv2D (not the original repo's Conv1D).

#### Encoder (LSE)

```
Input: (B, T_out, D)  → reshape → (B, 1, T_out, D)
       ┌────────────────────────────────────┐
       │ Conv2d(1 → 32, k=3) + BN + ReLU    │
       │ MaxPool2d(2)                       │
       │ Conv2d(32 → 64, k=3) + BN + ReLU   │
       │ MaxPool2d(2)                       │
       │ Conv2d(64 → 128, k=3) + BN + ReLU  │
       │ MaxPool2d(2)                       │
       │ AdaptiveAvgPool2d(1, 1)            │
       │ Flatten → Linear(128 → latent_dim) │
       └────────────────────────────────────┘
Output: (B, latent_dim)
```

#### Decoder (LSD)

```
Input: (B, latent_dim)
       ┌────────────────────────────────────────┐
       │ Linear(latent_dim → 128*4*4)           │
       │ Reshape → (B, 128, 4, 4)               │
       │ ConvTranspose2d(128 → 64, k=4, s=2)    │
       │ BN + ReLU                              │
       │ ConvTranspose2d(64 → 32, k=4, s=2)     │
       │ BN + ReLU                              │
       │ ConvTranspose2d(32 → 1, k=4, s=2)      │
       │ Interpolate → (T_out, D)               │
       └────────────────────────────────────────┘
Output: (B, T_out, D)
```

### 3.3 Diffusion Model

Based on paper Fig. 6 and the original repo's `NoiseNet.py`.

#### Noise Schedule

Default: cosine schedule (from repo). Configurable to linear.

```
cosine:  β_n = clip(1 - α̅_n / α̅_{n-1}, 1e-8, 0.999)
         α̅_n = cos²((n/N + s) / (1+s) * π/2) / cos²(s / (1+s) * π/2)
```

#### Sinusoidal Time Embedding

```python
t_emb = [sin(t * ω_k), cos(t * ω_k)]   for k = 0..half_dim-1
```

#### Noise Estimation Network (NEN)

Conv1D U-Net with encoder-bottleneck-decoder and skip connections:

```
Input: [z_t, cond_proj(cond_z), t_emb] concatenated → (B, 2*latent_dim + time_embed_dim)

Encoder:
  Conv1d(1 → 64, k=3) + BN + ReLU + MaxPool
  Conv1d(64 → 128, k=3) + BN + ReLU + MaxPool

Bottleneck:
  Conv1d(128 → 256, k=3) + ReLU

Decoder:
  ConvTranspose1d(256 → 128, k=2, s=2) + skip from encoder
  Conv1d(128+128 → 64, k=3) + BN + ReLU
  ConvTranspose1d(64 → 32, k=2, s=2) + skip from encoder
  Conv1d(32+64 → 32, k=3) + BN + ReLU

Output: AdaptiveAvgPool1d → Linear(32 → latent_dim)
```

### 3.4 Forward / Reverse Diffusion

#### Forward (q_sample)

```python
z_t = sqrt(α̅_t) * z_0 + sqrt(1 - α̅_t) * ε
```

#### Reverse (p_sample)

Uses the paper's derived closed-form denoising step:

```python
z_{t-1} = A_t * z_t - B_t * ε_θ(z_t, cond_z, t) + σ_t * η
```

#### p_sample_loop

Full reverse chain from `z_N ~ N(0, I)` to `z_0`:

```python
z = randn(B, latent_dim)
for t in reversed(range(N)):
    z = p_sample(z, cond_z, t)
return z
```

---

## 4. Output Format

```
Full pipeline:  X → TSS-CC → cond_z → p_sample_loop → z_sample → LSD → Y_hat
Y_hat shape:    (B, T_out, 200)
Per horizon h:  (B, 200)    — extracted at index h-1
Denormalized:   dBm via shared pipeline
```

---

## 5. Training Pipeline

### 5.1 Stage 1: Autoencoder (LSE + LSD)

| Parameter | Value |
|-----------|-------|
| Input | `Y` — future window `(B, T_out, 200)` |
| Loss | MSE: `||Y - LSD(LSE(Y))||²` |
| Optimizer | Adam, lr=1e-4 |
| Epochs | 300 (configurable) |
| Early stopping | Patience 30 on val loss |

The LSE/LSD learn to compress and reconstruct future spectrum windows in a compact `latent_dim` space.

### 5.2 Stage 2: TSS-CC Condition Training

| Parameter | Value |
|-----------|-------|
| Input | `X` (lookback) + frozen `LSE(Y)` (target latent) |
| Loss | MSE: `||z_pred - z_target||²` where `z_target = LSE(Y)` |
| Optimizer | Adam, lr=1e-4 |
| Frozen | LSE, LSD (from Stage 1) |
| Epochs | 200 (configurable) |

The TSS-CC learns to predict the latent `z` from the lookback window, approximating the encoding of the future window.

### 5.3 Stage 3: Diffusion Training

| Parameter | Value |
|-----------|-------|
| Input | `X` (lookback) + `Y` (future), cond from frozen TSS-CC, latent from frozen LSE |
| Loss | MSE: `||ε - NEN(z_t, cond_z, t)||²` |
| Optimizer | Adam, lr=1e-4 |
| Frozen | LSE, LSD, TSS-CC |
| Epochs | 1000 (configurable) |

At each training step:
1. Encode `Y → z_target` (frozen LSE)
2. Encode `X → cond_z` (frozen TSS-CC)
3. Sample random `t`, noise `ε`
4. Compute `z_t = q_sample(z_target, t, ε)`
5. Predict noise: `ε_pred = NEN(z_t, cond_z, t)`
6. Loss: `MSE(ε_pred, ε)`

### 5.4 Evaluation

For each horizon `h`:
1. Build test windows `(X, Y)`
2. Run full pipeline: `X → TSS-CC → cond_z → p_sample_loop → z_sample → LSD → Y_hat`
3. Extract `Y_hat[:, h-1, :]` as the prediction at horizon `h`
4. Denormalize and compute metrics

---

## 6. Changes from Main Branch (Standalone TSS-LCD)

| Aspect | Main Branch | Integrate Branch |
|--------|------------|------------------|
| **Data loading** | `get_dataloaders()` from dataset.py — 750-col CSV, `(T, 3, 250)`, masking | Shared `load_chunk()`, single CC2 node, `(T, 200)`, no masking |
| **T_in / T_out** | Configurable independently (e.g., 50 / 10) | From shared config: `T_in = lookback = 60`, `T_out = max_horizons = 60` |
| **L** (nodes) | 3 (CC1, CC2, LW1) | 1 (CC2 only) |
| **F** (bins) | 250 | 200 (per chunk) |
| **Missing rate** | Configurable (e.g., 0.25) with masking | 0 (complete observation) |
| **Training** | 3 separate scripts (train_autoencoder.py, train_tss_condition.py, train_diffusion.py) | Single `train_integrated.py` that runs all 3 stages |
| **Diffusion steps** | 1000 | 1000 (configurable) |
| **Normalization** | Min-max (TSS-LCD Normalizer) | Z-score via `load_chunk()` |
| **Config** | Per-model `config.yaml` (178 fields) | Shared `config.yaml`, `tss_lcd:` section (28 fields) |
| **Evaluation** | Evaluate.py on all T_out steps at once | Per-horizon extraction (1, 5, 15, 60 min) |

### Rationale for Changes

- **Single-node training:** The integrated pipeline trains per 200 MHz chunk on CC2. The TSS-CC's SpatialFE branch still operates (with L=1, it collapses to a temporal feature), but all three TSS branches remain active.
- **No masking:** The integrated pipeline evaluates on complete observations to match the other models' setup. Masking can be added back by wrapping the dataset with the TSS-LCD dataset's masking logic.
- **Unified training script:** The 3-stage training is combined into a single runner for simplicity. Each stage saves its own checkpoint.
- **Reduced config surface:** Many TSS-LCD-specific params (masking rate, masking strategy, split ratios, etc.) are handled by the shared pipeline or kept with sensible defaults.

---

## 7. Configuration Reference

All TSS-LCD settings are under `tss_lcd:` in `training/common/config.yaml`:

| Field | Default | Description |
|-------|---------|-------------|
| `hidden_dim` | 256 | Transformer hidden dimension (all TSS branches) |
| `attention_heads` | 4 | Self-attention heads (must divide hidden_dim) |
| `ffn_dim` | 1024 | Transformer FFN inner dimension |
| `num_attention_layers` | 2 | Stacked encoder layers per branch |
| `dropout` | 0.1 | Dropout probability |
| `use_temporal_branch` | true | Enable TemFE |
| `use_spectral_branch` | true | Enable SpeFE |
| `use_spatial_branch` | true | Enable SpaFE |
| `latent_dim` | 32 | Latent space dimension (z) |
| `autoencoder_num_blocks` | 3 | Conv2D down/up blocks in LSE/LSD |
| `autoencoder_initial_channels` | 32 | Initial channels (doubled each block) |
| `diffusion_steps` | 1000 | Number of noise levels N |
| `noise_schedule` | cosine | `cosine` or `linear` |
| `nen_encoder_channels` | [64, 128] | NEN U-Net encoder channel counts |
| `nen_bottleneck_channels` | 256 | NEN bottleneck channels |
| `nen_decoder_channels` | [128, 64] | NEN decoder channel counts |
| `nen_kernel_size` | 3 | NEN Conv1D kernel size |
| `time_embed_dim` | 32 | Sinusoidal time embedding dimension |
| `batch_size` | 32 | Mini-batch size |
| `autoencoder_epochs` | 300 | Stage 1 epochs |
| `autoencoder_learning_rate` | 0.0001 | Stage 1 learning rate |
| `tss_epochs` | 200 | Stage 2 epochs |
| `tss_learning_rate` | 0.0001 | Stage 2 learning rate |
| `diffusion_epochs` | 1000 | Stage 3 epochs |
| `diffusion_learning_rate` | 0.0001 | Stage 3 learning rate |
| `weight_decay` | 0.0 | L2 weight decay |
| `gradient_clip_norm` | 5.0 | Max gradient norm |
| `patience` | 30 | Early stopping patience |

---

## 8. Deviations from Original Paper

| Aspect | Paper | Our Integrated Version | Reason |
|--------|-------|------------------------|--------|
| Dataset | Custom dataset, 10,080 timesteps, unified 85–335 MHz | AERPAW, 6,839 timesteps, per-node offsets | Different data source |
| Nodes | 3 fixed nodes | 1 per chunk (CC2) | Per-chunk single-node pipeline |
| Frequency | 250 bins (unified band) | 200 bins per chunk | 200 MHz per chunk at 1 MHz resolution |
| T_in | Configurable (e.g., 50) | 60 (shared lookback) | Shared config standardization |
| T_out | Configurable (e.g., 10) | 60 (max horizon) | Multi-horizon evaluation |
| Missing rate | 0–35% with masking | 0 (complete observations) | Match other models' setup |
| LSE architecture | Conv1D per original repo | Conv2D (paper-faithful) | Paper describes Conv2D |
| Training scripts | 3 separate scripts | 1 integrated runner (3 stages inline) | Simplified workflow |
| Evaluation | Evaluate.py tests all T_out steps at once | Per-horizon extraction from full T_out output | Multi-horizon framework |

### Repo-Specific Differences (vs Original TSS-LCD Repository)

The original repository at github.com/Xlab2024/TSS-LCD differs from the paper in several ways. Our implementation follows the paper:

| Aspect | Original Repository | Our Implementation |
|--------|-------------------|-------------------|
| LSE/LSD | Conv1D (treats freq as channels) | Conv2D (paper-faithful) |
| Condition mechanism | ContextTransformerAE (reconstructs X) | TSS-CC direct condition (paper-faithful) |
| Autoencoder | FutureAutoencoder (Conv1D) | LSE/LSD (Conv2D, paper-faithful) |
| Noise schedule | Cosine (same) | Cosine (same, configurable) |
| NEN architecture | Conv1D U-Net with skip connections | Same structure |
| Fusion | Cross-attention FFM | Same |

---

## 9. Known Limitations

1. **Slow training and inference** — diffusion requires 1000 reverse steps per prediction. Each chunk trains in ~hours (Stage 1 + 2 + 3).
2. **Multi-stage checkpoint management** — three stages with freeze/thaw logic; failure mid-stage requires partial restart.
3. **No input masking in integrated pipeline** — the TSS-LCD paper's core contribution is handling incomplete observations. The integrated version trains on complete observations only, matching the other models.
4. **SpatialFE branch has trivial input with L=1** — with a single node, the spatial branch sees a single "location," making it effectively another temporal feature extractor.
5. **Diffusion sampling is slow** — 1000 reverse steps at inference time. Techniques like DDIM could accelerate this.
6. **CPU training is impractical** — the Conv1D U-Net and multi-head attention require a GPU.

---

## References

1. **TSS-LCD paper:** S. Cheng, X. Li, X. Lin, H. Ding, Y. Sun, "TSS-LCD: A Temporal–Spectral–Spatial-Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations," IEEE TCCN, 2026.
2. **Original repository:** https://github.com/Xlab2024/TSS-LCD
3. **AERPAW dataset:** DOI: [10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn)
