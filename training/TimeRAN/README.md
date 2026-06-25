# TimeRAN Spectrum Prediction — Integrated Pipeline

> **Based on:** *A Family of Open Time-Series Foundation Models for the Radio Access Network* — Panitsas, Tassiulas (arXiv 2026)
>
> **Built on:** MOMENT — a transformer-based time-series foundation model (`momentfm` library)
>
> **Repository:** https://github.com/panitsasi/TimeRAN (original upstream)
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

**TimeRAN is reused, not reconstructed.** We adapt the existing pretrained MOMENT foundation model to AERPAW spectrum forecasting by training lightweight per-chunk forecasting heads with a frozen backbone.

---

## Quick Start

### Prerequisites — Download Pretrained Checkpoint

```bash
pip install momentfm==0.1.4 gdown

mkdir -p training/TimeRAN/checkpoints/{small,base,large}

# NOTE: The upstream TimeRAN README mislabels these file IDs.
# ID 1fJNCk... is the small variant (d_model=512, ~145 MB), NOT base.
# ID 1gz23m... is the base variant (d_model=768, ~433 MB), NOT small.
gdown 1fJNCkufmfWC6zHecz10PUyreD0PhBOMJ -O training/TimeRAN/checkpoints/small/TimeRAN_small.pth
gdown 1gz23mmP4ZiNznCloObEaSlVaJH21fyxJ -O training/TimeRAN/checkpoints/base/TimeRAN_base.pth
gdown 1We9zE5BV6Iwkc_EKSAhP28B3wcM7RZRd -O training/TimeRAN/checkpoints/large/TimeRAN_large.pth
```

### Run

```bash
python3 training/TimeRAN/train_integrated.py
```

Outputs go to `training/results/TimeRAN/` by default:

```
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
models/<chunk_id>_timeran.pt
<chunk_id>_training_log.csv
```

---

## Scripts Reference

### `train_integrated.py` — Train and evaluate per chunk

The integrated runner trains one forecasting head per 200 MHz chunk on top of the frozen MOMENT backbone.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `training/common/config.yaml` | Shared config |
| `--output-dir` | `training/results/TimeRAN/` | Output directory |

The backbone size is controlled by `timeran.checkpoint_size` in the shared config (`small`, `base`, or `large`). The checkpoint path is derived automatically.

---

## 1. What the Model Is Intended to Do

TimeRAN provides a pretrained transformer backbone (built on MOMENT) that is adapted to spectrum forecasting. Given a window of past per-minute PSD measurements from a CC2 200 MHz chunk, it predicts the PSD values for multiple future time steps.

**Our task: multivariate multi-step spectrum forecasting per 200 MHz chunk.**

| Task | Why Not Suited |
|------|---------------|
| **Anomaly detection** | Reconstruction-based; does not predict future values |
| **Classification** | Single categorical label per window |
| **Imputation** | Fills missing values within a window |
| **Forecasting** | ✅ Correct: given `T_in` historical timesteps, predict `T_out` future timesteps |

---

## 2. Input Format

### 2.1 Per-Chunk Data Loading

```python
data = load_chunk(config, chunk)
train_input = data.splits[data.train_split].model_input   # (T_train, 200)
```

Each row is a 200-dimensional vector of normalized PSD values (dBm) for a single CC2 MHz chunk.

### 2.2 Tensor Shapes

```
Raw chunk:           (T, 200)
Window input (x):    (B, 200, 60)   — 200 channels × 60 lookback
Window target (y):   (B, 200, 60)   — 200 channels × 60 horizon
```

MOMENT expects `(batch, channels, sequence_length)`. Each frequency bin is treated as an independent channel.

---

## 3. Model Architecture

### 3.1 Overview

```
Chunk data: (T, 200)
    ↓
Sliding windows  ──►  X: (B, 200, 60)
                            ↓
              ┌──────────────────────────────┐
              │  MOMENT Backbone             │
              │  (T5 Transformer Encoder)    │
              │  Each channel = independent  │
              │  sequence in batch dim       │
              │  Shared nn.Linear embedder   │
              │     ↓                        │
              │  Output: (B, 200, N, d_model)│
              └──────────────────────────────┘
                            ↓
              ┌──────────────────────────────┐
              │  Forecasting Head            │
              │  Flatten: (B, 200, N*d_model)│
              │  Linear: N*d_model → T_out   │
              │     ↓                        │
              │  out.forecast: (B, 200, T_out)│
              └──────────────────────────────┘
                            ↓
                    Predictions
              (denormalized to dBm)
```

### 3.2 Backbone Architecture

Three variants:

| Variant | T5 Encoder | d_model | Layers | Heads | Params |
|---------|-----------|---------|--------|-------|--------|
| small | T5-small | 512 | 6 | 8 | ~40M |
| base  | T5-base  | 768 | 12 | 12 | ~125M |
| large | T5-large | 1024 | 24 | 16 | ~385M |

Key components:
- **Patching**: Unfold each channel into patches of length 8, stride 8 → N patches
- **PatchEmbedding**: Single shared `nn.Linear(8, d_model)` applied to every patch
- **T5 Encoder**: Processes each channel independently (stacked in batch dim: `[B*C, N, d_model]`)
- **RevIN**: Built-in per-channel reversible instance normalization

### 3.3 Forecasting Head

| Parameter | Value |
|-----------|-------|
| Type | `nn.Linear(N * d_model, T_out)` |
| Input | `(B, C, N * d_model)` — per-channel flattened patches |
| Output | `(B, C, T_out)` — per-channel predictions |
| Initialization | Random (checkpoint head weights always discarded) |

### 3.4 Training Mode

Default: **linear probing** (backbone frozen). Configurable via `timeran.training_mode`:
- `linear_probing` — freeze encoder + embedder, train only head
- `full_finetuning` — update all parameters (not recommended for small datasets)

---

## 4. Output Format

```
Raw output:         (B, 200, 60)   — MOMENT convention
Transpose:          (B, 60, 200)   — flat (time, freq)
Per horizon h:      (B, 200)       — extracted at index h-1
Denormalized:       dBm via shared pipeline
```

---

## 5. Training Pipeline

### 5.1 Data Loading

The `TimeRANDataset` creates sliding windows of length `lookback + max_horizon`:

```python
starts = np.arange(0, len(train_input) - window_len + 1)
# X: (start, start + lookback)
# Y: (start + lookback, start + lookback + max_horizon)
```

### 5.2 Loss, Optimizer, Training

| Parameter | Value |
|-----------|-------|
| Loss | MSE |
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Max LR | 1e-4 (OneCycleLR peak) |
| Weight decay | 0.0 |
| Gradient clip | 5.0 |
| Batch size | 1 (due to memory constraints) |
| Epochs | 10 |
| Val split | Last 10% of windows |

### 5.3 Mixed Precision

CUDA autocast is enabled automatically when running on GPU:

```python
if device.type == "cuda":
    with torch.amp.autocast("cuda"):
        out = model(x_enc=x, input_mask=input_mask)
```

### 5.4 Evaluation

For each horizon `h`:
1. Build lookback windows ending at `target_row - h`
2. Run frozen backbone + trained head
3. Extract `h`-th step from `out.forecast`
4. Denormalize and compute metrics

---

## 6. Changes from Main Branch (Standalone TimeRAN)

| Aspect | Main Branch | Integrate Branch |
|--------|------------|------------------|
| **Data loading** | `AERPAWDataset` loads 750-col CSV, splits 80/10/10 | Shared `load_chunk()`, single CC2 node, 200 bins per chunk |
| **Channels** | 750 (3 nodes × 250 bins) | 200 (single node chunk) |
| **T_in** | 128 | 60 (from shared `windowing.lookback`) |
| **T_out** | 16 | 60 (from shared `windowing.horizons`) |
| **Batch size** | 1 (due to 750 channels) | 1 (still conservative with 200 channels) |
| **Training script** | `train_head.py` (standalone) | `train_integrated.py` |
| **Config** | Per-model `config.yaml` | Shared `config.yaml`, `timeran:` section |
| **Normalization** | `revin_only` or `train_zscore` | Z-score via `load_chunk()` |
| **Inference** | Separate `evaluate.py` / `inference.py` | Built into `train_integrated.py` |

### Rationale for Changes

- **Reduced channels (750 → 200):** Per-chunk training on a single node dramatically reduces memory. The frozen backbone processes 200 independent channels per batch element instead of 750.
- **Shorter lookback (128 → 60):** Matches the shared config's lookback value. The AERPAW dataset's limited size makes longer lookbacks impractical.
- **Longer horizon (16 → 60):** The shared config evaluates at horizons 1/5/15/60. We set `T_out = 60` to cover the max horizon.
- **No separate evaluate.py:** The integrated runner evaluates inline, matching the other models' workflow.

---

## 7. Configuration Reference

All TimeRAN settings are under `timeran:` in `training/common/config.yaml`:

| Field | Default | Description |
|-------|---------|-------------|
| `batch_size` | 1 | Mini-batch size |
| `epochs` | 10 | Max training epochs |
| `learning_rate` | 1e-5 | Initial learning rate |
| `max_learning_rate` | 1e-4 | OneCycleLR peak |
| `weight_decay` | 0.0 | L2 weight decay |
| `gradient_clip_norm` | 5.0 | Max gradient norm |
| `checkpoint_size` | base | `small`, `base`, or `large` |
| `training_mode` | linear_probing | `linear_probing`, `full_finetuning` |

---

## 8. Known Limitations

1. **GPU memory with 200 channels × d_model** — still significant. Start with `base`, `batch_size=1`. Reduce `checkpoint_size` to `small` if needed.
2. **Backbone is frozen** — only the head is trained. Full fine-tuning may improve accuracy but risks overfitting with only 6839 timesteps.
3. **No cross-channel attention** — MOMENT processes channels independently. Each frequency bin is predicted primarily from its own history.
4. **Checkpoints must be downloaded** — the `.pth` files exceed GitHub limits and are hosted on Google Drive. Without them, raw MOMENT weights (no TimeRAN pretraining) are used instead.

---

## References

1. **TimeRAN paper:** I. Panitsas, L. Tassiulas, "A Family of Open Time-Series Foundation Models for the Radio Access Network," arXiv:2604.04271, 2026.
2. **TimeRAN repository:** https://github.com/panitsasi/TimeRAN
3. **MOMENT:** MOMENT: A Family of Open Time-series Foundation Models (ICML 2024). https://github.com/moment-timeseries-foundation-model/moment
4. **AERPAW dataset:** DOI: [10.5061/dryad.hmgqnk9zn](https://doi.org/10.5061/dryad.hmgqnk9zn)
