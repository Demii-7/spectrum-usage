# TimeRAN Spectrum Prediction — Adaptation

> **Based on:** *A Family of Open Time-Series Foundation Models for the Radio Access Network* — Panitsas, Tassiulas (arXiv 2026)
>
> **Built on:** MOMENT — a transformer-based time-series foundation model (`momentfm` library)
>
> **Repository:** https://github.com/panitsasi/TimeRAN (original upstream)
>
> **Target dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022)

##### **TimeRAN is reused, not reconstructed.** We are adapting the existing pretrained foundation model used in TIMERAN to AERPAW spectrum prediction. While TIMERAN has many task-specific heads, Forecasting is the most suitable task for spectrum prediction.  
---

## Quick Start

### Setup

```bash
cd /home/cc/spectrum-usage
pip install momentfm torch numpy pyyaml tqdm scikit-learn gdown
```

### Download Pretrained Checkpoint

Model checkpoints are not included in the repository due to GitHub file size limits. They must be downloaded from Google Drive using `gdown`:

```bash
# Create checkpoint directories
mkdir -p training/TimeRAN/checkpoints/{small,base,large}

# Download checkpoints from the upstream TimeRAN repository
cd training/TimeRAN/checkpoints

gdown 1fJNCkufmfWC6zHecz10PUyreD0PhBOMJ -O base/TimeRAN_base.pth
gdown 1gz23mmP4ZiNznCloObEaSlVaJH21fyxJ -O small/TimeRAN_small.pth
gdown 1We9zE5BV6Iwkc_EKSAhP28B3wcM7RZRd -O large/TimeRAN_large.pth

cd /home/cc/spectrum-usage
```

### Train Forecasting Head (Linear Probing)

```bash
python3 training/TimeRAN/train_head.py
```

### Evaluate

```bash
python3 training/TimeRAN/evaluate.py \
    --checkpoint training/TimeRAN/checkpoints/best_model.pt
```

### Run Inference on New Data

```bash
python3 training/TimeRAN/inference.py \
    --checkpoint training/TimeRAN/checkpoints/best_model.pt \
    --input /path/to/new_measurements.csv \
    --output predictions.csv
```

---

## Scripts Reference

### `train_head.py` — Train forecasting head

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration |
| `--mode` | from config | `linear_probing`, `full_finetuning`, or `lora` |
| `--batch-size` | from config | Override batch size |
| `--epochs` | from config | Override max epochs |
| `--lr` | from config | Override learning rate |

Output: `checkpoints/best_model.pt`, `checkpoints/last_model.pt`, `checkpoints/normalization_stats.pt`.

### `evaluate.py` — Evaluate a trained model

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` checkpoint (required) |
| `--config` | from checkpoint | Path to config |
| `--output` | `evaluation/` | Output directory |

Output: `evaluation/metrics.json`, `evaluation/predictions.csv`, `evaluation/spectrogram_*.png`.

### `inference.py` — Predict on new CSV data

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` checkpoint (required) |
| `--input` | — | Input CSV (750 cols) (required) |
| `--output` | `predictions.csv` | Output CSV path |

### `dataset.py` — Data loading and preprocessing (library)

| Function | Returns | Description |
|----------|---------|-------------|
| `create_datasets(csv_path, n_features, ...)` | `(train_ds, val_ds, test_ds, stats)` | Loads CSV, normalizes, windows, splits chronologically |
| `AERPAWDataset(data, t_in, t_out, indices)` | PyTorch `Dataset` | Returns `(X, Y)` as `(C, T_in)` and `(C, T_out)` |
| `load_csv(path)` | `ndarray (T, C)` | Loads CSV via `numpy.loadtxt` |
| `denormalize(data, mean, std)` | `ndarray` | Reverses z-score normalization |

### `utils.py` — Metrics and helpers (library)

| Function | Description |
|----------|-------------|
| `compute_metrics(pred, target)` | Returns `{"rmse", "mae"}` |
| `compute_metrics_per_horizon(pred, target)` | Per-timestep metrics |
| `compute_metrics_per_node(pred, target, names)` | Per-node metrics |
| `save_checkpoint(path, model, optimizer, ...)` | Saves model weights, optimizer state, config, norm stats |
| `load_checkpoint(path, device)` | Loads a saved checkpoint |
| `get_device(device_str)` | Returns `torch.device` |
| `set_seed(seed)` | Seeds all RNGs |

---

## File Structure

```
training/TimeRAN/
├── README.md                # This file
├── config.yaml              # Configuration / hyperparameters
├── dataset.py               # AERPAWDataset, data loading, normalization, windowing
├── train_head.py            # Training loop, checkpointing, linear probing / fine-tuning
├── evaluate.py              # Test set evaluation, metrics, visualizations
├── inference.py             # Predict on new data, save predictions
└── utils.py                 # Helpers: normalization, metrics, seeding, device setup
```

### State Flow

```
config.yaml
    │
    ▼
dataset.py ──► train_head.py ──► model.pt (head weights)
                                    │
                                    ▼
                               evaluate.py ──► metrics, plots, predictions.csv
                                    │
                                    ▼
                               inference.py ──► predictions on new data
```

---

## Configuration Reference

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| Data | `dataset_path` | `training/data/merged_power_data_sub6GHz_avg_per_minute.csv` | Input CSV path |
| Data | `n_features` | 750 | Total columns (3 nodes × 250 bins) |
| Data | `n_nodes` | 3 | Number of sensor nodes |
| Data | `bins_per_node` | 250 | Frequency bins per node |
| Data | `node_names` | `["CC1","CC2","LW1"]` | Node labels for plots/metrics |
| Preprocessing | `normalization` | `revin_only` | `revin_only` or `train_zscore` |
| Preprocessing | `fit_on_train_only` | true | Compute norm stats on training set only |
| Windowing | `input_sequence_length` | 128 | Past minutes (T_in) |
| Windowing | `prediction_horizon` | 16 | Future minutes (T_out) |
| Windowing | `stride` | 16 | Window stride |
| Split | `train_ratio` | 0.8 | Training set fraction |
| Split | `val_ratio` | 0.1 | Validation set fraction |
| Split | `test_ratio` | 0.1 | Test set fraction |
| Split | `chronological_split` | true | Chronological split |
| Model | `checkpoint_size` | `base` | `small`, `base`, or `large` |
| Model | `checkpoint_path` | `checkpoints/base/TimeRAN_base.pth` | Local TimeRAN `.pth` |
| Model | `task` | `forecasting` | Downstream task |
| Model | `freeze_backbone` | true | Freeze encoder + embedder |
| Model | `train_head_only` | true | Train only the head |
| Training | `batch_size` | 1 | Mini-batch size (start small for 750 channels) |
| Training | `epochs` | 10 | Max training epochs |
| Training | `learning_rate` | 1e-5 | Initial learning rate |
| Training | `max_learning_rate` | 1e-4 | OneCycleLR peak |
| Training | `optimizer` | `adam` | Optimizer |
| Training | `max_norm` | 5.0 | Gradient clipping |
| Training | `seed` | 42 | Random seed |
| Evaluation | `metrics` | `["rmse", "mae"]` | Metrics to report |
| Evaluation | `eval_horizons` | `[1, 4, 8, 16]` | Per-horizon reporting steps |
| Device | `device` | `auto` | `cuda:N`, `cpu`, or `auto` |

---

## 1. What the Model Is Intended to Do

TimeRAN is a family of open time-series foundation models pretrained on the TimeRAN DataPile, a large corpus of RAN telemetry. It provides a pretrained transformer backbone (built on MOMENT) that can be adapted to downstream tasks via lightweight task-specific heads.

**Our task: multivariate multi-step spectrum forecasting.**

Given a window of past per-minute PSD measurements across 750 features (3 nodes × 250 frequency bins), predict the PSD values for multiple future time steps across all 750 features.

| Task | Why Not Suited |
|------|---------------|
| **Anomaly detection** | Reconstruction-based; flags anomalous timesteps, does not predict future values |
| **Classification** | Single categorical label per window, not a continuous sequence |
| **Imputation** | Fills missing values within a window; no forward-looking prediction |
| **Forecasting** | ✅ Correct: given `T_in` historical timesteps across all channels, predict the next `T_out` timesteps |

---

## 2. Input Format

### 2.1 Raw CSV Format

| Property | Value |
|----------|-------|
| Rows | 6,839 |
| Columns | 750 |
| Header | None |
| Format | Comma-separated, 6 decimal places |
| Values | PSD in dBm (−137.78 to −105.57) |
| Missing | None |

**Column layout:** columns 0–249 = CC1, 250–499 = CC2, 500–749 = LW1. Each row is one minute, chronological.

### 2.2 CSV → Tensor Conversion

```
1. Load:  np.loadtxt(csv)  →  ndarray (6839, 750)

2. Normalize:  revin_only (MOMENT internal RevIN handles this)
   or optionally:  train_zscore (external z-score on training split)

3. Split chronologically first:
   Train:  first 80% of raw time steps
   Val:    next  10%
   Test:   last  10%

4. Window separately per split:
   For each split, slide a window of length T_in + T_out with given stride
   X = window[:T_in]           shape (T_in, C)
   Y = window[T_in:T_in+T_out] shape (T_out, C)
   Transpose each to (C, T_in) and (C, T_out)
```

**Critical: split first, then window.** Generating windows first then splitting leaks information across the boundary.

### 2.3 Expected Tensor Shapes

```
Raw CSV:                   (6839, 750)        — (T, C)
After transpose per window:
  Input:  (B, 750, T_in)   — (B, C, T_in)
  Target: (B, 750, T_out)  — (B, C, T_out)
Optional reshape for analysis:  (B, T_out, 3, 250)
```

MOMENT expects `(batch, channels, sequence_length)`. The dataset should transpose each window from `(T, C)` to `(C, T)` so the DataLoader produces `(B, C, T)`.

---

## 3. Model Architecture

### 3.1 Overview

```
AERPAW CSV
    ↓
Sliding windows  ──►  X: (B, 750, T_in)
                            ↓
              ┌──────────────────────────────┐
              │  MOMENT Backbone             │
              │  (T5 Transformer Encoder)    │
              │  Each channel = independent  │
              │  sequence in batch dim       │
              │  Shared nn.Linear embedder   │
              │     ↓                        │
              │  Output: (B, 750, N, d_model)│
              └──────────────────────────────┘
                            ↓
              ┌──────────────────────────────┐
              │  Forecasting Head            │
              │  Flatten: (B, 750, N*d_model)│
              │  Linear: N*d_model → T_out   │
              │     ↓                        │
              │  out.forecast: (B, 750, T_out)│
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
- **Patching**: Unfold each channel into patches of length 8, stride 8 → N patches per channel
- **PatchEmbedding**: Single shared `nn.Linear(8, d_model)` applied to every patch of every channel
- **T5 Encoder**: Processes each channel independently (channels stacked in batch dim: `[B*C, N, d_model]`)
- **RevIN**: Built-in per-channel reversible instance normalization

### 3.3 Forecasting Head

| Parameter | Value |
|-----------|-------|
| Type | `nn.Linear(N * d_model, T_out)` |
| Input | `(B, C, N * d_model)` — per-channel flattened patches |
| Output | `(B, C, T_out)` — per-channel predictions |
| Initialization | Random (checkpoint head weights are always discarded) |

### 3.4 Prediction Target

```
Raw output:  (B, 750, T_out)      — MOMENT convention
Transpose:   (B, T_out, 750)      — flat format
Reshape:     (B, T_out, 3, 250)   — for per-node per-bin analysis only
```

Output is in normalized space. Denormalize via RevIN (built-in) or external z-score inverse transform.

---

## 4. Output Format

```
out.forecast:   (B, 750, T_out)
→ transpose:    (B, T_out, 750)
→ reshape:      (B, T_out, 3, 250)  — analysis only
```

**Denormalization:**
- If `revin_only`: MOMENT's RevIN denormalizer handles it automatically
- If `train_zscore`: `pred_dbm = pred_normalized * feat_std + feat_mean` (per feature)

**CSV output:** Same layout as input — no header, 750 columns, one row per predicted minute.

**Visualization:** Reshape to `(T_out, 3, 250)`, select node, transpose to `(250, T_out)`, plot as spectrogram.

---

## 5. Training Pipeline

### 5.1 Data Loading

```python
train_dataset = AERPAWDataset(
    csv_path, split="train",
    T_in=128, T_out=16, stride=16,
    normalization="revin_only"
)
# X: (750, 128), Y: (750, 16)
```

The `AERPAWDataset`:
1. Loads CSV as `(6839, 750)`
2. Splits chronologically by time index (no windowing before split)
3. Creates sliding windows within the split
4. Transposes each window to `(C, T)` convention

### 5.2 Normalization

Two options:

**Option A — RevIN only** (default, start here):
- MOMENT's built-in `RevIN` normalizes per sample-channel before the encoder
- No external normalization needed
- Simple, safe baseline

**Option B — External z-score + RevIN** (experimental):
- Compute `mean_f`, `std_f` per feature from training split only
- `X_norm[:, f] = (X[:, f] - mean_f) / std_f`
- Applied to train/val/test before feeding to MOMENT (RevIN still runs internally)
- May help if RevIN alone is insufficient

### 5.3 Train / Validation / Test Split

**Strict chronological split, applied to raw time series before windowing:**

```
Raw time steps: 0 ─────────────────────────────── 6838
                  ├──────── 80% ────────┤──10%──┤─10%─┤
                  Train (0..5470)      Val    Test
```

Generate windows inside each split independently. This ensures no boundary overlap leaks information.

### 5.4 Window Generation

With `T_in=128`, `T_out=16`, `stride=16`:

| Split | Raw timesteps | Windows |
|-------|---------------|---------|
| Train | 0 – 5470 | (5471 − 144) / 16 + 1 ≈ 333 |
| Val | 5471 – 6154 | (684 − 144) / 16 + 1 ≈ 34 |
| Test | 6155 – 6838 | (684 − 144) / 16 + 1 ≈ 34 |

Each window: `X = (128, 750)` → transpose → `(750, 128)`, `Y = (16, 750)` → transpose → `(750, 16)`.

### 5.5 Loss Function

Mean Squared Error (MSE):
```
L = (1 / (B × T_out × 750)) × Σ(predicted − target)²
```

### 5.6 Optimizer

| Parameter | Value |
|-----------|-------|
| Type | Adam |
| Learning rate | 1e-5 (linear probing) |
| Max LR | 1e-4 (OneCycleLR peak, for full fine-tuning) |
| Gradient clip | 5.0 |

### 5.7 Batch Size

Default: **1** (due to 750 channels × memory considerations). Increase to 2–8 after proving the pipeline works.

### 5.8 Epochs

Default: **10** for linear probing (converges quickly). Increase to 20–50 for LoRA or full fine-tuning.

### 5.9 Training Strategy — Three Phases

**Phase 1: Linear probing** (default, safest first experiment)
- Backbone frozen (`freeze_encoder=true`, `freeze_embedder=true`)
- Only forecasting head weights updated
- Fast, minimal overfitting risk
- Establishes whether the pretrained features are useful

**Phase 2: LoRA**
- Freeze most backbone, train low-rank adapters on attention projections + head
- Good middle ground if head-only underfits

**Phase 3: Full fine-tuning**
- All parameters updated
- Highest potential accuracy but risk of catastrophic forgetting
- Only attempt after memory and overfitting are understood

### 5.10 Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| RMSE | √(mean((ŷ − y)²)) | Primary, same unit as dBm |
| MAE | mean(\|ŷ − y\|) | Robust to outliers |

Computed:
- **Overall**: across all dimensions
- **Per horizon**: RMSE at each future time step (t=1, 4, 8, 16)
- **Per node**: RMSE for CC1, CC2, LW1 separately
- **Per frequency bin**: RMSE per bin to identify problematic bands

---

## 6. Assumptions and Design Decisions

### Assumptions

1. **AERPAW has 750 features** (3 nodes × 250 bins). MOMENT supports arbitrary channel counts — each channel is an independent batch element for the T5 encoder.
2. **The CSV has no header**, pure comma-separated dBm values in chronological order.
3. **Normalization statistics** (if using `train_zscore`) are computed per feature across time, from training split only.
4. **The split is chronological**, not random (time-series standard).
5. **TimeRAN checkpoint head weights are always discarded** and the head is reinitialized for our specific `T_in` and `T_out`.
6. **The embedder is fully channel-agnostic.** A single `nn.Linear(8, d_model)` is shared across all channels. No per-channel parameters exist, so the 8 → 750 change causes no loading issues.

### Design Decisions

1. **Foundation model adaptation, not reconstruction** — we reuse the pretrained backbone and only adapt the dataset interface and forecasting head.
2. **750 features as independent channels** — unlike ConvLSTM (which reshapes to 2D spatial-spectral map), MOMENT processes each channel independently. No spatial reshape needed.
3. **Linear probing first** — establishes baseline with minimal compute. LoRA and full fine-tuning are follow-up experiments.
4. **Split before windowing** — prevents data leakage across split boundaries.
5. **Start with `T_in=128`, batch=1** — conservative memory footprint. Scale up after pipeline is verified.
6. **No cross-channel attention** — MOMENT stacks channels in the batch dimension, meaning the transformer never attends across channels. Each channel is predicted independently.

---

## 7. Deviations from Original TimeRAN Setup

### AERPAW Adaptation

| Aspect | Original (TelecomTS) | Our Adaptation | Reason |
|--------|---------------------|----------------|--------|
| Dataset | TelecomTS (8 channels, named KPI columns) | AERPAW (750 channels, unnamed PSD columns) | Different data source |
| Dataset class | `TelecomTS` | `AERPAWDataset` (new) | Different file structure |
| Channels | 8 | 750 | Spectrum vs KPI telemetry |
| seq_len | 512 | 128 | Limited to 6839 timesteps |
| horizon | 208 | 16 | Shorter prediction window |
| stride | 512 (no overlap) | 16 (overlap) | Need more training windows |
| Split | Per-file train_ratio | 3-way chronological on single file | Explicit val split |
| Normalization | None (raw values) | `revin_only` (default) | MOMENT's built-in normalizer |

### Implementation Choices

1. **Linear probing as default**: Original notebooks use `full_finetuning`. We default to `linear_probing` because AERPAW (6839 rows) is much smaller than TelecomTS.
2. **Three-way chronological split**: Original uses single `train_ratio` with remainder as test. We add explicit validation.
3. **Start batch=1**: Memory scales with C × N × d_model (channels × patches × model size). Batch=1 keeps the initial footprint manageable.
4. **`revin_only` normalization first**: Simplest path. Add `train_zscore` only if needed.

### Experimental Options

1. **Checkpoint size**: `small`, `base`, `large` — trade off model capacity vs GPU memory.
2. **Sequence length**: 128 (default), 256, 512 — longer context captures more history but increases memory linearly.
3. **Prediction horizon**: Configurable — longer horizons are harder to predict; start short and extend.
4. **Precision**: `fp32` or mixed precision (`bf16`/`fp16`) — mixed precision reduces memory and may speed up training.
5. **Full fine-tuning**: Higher potential accuracy, risk of overfitting on small dataset.
6. **LoRA**: Parameter-efficient middle ground.
7. **Stride tuning**: Reduce to 1 for max windows, increase to reduce overlap.
8. **External z-score**: Add explicit normalization if RevIN alone is insufficient.

---

## 8. Known Limitations

1. **GPU memory with 750 channels**
   TimeRAN is channel-agnostic architecturally, but memory usage scales with `checkpoint_size`, `input_sequence_length`, `batch_size`, `precision`, and number of channels. Start conservatively (`base`, `T_in=128`, `batch_size=1`) and adjust based on empirical measurements. If memory is insufficient, reduce `T_in`, switch to `small`, enable mixed precision, or use gradient checkpointing.

2. **RevIN may not fully replace dataset-specific normalization**
   While MOMENT's built-in RevIN normalizes per sample-channel, some datasets may benefit from an external z-score computed across the training split. This is configurable via `normalization` and should be tested early.

3. **Window count is limited by dataset size**
   With only 6839 time steps, stride directly controls how many training windows are available. A stride of 16 with `T_in=128`, `T_out=16` yields ~333 training windows. Reducing stride to 1 gives ~6839 windows but with heavy overlap, which risks information leakage between nearby windows if not handled carefully.

4. **LoRA effectiveness is unknown for this task**
   The TimeRAN paper reports that LoRA was not competitive with other fine-tuning regimes for their evaluation. However, our dataset and task differ from TelecomTS, so LoRA may still be worth testing after the linear probing baseline is established.

5. **Checkpoints must be downloaded from Google Drive before training**
   The TimeRAN checkpoint files exceed GitHub's file size limits and are hosted on Google Drive. Run the `gdown` commands in the Quick Start section above to download them into `training/TimeRAN/checkpoints/{small,base,large}/`. Without a checkpoint, the pipeline falls back to raw MOMENT weights (no TimeRAN pretraining).

---

## 9. Implementation Notes

- TimeRAN should be treated as a **pretrained forecasting backbone**, not rebuilt from scratch. The only new code needed is the dataset loader (`AERPAWDataset`) and training/evaluation scripts.
- The default mode is **linear probing**: freeze the backbone and train only the forecasting head. This is the safest first experiment. LoRA and full fine-tuning are follow-up phases.
- `input_sequence_length`, `batch_size`, `checkpoint_size`, and `precision` are configurable because they directly affect **GPU memory usage** — the main practical constraint when running 750 channels through the T5 encoder.
- `normalization` is configurable so that `revin_only` (MOMENT's built-in RevIN) can be compared against `train_zscore` (external z-score from training split only) and `train_zscore_plus_revin` (both). The best strategy depends on the data distribution and should be tested experimentally.
- `stride` is configurable because it controls the tradeoff between **more training windows** (smaller stride, more overlap) and **less overlap / more independence** (larger stride, fewer windows). With only 6839 time steps, this tradeoff matters.
- LoRA is an **optional experimental mode** after the head-only baseline is working. It provides lightweight backbone adaptation without the cost or risk of full fine-tuning.
- TimeRAN's encoder processes channels independently by stacking channels in the effective batch dimension (`[B*C, N, d_model]`). Therefore, cross-channel relationships are not modeled explicitly through transformer attention. Each frequency bin is predicted primarily from its own history, which is appropriate for spectrum data where bins are not spatially correlated in the same way as images.

---

## References

1. **TimeRAN paper:** I. Panitsas, L. Tassiulas, "A Family of Open Time-Series Foundation Models for the Radio Access Network," arXiv:2604.04271, 2026.
2. **TimeRAN repository (original upstream):** https://github.com/panitsasi/TimeRAN
3. **MOMENT:** MOMENT: A Family of Open Time-series Foundation Models (ICML 2024). https://github.com/moment-timeseries-foundation-model/moment
4. **AERPAW dataset:** AERPAW sub-6 GHz spectrum monitoring dataset — Fixed nodes CC1, CC2, LW1 (Feb 2022). DOI: 10.5061/dryad.hmgqnk9zn.
