# TimeRAN Adaptation for AERPAW Spectrum Prediction

## 1. What TimeRAN Already Provides

TimeRAN is a family of open time-series foundation models for the Radio Access Network (RAN), built on top of **MOMENT** (`MOMENTPipeline` from `momentfm`). The repository at `git@github.com:Demii-7/TimeRAN.git` provides:

### Pretrained Backbone Checkpoints
| Variant | Hugging Face Model | Checkpoint File |
|---------|--------------------|-----------------|
| small   | `AutonLab/MOMENT-1-small` | `TimeRAN_small.pth` |
| base    | `AutonLab/MOMENT-1-base`  | `TimeRAN_base.pth`  |
| large   | `AutonLab/MOMENT-1-large` | `TimeRAN_large.pth`  |

The checkpoints are fine-tuned versions of MOMENT backbone weights (encoder + embedder) trained on the TimeRAN DataPile. Task heads are **not** included in the checkpoint — they are stripped and reinitialized at load time.

### Downstream Task Heads (via MOMENTPipeline)
| Task | `task_name` | Output Attribute | Description |
|------|-------------|------------------|-------------|
| Forecasting | `"forecasting"` | `out.forecast` | Linear projection to `forecast_horizon` |
| Anomaly Detection | `"reconstruction"` | `out.reconstruction` | Reconstruction-based scoring |
| Imputation | `"reconstruction"` | `out.reconstruction` | Masked reconstruction |
| Classification | `"classification"` | `out.logits` | Linear head to `num_class` |

### Training Modes
- **`full_finetuning`** — updates all parameters
- **`linear_probing`** — freezes backbone, trains only task head
- **`lora: true`** — applies LoRA to encoder (PEFT)

### Task Folders
```
TimeRAN_Foundation_Model/
  forecasting/          forecasting.ipynb + config.yaml
  anomaly_detection/    zero_shot_anomaly_detection.ipynb + config.yaml
  classification/       classification.ipynb + config.yaml
  imputation/           zero_shot_imputation.ipynb + config.yaml
  data/
    interfaces/
      TelecomTS.py      Dataset class for all tasks
    checkpoints/        Local .pth checkpoint storage
    datasets/           TelecomTS CSV data organized by task
```

## 2. Why Spectrum Prediction = Forecasting

| Task | Why NOT suited |
|------|---------------|
| **Anomaly Detection** | Reconstruction-based; detects anomalous timesteps, does not predict future values. |
| **Classification** | Produces a single categorical label per window, not a continuous multi-step multi-variable sequence. |
| **Imputation** | Fills missing values within a window; no forward-looking prediction. |
| **Forecasting** | **Exactly** what we need: given `T_in` historical timesteps across all 750 channels, predict the next `T_out` timesteps. |

Spectrum prediction is a **multivariate multi-step forecasting** problem — the forecasting task is the correct interface.

## 3. Adaptation Pipeline

```
merged_power_data_sub6GHz_avg_per_minute.csv   (6839, 750)
                       |
                       ↓
            Load as multivariate time series
                       |
                       ↓
                Normalize (fit on training split only)
                       |
                       ↓
                Create sliding windows
            X: (num_windows, T_in, 750)
            Y: (num_windows, T_out, 750)
                       |
                       ↓
            Load pretrained TimeRAN checkpoint
            (MOMENTPipeline + TimeRAN .pth weights)
                       |
                       ↓
            Attach forecasting head
            (task_name="forecasting", forecast_horizon=T_out)
                       |
                       ↓
            Train head first (freeze backbone = linear_probing mode)
                       |
                       ↓
            Optionally fine-tune backbone (full_finetuning or LoRA)
                       |
                       ↓
            Evaluate: RMSE / MAE per horizon, node, frequency bin
```

## 4. Expected Input/Output Shapes

| Stage | Shape | Notes |
|-------|-------|-------|
| Raw CSV | `(6839, 750)` | 6839 timesteps × 750 features |
| Input window (X) | `(B, T_in, 750)` | MOMENT expects `(B, C, T)` — may need transpose to `(B, 750, T_in)` |
| Target window (Y) | `(B, T_out, 750)` | Forecasting head produces `(B, 750, T_out)` |
| Analysis reshape | `(B, T_out, 3, 250)` | 3 nodes × 250 bins; for per-node/per-bin metrics only |

**Note:** MOMENT internally treats input as `(batch, channels, sequence_length)`. The `TelecomTS` dataset transposes CSV rows from `(T, C)` to `(C, T)`. Our dataset must follow the same convention.

## 5. Reconstruction Strategy

This is **not** a full model rebuild like ConvLSTM. We are reusing the pretrained TimeRAN foundation model as-is.

What needs adaptation:
- **Dataset interface** — replace `TelecomTS` with an `AERPawDataset` that loads our CSV
- **Forecasting head dimension** — confirm the existing head handles 750 channels (MOMENT processes each channel independently, so this should work natively)
- **Metrics** — RMSE/MAE computed per horizon step, per node, per frequency bin

What stays unchanged:
- Backbone architecture (MOMENT transformer encoder)
- Checkpoint loading mechanism
- Training loop structure (from `forecasting.ipynb`)
- Hyperparameter config structure

## 6. Open Questions (to be resolved during implementation)

1. **Input shape**: Does `MOMENTPipeline` with `task_name="forecasting"` accept `(B, 750, T_in)` or does it expect an extra dimension? (`forecasting.ipynb` adds `unsqueeze(1)` when `ndim == 2`, suggesting `(B, C, T)` where C=1 for univariate — need to test with C=750.)

2. **Forecasting head**: Is the linear head `head.linear` mapping from `d_model` to `forecast_horizon` per channel, or globally? Does the head output shape match `(B, C, H)` automatically?

3. **750-channel support**: Does MOMENT's embedder handle 750 input channels natively, or does it have a maximum channel limit? The `n_channels` parameter in classification config suggests channel count is configurable.

4. **Checkpoint loading**: Does `load_state_dict(strict=False)` silently ignore mismatched embedder weights if channel counts differ? The TimeRAN checkpoint was trained on 8-channel TelecomTS data — loading into a 750-channel model may cause shape mismatch on the embedder.

5. **Freezing backbone**: Does `freeze_encoder=True` + `freeze_embedder=True` in `model_kwargs` cleanly freeze all non-head parameters? Verified from the code: yes, but only if the embedder shape matches (otherwise loading fails before freezing).

6. **Input normalization**: Does the TelecomTS dataset apply any normalization? Reading the code — **no**, it loads raw CSV values. MOMENT may expect standardized inputs; this needs empirical testing.

7. **Dataset interface**: The existing `TelecomTS.__init__` expects a specific data directory structure. We will need a new dataset class (or a flag) that loads a single CSV and wraps it with sliding windows.

8. **Stride and overlap**: With 6839 timesteps, what stride yields sufficient training samples? If `stride < seq_len`, windows overlap — acceptable but must avoid data leakage between train/val/test splits.

## 7. Proposed Files for Future Implementation

```
training/TimeRAN/
  README.md             # This file
  config.yaml           # Configuration
  dataset.py            # AERPawDataset — loads CSV, normalizes, creates windows
  train_head.py         # Training script (linear probing + optional full fine-tune)
  evaluate.py           # Evaluation script (RMSE/MAE per horizon/node/bin)
  inference.py          # Inference script for deployment
  utils.py              # Normalization, metrics, plotting helpers
```

## 8. Quick Start (Proposed — scripts do not exist yet)

```bash
# 1. Install dependencies
pip install momentfm torch pandas numpy pyyaml

# 2. Download TimeRAN checkpoint
#    (place in a local checkpoints/ directory)

# 3. Edit config.yaml with your paths and hyperparameters

# 4. Train forecasting head (backbone frozen)
python train_head.py --config config.yaml

# 5. Evaluate on test split
python evaluate.py --config config.yaml

# 6. Run inference on new data
python inference.py --input new_spectrum.csv --checkpoint runs/best_model.pth
```

## 9. ⚠️ Data Leakage Warning

- **Do not** normalize using global statistics. Compute mean/std **only on the training split** and apply the same transform to validation and test splits.
- **Do not** allow validation or test windows to overlap with training windows. Use strict temporal split (first N% for train, next M% for val, last K% for test).
- **Do not** shuffle across time when splitting — preserve temporal order.
- **Evaluation** must be done autoregressively or using the same forecasting setup intended for inference (matching context length and horizon).
- **Stride** must be chosen so that train/val/test windows come from disjoint time regions.
