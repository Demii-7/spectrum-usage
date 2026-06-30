# VanillaLSTM Spectrum Prediction

> **Baseline type:** simple single-site recurrent forecaster
>
> **Target task:** predict future spectrum values for one site, such as `CC2`, from that site's own historical measurements only

---

## Quick Start

```bash
python3 training/VanillaLSTM/train.py --config training/VanillaLSTM/config.yaml
python3 training/VanillaLSTM/evaluate.py --checkpoint training/VanillaLSTM/checkpoints/best_model.pt
```

---

## 1. What the Model Is Intended to Do

`VanillaLSTM` is a direct multi-step forecasting baseline for single-site spectrum prediction.

Given a historical sequence of per-timestep spectrum measurements from one site:

```text
Input history:  (T_in, F)
```

the model predicts the next `T_out` future timesteps across the same `F` frequency bins:

```text
Forecast:       (T_out, F)
```

This is intended as a lightweight baseline against more complex spectrum predictors.

---

## 2. Input Format

### Raw CSV

- Path: configurable via `data.dataset_path`
- Expected shape: `(T, F)`
- Default site metadata: `CC2`
- Default frequency bins: `250`
- Default smoke-test path: `training/data/cc2_smoke_test.csv`

Each row is one timestamp. Each column is one frequency bin for a single site only.

### Windowed Tensors

```text
X: (B, T_in, F)
Y: (B, T_out, F)
```

Chronological raw splits are created before windowing, then each split is windowed with its own stride.

---

## 3. Architecture

### Tensor Flow

```text
Input:  (B, T_in, F)
   ↓
LSTM over time
   input_size  = F
   hidden_size = configurable
   num_layers  = configurable
   dropout     = configurable
   batch_first = true
   ↓
Output head
   final_hidden: Linear(hidden -> T_out * F)
   or
   all_hidden:   Linear(T_in * hidden -> T_out * F)
   ↓
Reshape
   (B, T_out, F)
```

### Default Output Strategy

Default: `final_hidden`

The final hidden state of the last LSTM layer is projected directly into the full forecast horizon.

---

## 4. Output Format

Model output:

```text
(B, T_out, F)
```

Evaluation exports denormalized dBm predictions as:

- `metrics.json`
- `predictions.csv`
- `ground_truth.csv`
- `spectrogram_CC2.png`
- `error_analysis.png`

`predictions.csv` and `ground_truth.csv` flatten `(N_windows, T_out, F)` into `(N_windows * T_out, F)`.

---

## 5. Training Pipeline

1. Load one single-site CSV with shape `(T, F)`.
2. Split the raw time series chronologically into train, val, and test.
3. Fit z-score normalization on the train split only.
4. Apply the same normalization statistics to val and test.
5. Create sliding windows:
   `X = (T_in, F)`, `Y = (T_out, F)`
6. Train with MSE loss in normalized space.
7. Save:
   - `best_model.pt`
   - `last_model.pt`
   - `normalization_stats.pt`
   - `training_log.json`
8. Evaluate the best checkpoint in denormalized dBm space.

---

## 6. Assumptions And Design Decisions

- Single-site only. The loader expects one site's spectrum bins in the CSV.
- Frequency bins are treated as feature dimensions, not spatial dimensions.
- The baseline uses direct multi-step prediction instead of autoregressive rollout.
- Default LSTM depth is `1`, but `model.num_layers` is configurable.
- Normalization is fit on train only to avoid temporal leakage.
- Metrics and plots are reported in denormalized dBm, not normalized space.

---

## 7. Known Limitations

- No cross-site information is used.
- No explicit modeling of local frequency structure beyond the dense projection head.
- The default head predicts the full horizon at once, so forecast errors are not conditioned on earlier predicted steps.
- `all_hidden` increases head size because it flattens the full encoded sequence.
- This baseline assumes the CSV is already aligned and cleaned enough to train directly.

---

## 8. Configuration Reference

### `data`

| Field | Default | Description |
|-------|---------|-------------|
| `dataset_path` | `training/data/cc2_smoke_test.csv` | Single-site CSV path |
| `site_name` | `CC2` | Site label used in outputs |
| `n_frequency_bins` | `250` | Number of columns expected in the CSV |
| `max_rows` | `null` | Optional row cap for quick experiments |
| `has_header` | `false` | Whether the CSV has a header row |

### `windowing`

| Field | Default | Description |
|-------|---------|-------------|
| `input_sequence_length` | `12` | Historical timesteps per input window |
| `prediction_horizon` | `6` | Future timesteps per target window |
| `train_stride` | `1` | Training window stride |
| `val_stride` | `6` | Validation window stride |
| `test_stride` | `6` | Test window stride |

### `split`

| Field | Default | Description |
|-------|---------|-------------|
| `train_ratio` | `0.8` | Raw chronological train ratio |
| `val_ratio` | `0.1` | Raw chronological val ratio |
| `test_ratio` | `0.1` | Raw chronological test ratio |
| `chronological_split` | `true` | Split before windowing |

### `preprocessing`

| Field | Default | Description |
|-------|---------|-------------|
| `normalization` | `zscore` | Per-frequency z-score normalization |
| `fit_on_train_only` | `true` | Prevent leakage into val/test |

### `model`

| Field | Default | Description |
|-------|---------|-------------|
| `input_size` | `250` | Must match `n_frequency_bins` |
| `hidden_size` | `128` | LSTM hidden size |
| `num_layers` | `1` | Number of stacked LSTM layers |
| `dropout` | `0.1` | LSTM dropout when `num_layers > 1` |
| `output_strategy` | `final_hidden` | `final_hidden` or `all_hidden` |
| `bidirectional` | `false` | Optional bidirectional LSTM |

### `training`

| Field | Default | Description |
|-------|---------|-------------|
| `batch_size` | `32` | Batch size |
| `epochs` | `100` | Maximum epochs |
| `learning_rate` | `0.001` | Adam learning rate |
| `optimizer` | `adam` | Optimizer |
| `loss` | `mse` | Training loss |
| `early_stopping` | `true` | Stop on stalled val loss |
| `patience` | `15` | Early-stopping patience |
| `gradient_clip` | `1.0` | Gradient clipping norm |
| `seed` | `42` | Random seed |

### `evaluation`

| Field | Default | Description |
|-------|---------|-------------|
| `metrics` | `['rmse', 'mae', 'r2']` | Reported metrics |
| `eval_horizons` | `[1, 3, 6]` | Key horizons to inspect |
| `export_predictions` | `true` | Write prediction CSVs |
| `plot_denormalized_dbm` | `true` | Plot dBm-space spectrograms |

---

## 9. Smoke Test Report Format

For the smoke test, report:

1. Files created
2. Command run
3. Train, val, and test window counts
4. Parameter count
5. Final train and val loss
6. Test RMSE, MAE, and R2
7. Per-horizon RMSE
8. Whether CSV exports and plots were created
9. Any issues encountered

Recommended smoke-test command shape:

```bash
python3 training/VanillaLSTM/train.py \
  --config training/VanillaLSTM/config.yaml \
  --csv training/data/cc2_smoke_test.csv \
  --epochs 5

python3 training/VanillaLSTM/evaluate.py \
  --checkpoint training/VanillaLSTM/checkpoints/best_model.pt \
  --config training/VanillaLSTM/config.yaml \
  --csv training/data/cc2_smoke_test.csv
```
