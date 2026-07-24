## Manual Training Plan

This guide is for running the model folders the way an outside user would: by copying and editing each model's own `config.yaml` and following that model's README.

## First Decision

Do not use one config strategy for all three claims.

- Use **standalone model configs** for `VanillaLSTM`, `Autoformer-CSA`, `ConvLSTM`, `STS-PredNet`, `TimeRAN`, and `DeepSPred`.
- Keep a **separate YAML per claim, per model, per band**.
- Keep source and target bands matched. Do not train on `600_800` and test on `2400_2600` unless that is the claim you want to make.

Recommended naming:

- `training/VanillaLSTM/config.powder.temporal.guesthouse.600_800.yaml`
- `training/Autoformer-CSA/config.powder.temporal.2site.600_800.yaml`
- `training/ConvLSTM/config.powder.spatial.map.600_800.yaml`
- `training/TimeRAN/config.transfer.aerpaw_to_powder.600_800.yaml`

## Current Data Reality

What already exists in the repo:

- POWDER single-site CSVs with header + timestamp column in `data/`
- POWDER combined interpolated maps in `.npz` form for:
  - `data/powder_20260618T0036Z_humanities_guesthouse_600_800.npz`
  - `data/powder_temporal_test_split_humanities_guesthouse_600_800.npz`
  - `data/powder_20260618T0036Z_humanities_guesthouse_2400_2600.npz`
  - `data/powder_temporal_test_split_humanities_guesthouse_2400_2600.npz`
- AERPAW CSVs in `evaluation/aerpaw/`

What does **not** currently exist:

- ready-to-train POWDER numeric CSVs with the timestamp column removed
- ready-to-train merged 2-site POWDER sequence CSVs for `STS-PredNet` / `Autoformer-CSA`
- separate 6/28 POWDER map `.npz` files
- ARA / COSMOS training-ready files in `data/`

## Hard Constraint

Most standalone CSV loaders are **not** compatible with the current POWDER CSV files as-is.

These model folders expect numeric-only CSVs with no timestamp text column:

- `training/Autoformer-CSA/`
- `training/ConvLSTM/` in CSV mode
- `training/STS-PredNet/` in CSV mode
- `training/TimeRAN/`
- `training/DeepSPred/`
- `training/VanillaLSTM/` also expects numeric-only data even though it has `has_header`

So before config edits matter, you need preprocessed CSVs like:

- single-site numeric: shape `(T, 200)`
- merged 2-site numeric: shape `(T, 400)` ordered as `[humanities_200_bins, guesthouse_200_bins]`

## Claim 1: Temporal Pattern Learning

Claim:

- train on POWDER 6/18 run: humanities + guesthouse
- test on POWDER 6/28 run or the temporal-split files

Models:

- `STS-PredNet`
- `Autoformer-CSA`
- `VanillaLSTM` as LSTM baseline

### VanillaLSTM

Best use here:

- train **one model per site per band**
- do not try to pack two POWDER sites into this model

Config changes from `training/VanillaLSTM/config.yaml`:

```yaml
data:
  dataset_path: data/powder_20260618T0036Z_guesthouse_600_800_numeric.csv
  site_name: guesthouse
  n_frequency_bins: 200
  has_header: false

windowing:
  input_sequence_length: 60
  prediction_horizon: 60
  train_stride: 1
  val_stride: 60
  test_stride: 60

split:
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0

evaluation:
  eval_horizons: [1, 5, 15, 60]
```

Notes:

- Use one config for `guesthouse`, another for `humanities`.
- For the cross-week test, point the evaluation step at the 6/28 numeric CSV, not the training CSV.

### Autoformer-CSA

Best use here:

- merged 2-site sequence model
- train on a numeric `(T, 400)` CSV built from humanities + guesthouse on the same timestamps

Use the shared configuration and run `training/Autoformer-CSA/train_integrated.py`.
The model-specific settings live under `autoformer_csa` in
`training/common/config.yaml`.

```yaml
data:
  representation: 2d
  files:
    - path: data/powder_20260618T0036Z_humanities_guesthouse_600_800_numeric.csv
      partition: train
    - path: data/powder_20260628T0436Z_humanities_guesthouse_600_800_numeric.csv
      partition: test
  concat: rows
  reference_site: humanities_guesthouse
  chunks:
    - id: powder_600_800
      start_mhz: 600.0
      end_mhz: 800.0

windowing:
  lookback: 96
  horizons: [1, 5, 15, 60]

preprocessing:
  normalize: true
  impute: true
  max_missing_gap: 5

autoformer_csa:
  seq_len: 96
  label_len: 48
  pred_len: 60
  batch_size: 32
  epochs: 20
  learning_rate: 0.0001
  optimizer: adam
  loss: rmse
  model:
    encoder_layers: 2
    decoder_layers: 1
    n_heads: 8
```

### STS-PredNet

Best use here:

- merged 2-site sequence CSV for temporal learning
- or map mode for spatial learning, not both in the same config

Config changes from `training/STS-PredNet/config.yaml`:

```yaml
data:
  format: csv
  dataset_path: data/powder_20260618T0036Z_humanities_guesthouse_600_800_numeric.csv
  n_nodes: 2
  bins_per_node: 200
  n_features: 400
  node_names: [humanities, guesthouse]

model:
  input_channels: 1
  map_height: 2
  map_width: 200

branches:
  use_closeness: true
  use_period: true
  use_trend: false
  lc: 36
  lp: 3
  period_interval: 1440

evaluation:
  eval_horizons: [1, 5, 15, 60]
```

Important limitation:

- the standalone 6/28 POWDER run is only about `4155` rows for `600_800`
- `lp: 3` and `period_interval: 1440` need `4320` rows of history
- that means **6/28 alone is not enough** for the default daily-period branch

If you want to keep daily periodicity for STS-PredNet, use the combined temporal-split dataset instead of the bare 6/28 file.

If you insist on evaluating on the 6/28 file alone, reduce one of:

- `branches.lp`
- `branches.period_interval`
- or disable `use_period`

## Claim 2: Spatial Pattern Learning

Claim:

- train on POWDER 6/18 map/sequence data with variable sensors
- test on POWDER 6/28 map/sequence data
- evaluate known nodes and held-out/unseen nodes

Models:

- `ConvLSTM`
- `STS-PredNet`
- `VanillaLSTM` baseline

### ConvLSTM

Use **interpolated map mode**.

Config changes from `training/ConvLSTM/config.yaml`:

```yaml
data:
  format: interpolated_map
  map_path: data/powder_20260618T0036Z_humanities_guesthouse_600_800.npz
  map_key: map_db
  n_freq_bins: 200
  grid_height: 10
  grid_width: 10

windowing:
  input_sequence_length: 60
  prediction_horizon: 60
  stride: 1

split:
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0
  chronological_split: true

model:
  input_channels: 200
  use_channel_projection: true
  channel_projection_dim: 16

evaluation:
  eval_horizons: [1, 5, 15, 60]
```

### STS-PredNet

Use **interpolated map mode** here too.

Config changes from `training/STS-PredNet/config.yaml`:

```yaml
data:
  format: interpolated_map
  map_path: data/powder_20260618T0036Z_humanities_guesthouse_600_800.npz
  map_key: map_db
  temporal_overrides:
    lc: 6
    lp: 2
    period_interval: 6

model:
  # these are auto-overridden in map mode, but keeping them explicit helps readability
  input_channels: 200
  map_height: 10
  map_width: 10

branches:
  use_closeness: true
  use_period: true
  use_trend: false
```

Why the temporal override matters:

- map files are shorter than long AERPAW histories
- the AERPAW-style daily branch settings are usually too large for map-mode experiments

### VanillaLSTM baseline

This is only a **non-spatial baseline**.

Recommended use:

- run one model per site per band on the single-site POWDER numeric CSVs
- compare against the spatial models to show what is lost when spatial structure is ignored

### Important spatial gap

The repo currently does **not** have a separate 6/28 POWDER map file.

You currently have:

- `data/powder_20260618T0036Z_humanities_guesthouse_600_800.npz`
- `data/powder_temporal_test_split_humanities_guesthouse_600_800.npz`

So for a clean train-on-6/18 / test-on-6/28 spatial claim, config edits alone are not enough. You need one of these:

1. generate a standalone 6/28 map `.npz`
2. train/evaluate from the combined temporal-split `.npz` using exact chronological boundaries
3. add explicit train-path / test-path support to the map-mode scripts

Also, held-out or unseen-node evaluation is **not** controlled by config alone right now. That requires the data product itself to exclude one node at map construction time or a custom evaluation routine at node coordinates.

## Claim 3: Transferability

Claim:

- train on AERPAW
- test or fine-tune on POWDER, ARA, COSMOS

Models:

- `DeepSPred`
- `TimeRAN`

### DeepSPred

Source training on AERPAW can stay close to the existing config.

For target-testbed evaluation, the target CSV should be a **numeric single-site CSV** with no timestamp column.

For POWDER `600_800`, config changes from `training/DeepSPred/config.yaml` would be:

```yaml
data:
  dataset_path: data/powder_20260628T0436Z_guesthouse_600_800_numeric.csv
  nodes:
    guesthouse:
      col_start: 0
      col_end: 200
  bins_per_node: 200

frames:
  minutes_per_frame: 256
  w_pad: 256
  w_orig: 200
```

Important interpretation detail:

- DeepSPred pools nodes as independent samples
- that makes single-site target transfer feasible
- but your source and target frame geometry must still be compatible with the checkpoint you use

### TimeRAN

TimeRAN is the easiest transfer candidate.

For POWDER `600_800`, config changes from `training/TimeRAN/config.yaml`:

```yaml
data:
  dataset_path: data/powder_20260628T0436Z_guesthouse_600_800_numeric.csv
  n_features: 200
  n_nodes: 1
  bins_per_node: 200
  node_names: [guesthouse]

windowing:
  input_sequence_length: 128
  prediction_horizon: 16
  stride: 16

evaluation:
  eval_horizons: [1, 4, 8, 16]
```

Notes:

- keep the target band identical to the source band
- keep `prediction_horizon` the same between training and evaluation because the forecasting head is horizon-specific
- for first transfer runs, keep `model.freeze_backbone: true`

### ARA / COSMOS

At the moment there are no ARA or COSMOS training-ready files in `data/`, so config planning is conceptual until those files exist.

When they do exist, follow the same pattern as POWDER:

- single-site numeric CSV for `TimeRAN`
- single-site numeric CSV for `DeepSPred`
- same band and feature width as the source training setup

## What You Should Change vs What You Should Not Change

Change these by claim:

- `data.dataset_path`
- `data.format`
- `data.map_path`
- `data.n_features`
- `data.n_nodes`
- `data.bins_per_node`
- `data.node_names`
- `model.enc_in`, `model.dec_in`, `model.c_out` for Autoformer
- `model.map_height`, `model.map_width`, `model.input_channels` where spatial shape changes
- `windowing.*`
- `split.*`
- `evaluation.eval_horizons`

Do not casually change these unless there is a reason:

- optimizer family
- learning-rate schedules
- hidden dimensions
- transformer depth / head counts
- diffusion depth / patch depths

## Recommended Order

1. Build POWDER numeric CSVs first.
2. Run temporal claim first with `VanillaLSTM` on one site.
3. Then run `Autoformer-CSA` on merged 2-site CSV.
4. Then run `STS-PredNet` temporal, using the combined temporal-split file if you want the daily period branch.
5. For spatial claim, use map mode only after producing or confirming the exact train/test map assets.
6. For transfer claim, start with `TimeRAN`, then `DeepSPred`.

## Bottom Line

Config changes are enough only after the data is in the format each standalone model expects.

Right now:

- **temporal claim**: mostly blocked by missing numeric no-timestamp POWDER CSVs
- **spatial claim**: additionally blocked by missing standalone 6/28 map files and unseen-node evaluation assets
- **transfer claim**: blocked for POWDER by missing numeric target CSVs, and blocked for ARA/COSMOS by missing target data files
