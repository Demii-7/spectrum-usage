# Integrated Pipeline

This branch keeps a shared chunk-based training and evaluation pipeline under `training/common/` while allowing each model directory to keep its latest standalone implementation.

## Purpose

- `main` carries the latest model implementations.
- `integrate` carries the shared AERPAW chunk pipeline.
- Integrated runners live in each model directory as `train_integrated.py`.

## Shared Config

- Main config: `training/common/config.yaml`
- Smoke config: `training/common/config.smoke.yaml`
- Shared keys:
  - `data`: AERPAW source path, chunk list, optional `max_rows`, optional `test_rows`
  - `windowing`: shared evaluation horizons
  - `preprocessing`: shared normalization toggle for the common numeric pipeline
  - model-specific sections for `linear_autoregressive`, `convlstm`, `stsprednet`, `timeran`, `tss_lcd`

## Chunk Loading

- Shared loader: `training.common.data.load_chunk`
- Source files: `evaluation/aerpaw/Results<Site>Feb2022_SigMF_power_1mhz_avg_per_minute.csv`
- Default reference site: `CC2`
- Default chunks:
  - `chunk_600_800`
  - `chunk_2400_2600`
  - `chunk_3500_3700`

The loader keeps a chronological train/test split. For smoke tests, `data.max_rows` and `data.test_rows` can reduce the split size without changing runner code.

## Horizons

- Shared evaluation horizons come from `windowing.horizons`.
- All integrated models on this branch score directly on denormalized dBm predictions.

## Train One Model

Use the integrated runner in the model directory:

```bash
./.venv/bin/python training/ConvLSTM/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/STS-PredNet/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/TimeRAN/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/TSS-LCD/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/LinearAutoRegressive/train.py --config training/common/config.yaml
```

## Train All Models

Run the commands above in sequence, or script them in the validation order used for this branch:

1. `LinearAutoRegressive`
2. `ConvLSTM`
3. `STS-PredNet`
4. `TimeRAN`
5. `TSS-LCD`

## Results Layout

Each integrated runner writes to `training/results/<ModelName>/`.

Expected outputs:

- `aggregate_metrics.csv`
- `per_band_metrics.csv`
- `per_frequency_metrics.csv`
- `chunk_<chunk_id>_training_log.csv`
- `report.txt`
- `checkpoints/`

## Assemble Results

```bash
./.venv/bin/python training/common/assemble_results.py --config training/common/config.yaml
```

Default aggregation includes the integrated branch models plus baselines.

## Smoke Tests

Use the shared smoke config:

```bash
./.venv/bin/python training/LinearAutoRegressive/train.py --config training/common/config.smoke.yaml
./.venv/bin/python training/ConvLSTM/train_integrated.py --config training/common/config.smoke.yaml
./.venv/bin/python training/STS-PredNet/train_integrated.py --config training/common/config.smoke.yaml
./.venv/bin/python training/TimeRAN/train_integrated.py --config training/common/config.smoke.yaml
./.venv/bin/python training/TSS-LCD/train_integrated.py --config training/common/config.smoke.yaml
```

## Model Notes

- `LinearAutoRegressive`: preserved from `integrate`.
- `ConvLSTM`, `STS-PredNet`, `TimeRAN`, `TSS-LCD`: existing integrated runners on this branch, updated to share output directory, checkpoint, and reporting helpers.
- `data.max_rows` and `data.test_rows` are available for smoke runs without editing runner code.
