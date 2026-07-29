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
  - `data`: representation (`1d`, `2d`, or `4d`), source `files`, frequency selection, split settings, and chunk list
  - `windowing`: shared evaluation horizons
  - `preprocessing`: shared normalization toggle for the common numeric pipeline
  - model-specific sections such as `convlstm`, `autoformer_csa`, `deepspred`

## Data Loading

- Shared routing entry point: `training.common.data.load_chunk`
- Single shared loader: `training.common.data`; spatial map construction is handled internally by `training.common.map_builder`.
- `data.files` is the only source-file setting; each entry has `path` and `partition: train|test`.
- Validation is carved from the train partition; the test partition is used in full.
- 4D loading can use a named cached map or build it from CSV files under `data/maps`.
- Loaded splits carry sequence segments so windows cannot cross files, sites, or frequency-bin segments.
- Set `data.map.permute: true` to randomly reassign collection-point coordinates; use `data.map.permute_seed` for reproducible permutations.
- For 2D/4D inputs, `data.mask.frequency_ranges` preserves selected ranges and replaces all other channels with `data.mask.noise_floor`.
- `preprocessing.max_missing_gap` bounds interpolation and forward/back filling.

## Horizons

- Shared evaluation horizons come from `windowing.horizons`.
- Numeric models score directly on denormalized dBm predictions.
- DeepSPred maps minute horizons into rows inside predicted spectrogram frames.

## Train One Model

Use the integrated runner in the model directory:

```bash
./.venv/bin/python training/ConvLSTM/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/STS-PredNet/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/common/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/TSS-LCD/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/VanillaLSTM/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/Autoformer-CSA/train_integrated.py --config training/common/config.yaml
./.venv/bin/python -m training.common.train_integrated --config <dswinlstm-4d-config.yaml>
./.venv/bin/python training/DeepSPred/train_integrated.py --config training/common/config.yaml
./.venv/bin/python training/LinearAutoRegressive/train.py --config training/common/config.yaml
```

## Train All Models

Run the commands above in sequence, or script them in the validation order used for this branch:

1. `LinearAutoRegressive`
2. `ConvLSTM`
3. `STS-PredNet`
4. `TimeRAN`
5. `TSS-LCD`
6. `VanillaLSTM`
7. `Autoformer-CSA`
8. `DSwinLSTM-I`
9. `DeepSPred`

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

Default aggregation now includes all integrated models plus baselines.

## Smoke Tests

Use the shared smoke config:

```bash
./.venv/bin/python training/LinearAutoRegressive/train.py --config training/common/config.smoke.yaml
./.venv/bin/python training/ConvLSTM/train_integrated.py --config training/common/config.smoke.yaml
./.venv/bin/python training/STS-PredNet/train_integrated.py --config training/common/config.smoke.yaml
./.venv/bin/python training/common/train_integrated.py --config training/common/config.smoke.yaml
./.venv/bin/python training/TSS-LCD/train_integrated.py --config training/common/config.smoke.yaml
./.venv/bin/python training/VanillaLSTM/train_integrated.py --config training/common/config.smoke.yaml
./.venv/bin/python training/Autoformer-CSA/train_integrated.py --config training/common/config.smoke.yaml
./.venv/bin/python training/DeepSPred/train_integrated.py --config training/common/config.smoke.yaml
```

## Model Notes

- `LinearAutoRegressive`: preserved from `integrate`.
- `ConvLSTM`, `STS-PredNet`, `TimeRAN`, `TSS-LCD`: integrated runners preserved from `integrate`, model sources updated from `main`.
- `VanillaLSTM`, `Autoformer-CSA`, and `DeepSPred`: restored from `main` and wrapped with integrated runners.
- `DSwinLSTM-I`: uses the canonical model and shared 4D training/evaluation pipeline.
- `DeepSPred`: first integration pass uses CSV chunk data converted into colormap spectrogram frames.
- Interpolated-map support remains model-specific and optional.
