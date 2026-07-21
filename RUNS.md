# Training Runs

Every training run produces a self-contained directory under `runs/`.

## Naming convention

```
runs/<model_name>_<YYYYMMDD_HHMMSS>/
```

The timestamp is the training start time in UTC. The experiment name is
generated automatically by the training script (`train_integrated.py`)
and can be overridden with `--name`.

## Run directory contents

Each run directory contains everything needed to reproduce and understand
the results of a training run:

```
runs/vanillalstm_20260720_072637/
  config.yaml                                # copy of the config used
  <chunk_id>_training_log.csv                # epoch-by-epoch metrics
  <chunk_id>_training_summary.txt            # human-readable summary
  aggregate_metrics.csv                      # evaluation: one row per model/chunk/split/horizon
  per_frequency_metrics.csv                  # evaluation: one row per frequency
  per_band_metrics.csv                       # evaluation: one row per band
  report.txt                                 # evaluation summary
  forecasts/                                 # prediction arrays
    <chunk_id>_<model>_predictions.npz
    <chunk_id>_<model>_targets.npz
    <chunk_id>_<model>_metadata.json
  checkpoints/                               # model weights
    <chunk_id>_<model>.pt
  *.png                                      # plots
```

## Multi-model experiments

When a single config defines multiple models (e.g. `safe_comparison.yaml`),
each model gets a subdirectory inside the run:

```
runs/safe_comparison_20260721_194001/
  config.yaml
  VanillaLSTM/
    checkpoints/
    ...
  LinearAutoRegressive/
    checkpoints/
    ...
  LookbackMean/
    ...
```

## Evaluation only

When running `evaluation_integrated.py` against an existing run, pass the
run directory with `--name` (must match the training run name) so outputs
land in the same place.

## Git

The `runs/` directory is gitignored. Run artifacts are local-only.
