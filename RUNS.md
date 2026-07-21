# Training Runs

The shared training script writes each experiment under `runs/` unless you pass
`--output-dir`.

## Run names

Single-model runs use this default name:

```text
runs/<model_name>_<YYYYMMDD_HHMMSS>/
```

Multi-model runs use `comparison_<YYYYMMDD_HHMMSS>`. You can set the base name
with `--name` or set the full path with `--output-dir`.

Training does not reuse an existing directory. If `runs/example` exists, a run
requested as `example` uses `runs/example_1`, then `runs/example_2`.

## Single-model configs

Select one model with `training.model_name`:

```yaml
training:
  device: auto
  model_name: vanillalstm
```

The config must contain a section with the same lowercase model name. Train and
evaluate the run with:

```bash
python3 training/common/train_integrated.py \
  --config training/common/config.yaml \
  --name vanillalstm_baseline

python3 training/common/evaluation_integrated.py \
  --config training/common/config.yaml \
  --name vanillalstm_baseline
```

## Multi-model configs

Select an ordered model list with `training.models`:

```yaml
training:
  device: auto
  models:
    - vanillalstm
    - lookbackmean2d
    - linearar2d
```

Each listed model needs its own top-level config section. See
`training/configs/config_safe_comparison.yaml` for a complete example. One
training command runs each model and writes one subdirectory per model:

```text
runs/safe_comparison/
  config.yaml
  vanillalstm/
    checkpoints/
    <chunk_id>_training_log.csv
    <chunk_id>_training_summary.txt
  lookbackmean2d/
    checkpoints/
    ...
  linearar2d/
    checkpoints/
    ...
```

Use the same config and run name for evaluation:

```bash
python3 training/common/train_integrated.py \
  --config training/configs/config_safe_comparison.yaml \
  --name safe_comparison

python3 training/common/evaluation_integrated.py \
  --config training/configs/config_safe_comparison.yaml \
  --name safe_comparison
```

Evaluation requires `--name` or `--output-dir`. It checks that the run directory
and every configured chunk checkpoint exist before evaluating any chunk.

## Model directory contents

Each model directory contains training artifacts and, after evaluation, metric
tables, forecasts, reports, and plots:

```text
<model-directory>/
  <chunk_id>_training_log.csv
  <chunk_id>_training_summary.txt
  aggregate_metrics.csv
  per_frequency_metrics.csv
  per_band_metrics.csv
  report.txt
  checkpoints/
    <chunk_id>_<model>.pt
  forecasts/
    <chunk_id>_<model>_predictions.npz
    <chunk_id>_<model>_targets.npz
    <chunk_id>_<model>_metadata.json
  *.png
```

The top-level run directory stores the input config as `config.yaml`. Relative
data paths still refer to the repository root, so keep the source data available
when reproducing or evaluating a run.

## Git

Git ignores `runs/`. Archive any run artifacts that you need to retain.
