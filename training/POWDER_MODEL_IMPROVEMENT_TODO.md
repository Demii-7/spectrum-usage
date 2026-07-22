# POWDER VanillaLSTM and ConvLSTM Improvement Runbook

Use this document as an execution checklist. Do not skip validation steps, reuse
the July 3 test data during tuning, or compare models evaluated on different
sites, timestamps, frequencies, or horizons.

## Objective

Improve VanillaLSTM and ConvLSTM on POWDER 600–800 MHz forecasting while
preserving a chronological, unseen test set. Compare every experiment with
LookbackMean and LinearAR using the same inputs and target rows.

Report MAE and RMSE at horizons 1, 5, 15, and 60 minutes. For spatial models,
report both full-grid metrics and metrics at physical site locations. Treat the
physical-site metrics as the primary result.

## Remote Environment

Run all training and evaluation on:

```text
Host: cc@129.114.27.70
Repository: /home/cc/spectrum-usage
Python: /home/cc/spectrum-usage/.venv/bin/python
Branch: integrate
Run root: /home/cc/spectrum-usage/runs
```

Before each work session:

```bash
git -C /home/cc/spectrum-usage status --short --branch
git -C /home/cc/spectrum-usage pull --ff-only origin integrate
/home/cc/spectrum-usage/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

Do not delete or overwrite an existing run. The trainer adds `_1`, `_2`, and so
on when a requested run directory exists. Record the actual directory printed
at the end of training.

## Code-Change Policy

Do not modify application code unless a required experiment cannot run
correctly with the current implementation. Prefer these actions in order:

1. Reuse an existing config and override only experiment settings.
2. Create a new config under `training/configs/`.
3. Reuse an existing analysis or plotting script.
4. Add a small analysis script when no existing script produces the required
   table.
5. Modify shared training, loading, model, or evaluation code only when a
   concrete prerequisite in this runbook is missing.

Do not refactor working code while preparing an experiment. Do not rename files,
reorganize directories, replace APIs, introduce compatibility layers, or clean
up unrelated style. Do not modify model architecture code merely to expose a
setting that the config already controls.

Before changing shared code:

1. Write down the exact experiment blocked by the current behavior.
2. Identify the smallest function that must change.
3. Add a focused regression test that fails before the change.
4. Make the smallest change that passes the test.
5. Run existing tests and a bounded smoke test.
6. Commit the code change separately from configs and generated results.
7. Push the commit and pull it on the remote before starting training.

If an experiment can run correctly without a code change, do not change code.
If a requested feature would require a broad redesign, stop and record the
blocker in the experiment index instead of improvising an architecture.

## Available 600–800 MHz Collections

Use these exact collection IDs and site names:

| Collection | Sites |
|---|---|
| `20260618T0036Z` | `guesthouse-nuc1`, `humanities-nuc1`, `law73-nuc1` |
| `20260628T0436Z` | `ebc-nuc1`, `guesthouse-nuc1`, `humanities-nuc1` |
| `20260628T0437Z` | `cpg-nuc1`, `madsen-nuc1`, `moran-nuc1`, `sagepoint-nuc1` |
| `20260630T1949Z` | `law73-nuc1`, `web-nuc1` |
| `20260703T1839Z` | `cpg-nuc1`, `ebc-nuc1`, `guesthouse-nuc1`, `humanities-nuc1`, `madsen-nuc1`, `moran-nuc1`, `sagepoint-nuc1` |

The files are under:

```text
data/powder/<site>/<collection>/600_800/power_1mhz_avg_per_minute.csv
```

June 28 uses two collection IDs because four nodes started one minute later.
Treat `20260628T0436Z` and `20260628T0437Z` as one collection. Align those files
by `timestamp_utc`; never align them by row number.

## Locked Test Policy

Reserve all `20260703T1839Z` data for final testing. Do not use it to:

- fit normalization statistics;
- choose model size, lookback, learning rate, dropout, weight decay, loss, or
  early-stopping epoch;
- inspect training curves during tuning;
- build interpolation parameters or choose map geometry;
- decide which frequencies, regions, or sites to report.

Running the final test repeatedly and selecting the best result is test-set
leakage. During development, use June 28 as the chronological validation
collection. Run July 3 evaluation only after selecting a configuration from
June 28 validation results.

## Required Data-Split Work

The current trainer applies `val_fraction` to the tail of a concatenated
training array. That is insufficient for multi-collection experiments because
it can place portions of different sites or collections on both sides of the
split. Complete this work before tuning models:

1. Add explicit `train`, `validation`, and `test` partitions to the shared data
   configuration and loader.
2. Preserve each source file as a separate temporal segment. Windows must not
   cross site or collection boundaries.
3. Fit normalization on the training partition only.
4. Apply the saved training normalization to validation and test data.
5. Reject configurations where a file appears in more than one partition.
6. Record source paths, row counts, timestamp ranges, and segment counts in the
   run metadata.
7. Add a test proving that no lookback or target window crosses a segment
   boundary.
8. Add a test proving that validation and test timestamps occur after all
   training timestamps for the chronological protocol.
9. Split data before imputation or other temporal preprocessing. Training
   values must not fill validation gaps, and validation values must not fill
   training gaps.
10. Preserve timestamps in split metadata so chronology checks and run
    manifests can inspect actual ranges.
11. Update config validation, `load_chunk()`, trainer inputs, evaluator inputs,
    and checkpoint normalization checks together. Do not implement a third
    partition in only one layer of the pipeline.
12. Add validation MAE logging and an explicit checkpoint-selection metric.
    The current trainer selects checkpoints by MSE. Do not claim selection by
    MAE until the trainer supports and records it.

Use this development split after the loader supports it:

```text
Training:   June 18 collection
Validation: June 28 collection
Test:       July 3 collection, locked
```

Use June 30 only in the later expanded-data stage. Its two-site composition is
not comparable with the seven-site July 3 collection, so do not use June 30 as
the only validation collection for spatial models.

## Required Baselines

Before changing either neural model, run LookbackMean and LinearAR with every
new split and representation. A neural model is an improvement only if it beats
LookbackMean on the same site-level targets.

Create these development configs:

```text
training/configs/powder_dev_2d_baselines.yaml
training/configs/powder_dev_4d_baselines.yaml
```

Use these run names:

```text
powder_dev_2d_baselines_jun18_val_jun28
powder_dev_4d_baselines_jun18_val_jun28
```

The 2D run must include `lookbackmean2d` and `linearar2d`. The 4D run must
include `lookbackmean4d` and `linearar4d`. Evaluate the 4D baselines at physical
site cells as well as over the full grid.

## VanillaLSTM Development Stage

### V0: Reproduce the current model

Create:

```text
training/configs/powder_vanilla_v0_jun18_val_jun28.yaml
```

Run name:

```text
powder_vanilla_v0_jun18_val_jun28
```

Use the current architecture and seed 42. Fix the current warning before
running: a one-layer PyTorch LSTM does not apply recurrent dropout. Set dropout
to `0.0` for one layer or use two layers. For V0, set dropout to `0.0` so the
architecture remains one layer.

Train on all June 18 site segments. Validate on all June 28 site segments. Do
not evaluate July 3 until V0 and the experiments below have been selected using
June 28.

### V1: One-step versus direct 60-step prediction

Create two configs:

```text
training/configs/powder_vanilla_v1_one_step.yaml
training/configs/powder_vanilla_v1_direct60.yaml
```

Run names:

```text
powder_vanilla_v1_one_step_jun18_val_jun28
powder_vanilla_v1_direct60_jun18_val_jun28
```

For `one_step`, set `prediction_horizon: 1` and use autoregressive rollout to
60 minutes. For `direct60`, set `prediction_horizon: 60`. Keep all other
settings equal. Compare validation MAE at every horizon, not only average loss.

### V2: Capacity sweep

Use the better prediction strategy from V1. Run these candidates with seed 42:

| Run suffix | Hidden size | Layers | Dropout |
|---|---:|---:|---:|
| `h64_l1` | 64 | 1 | 0.0 |
| `h128_l1` | 128 | 1 | 0.0 |
| `h128_l2` | 128 | 2 | 0.1 |
| `h256_l2` | 256 | 2 | 0.2 |

Name runs as:

```text
powder_vanilla_v2_<suffix>_jun18_val_jun28
```

Stop a candidate if it runs out of memory, produces non-finite loss, or its
validation MAE is worse than LookbackMean at every horizon after early
stopping. Do not increase capacity further when the training-validation gap
grows.

### V3: Lookback sweep

Use the best V2 architecture. Test lookbacks 15, 30, 60, 120, and 240 minutes.
Keep the forecast horizons `[1, 5, 15, 60]`.

Name runs as:

```text
powder_vanilla_v3_lb015_jun18_val_jun28
powder_vanilla_v3_lb030_jun18_val_jun28
powder_vanilla_v3_lb060_jun18_val_jun28
powder_vanilla_v3_lb120_jun18_val_jun28
powder_vanilla_v3_lb240_jun18_val_jun28
```

### V4: Optimizer and regularization sweep

Use the best architecture and lookback. Test this small grid:

| Suffix | Learning rate | Weight decay | Gradient clip |
|---|---:|---:|---:|
| `lr1e3_wd0` | 0.001 | 0.0 | 1.0 |
| `lr3e4_wd0` | 0.0003 | 0.0 | 1.0 |
| `lr3e4_wd1e4` | 0.0003 | 0.0001 | 1.0 |
| `lr1e4_wd1e4` | 0.0001 | 0.0001 | 1.0 |

Name runs as:

```text
powder_vanilla_v4_<suffix>_jun18_val_jun28
```

Use early stopping with patience 10 and a maximum of 100 epochs. Save the best
validation checkpoint, not the last checkpoint.

### V5: Per-site normalization

The current 2D row-concatenated input shares per-frequency normalization across
sites. Add an option to fit normalization by site and frequency using training
data only. Compare:

```text
powder_vanilla_v5_global_norm_jun18_val_jun28
powder_vanilla_v5_site_norm_jun18_val_jun28
```

Store the site-specific means and standard deviations in the checkpoint. The
evaluator must reject a site that has no applicable normalization policy unless
an explicit fallback is configured. Because five July 3 sites are absent from
June 18, define the fallback before running this experiment. The fallback must
use training-derived global per-frequency statistics; it must not fit statistics
from June 28 or July 3.

### V6: Seed confirmation

Run the selected VanillaLSTM configuration with seeds 7, 42, and 2026:

```text
powder_vanilla_v6_seed007_jun18_val_jun28
powder_vanilla_v6_seed042_jun18_val_jun28
powder_vanilla_v6_seed2026_jun18_val_jun28
```

Report mean and standard deviation of validation MAE. Select the configuration,
not the best seed.

## ConvLSTM Spatial Data Work

Do not tune ConvLSTM until the map pipeline handles multiple collections and
changing site availability correctly.

1. Build one map segment per collection. Do not intersect timestamps across
   June 18 and June 28 because those collections do not overlap in time.
2. Within a collection, align sites by `timestamp_utc`.
3. Preserve collection boundaries as sequence boundaries.
4. Use the fixed physical coordinates in `data/locations/powder.json`.
5. Add or verify a `cpg-nuc1` alias for the `central parking garage` location
   entry before building June 28 or July 3 maps. Add a focused alias-resolution
   test only if the current code cannot resolve both names.
6. Keep grid geometry identical across train, validation, and test. Fix the
   geographic bounds, origin, dimensions, and resolution in config. Do not let
   each collection derive its extent from the sites present in that collection.
7. Add an observation mask with one at measured site cells and zero elsewhere.
8. Keep the observation mask separate from the interpolated power map.
9. Keep aligned raw site-value targets separate from the interpolated map. Use
   raw site values for observed-site loss and site metrics. Do not treat the
   nearest interpolated grid cell as a physical measurement.
10. Detect grid-cell collisions when two sites quantize to the same cell. Stop
    map generation if a collision makes site-level targets ambiguous.
11. Define a missing-timestamp policy and record a time-varying observation
    mask. Do not silently change interpolation support as sites appear or
    disappear at individual timestamps.
12. Fit normalization on June 18 training maps only.
13. Do not use July 3 values to fit interpolation, normalization, or map
   parameters.
14. Save site names, site coordinates, grid indices, source files, timestamp
    ranges, and observation masks in map metadata.
15. Fingerprint map caches using source files, collection IDs, site set,
    geographic bounds, grid dimensions, interpolation settings, masks, and
    preprocessing settings. Do not reuse a cache based only on map name and
    partition.

The primary ConvLSTM loss should initially use aligned raw physical-site targets
only. Do not assume a fixed number of observed grid cells: site availability
changes by collection, and grid collisions must be checked. Keep full-grid loss
as an optional secondary term.

Implement configurable loss weights:

```yaml
convlstm:
  train:
    observed_site_loss_weight: 1.0
    interpolated_grid_loss_weight: 0.0
```

Later test interpolated-grid weights `0.01`, `0.1`, and `1.0`, but retain
observed-site MAE as the model-selection metric.

## ConvLSTM Development Stage

### C0: Same-representation baselines

Confirm that `lookbackmean4d` and `linearar4d` work on the exact maps,
normalization, masks, windows, and target rows used by ConvLSTM. Do not compare
ConvLSTM full-grid MAE against LookbackMean2D site MAE.

### C1: Reproduce the current ConvLSTM

Create:

```text
training/configs/powder_convlstm_c1_current.yaml
```

Run name:

```text
powder_convlstm_c1_current_jun18_val_jun28
```

Use June 18 maps for training and June 28 maps for validation. Use observed-site
loss only. Record site-level validation metrics separately for Guesthouse,
Humanities, and every June 28 site. Mark June 28 sites absent from June 18 as
spatial generalization sites.

### C2: One-step versus direct 60-step prediction

Create:

```text
training/configs/powder_convlstm_c2_one_step.yaml
training/configs/powder_convlstm_c2_direct60.yaml
```

Run names:

```text
powder_convlstm_c2_one_step_jun18_val_jun28
powder_convlstm_c2_direct60_jun18_val_jun28
```

Keep architecture and loss fixed. Compare site-level MAE at horizons 1, 5, 15,
and 60. The current direct 60-step model is a much larger output problem; do not
assume it is preferable.

### C3: Smaller architecture sweep

The current ConvLSTM has enough capacity to overfit June 18. Test smaller
models before larger ones:

| Suffix | Hidden channels | Encoder layers | Decoder channels | Dropout |
|---|---|---:|---:|---:|
| `hc8_l1` | `[8]` | 1 | 8 | 0.0 |
| `hc16_l1` | `[16]` | 1 | 16 | 0.0 |
| `hc16_32_l2` | `[16, 32]` | 2 | 16 | 0.1 |
| `hc32_64_l2` | `[32, 64]` | 2 | 32 | 0.3 |

Use `3 × 3` spatial kernels for the first three candidates. The current `1 × 3`
kernel only spans one map axis. Name runs as:

```text
powder_convlstm_c3_<suffix>_jun18_val_jun28
```

### C4: Grid resolution

Two or three observed June 18 sites do not justify a dense `10 × 10` target
grid without a mask-aware loss. Compare fixed grids:

```text
powder_convlstm_c4_grid04x04_jun18_val_jun28
powder_convlstm_c4_grid06x06_jun18_val_jun28
powder_convlstm_c4_grid10x10_jun18_val_jun28
```

Rebuild separate map caches for each grid. Include grid size in the cache name
so one experiment cannot load another experiment's cache.

### C5: Map input representation

Compare these inputs while retaining observed-site loss:

1. Interpolated power map only.
2. Interpolated power map plus observation-mask channel.
3. Sparse observed values plus mask, with unobserved values set to the training
   mean.

Name runs:

```text
powder_convlstm_c5_interpolated_jun18_val_jun28
powder_convlstm_c5_interpolated_mask_jun18_val_jun28
powder_convlstm_c5_sparse_mask_jun18_val_jun28
```

This experiment determines whether ConvLSTM learns RF behavior or artifacts of
the interpolation procedure.

### C6: Normalization

Compare global per-frequency normalization with site-aware normalization at
observed cells. Do not fit a separate normalization from June 28 or July 3.

```text
powder_convlstm_c6_global_norm_jun18_val_jun28
powder_convlstm_c6_site_norm_jun18_val_jun28
```

### C7: Optimizer and regularization

Use the best architecture, grid, input, and normalization. Test:

| Suffix | Optimizer | Learning rate | Weight decay |
|---|---|---:|---:|
| `adam_2e4_4e3` | Adam | 0.0002 | 0.004 |
| `adam_1e4_1e4` | Adam | 0.0001 | 0.0001 |
| `adamw_3e4_1e4` | AdamW | 0.0003 | 0.0001 |
| `nadam_2e4_4e3` | NAdam | 0.0002 | 0.004 |

Add NAdam support before the last candidate. Name runs as:

```text
powder_convlstm_c7_<suffix>_jun18_val_jun28
```

Use a maximum of 150 epochs, patience 20, gradient clipping, and the best
validation checkpoint. Plot training and validation losses. Reject a candidate
with non-finite values or a widening training-validation gap without improved
site-level MAE.

### C8: Seed confirmation

Run the selected ConvLSTM configuration with seeds 7, 42, and 2026:

```text
powder_convlstm_c8_seed007_jun18_val_jun28
powder_convlstm_c8_seed042_jun18_val_jun28
powder_convlstm_c8_seed2026_jun18_val_jun28
```

Report mean and standard deviation across seeds for every site and horizon.

## Expanded-Data Stage

Perform this stage only after selecting VanillaLSTM and ConvLSTM configurations
from June 28 validation results.

### VanillaLSTM expanded training

Train on June 18, June 28, and June 30 as separate chronological segments. Use
the final 15% of each June 30 site segment for validation, or add explicit
validation segments from the end of the latest pre-July collection. Never let a
window cross from one site or collection into another.

Run names:

```text
powder_vanilla_final_all_prejul_seed007
powder_vanilla_final_all_prejul_seed042
powder_vanilla_final_all_prejul_seed2026
```

### ConvLSTM expanded training

June 30 has only Law73 and WEB, while July 3 has seven different sites. Do not
make June 30 the sole spatial validation set. Build separate map segments for
June 18, June 28, and June 30. Train on June 18 plus June 28. Use a chronological
tail of June 28 as the primary validation segment and June 30 as a secondary
site-availability stress test. After selecting the epoch policy, optionally fit
on all pre-July segments while retaining a pre-July chronological validation
tail.

Run names:

```text
powder_convlstm_final_prejul_seed007
powder_convlstm_final_prejul_seed042
powder_convlstm_final_prejul_seed2026
```

## Final July 3 Evaluation

Evaluate only the selected three-seed VanillaLSTM and ConvLSTM runs on all seven
July 3 sites. Also evaluate matching LookbackMean and LinearAR baselines.

Evaluate each seed in its own existing run directory. Do not overwrite one
seed's metrics with another seed. After all seed evaluations finish, run a
separate aggregation script that reads the three result tables and writes mean
and standard deviation tables. If no aggregation script exists, add one focused
analysis script under `evaluation/analysis/`; do not modify the evaluator to
perform unrelated cross-run orchestration.

Required output tables:

```text
aggregate_metrics.csv
per_frequency_metrics.csv
per_band_metrics.csv
site_metrics.csv
```

Required summaries:

1. MAE and RMSE by model and horizon.
2. MAE and RMSE by physical site and horizon.
3. Mean and standard deviation across seeds.
4. Difference from LookbackMean at each site and horizon.
5. Metrics by annotated frequency region.
6. Entropy-versus-MAE rectangle plots produced by the committed analysis and
   plotting scripts.

Use final evaluation labels when writing the cross-seed summaries:

```text
powder_final_eval_jul03_vanilla
powder_final_eval_jul03_convlstm
powder_final_eval_jul03_baselines_2d
powder_final_eval_jul03_baselines_4d
```

## Per-Run Checklist

For every experiment:

1. Create a config under `training/configs/` with the run name in the filename.
2. Confirm all configured files exist.
3. Print each file's first and last timestamp and row count.
4. Confirm the training, validation, and test file sets are disjoint.
5. Confirm normalization uses training data only.
6. Run a bounded smoke test before the full run.
7. Run training with an explicit `--name` or `--output-dir`.
8. Confirm the training summary reports the expected epochs and best epoch.
9. Confirm every expected checkpoint exists.
10. Run evaluation with `--skip-plots` first.
11. Confirm no chunk or site was skipped.
12. Confirm metric tables contain horizons 1, 5, 15, and 60.
13. Generate plots only after metric validation succeeds.
14. Record the Git commit, config path, actual run directory, GPU, duration, and
    result summary in an experiment index CSV.

## Experiment Index

Maintain:

```text
runs/powder_experiment_index.csv
```

Use these columns:

```text
run_name,model,config_path,git_commit,seed,train_collections,
validation_collections,test_collections,sites,representation,lookback,
prediction_horizon,best_epoch,best_validation_loss,run_directory,status,notes
```

Set `status` to one of `planned`, `running`, `completed`, `failed`, or
`rejected`. Record the full error in `notes` for failed runs. Do not silently
rerun a failed name and replace its record.

## Selection Rules

Use these rules in order:

1. Reject data leakage, missing-site, missing-checkpoint, and non-finite runs.
2. Select by mean physical-site validation MAE across horizons.
3. Require the candidate to improve over its same-representation LookbackMean
   baseline at least at one target horizon.
4. Prefer the smaller model when validation differences are within 1%.
5. Prefer stable seed performance over one exceptional seed.
6. Report negative results. Do not hide experiments where LookbackMean wins.

Do not use full-grid ConvLSTM metrics as the primary selection criterion. The
grid contains interpolated cells, while the scientific target is forecasting
power at physical sites and evaluating spatial generalization.
