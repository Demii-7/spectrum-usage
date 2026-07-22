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

## Experiment Table

Run the rows in order. Each row is one run. Use seed 42 for every trainable
model; the seed is fixed for reproducibility and is not an experiment. "Selected"
means the setting with the lowest mean physical-site validation MAE from the
referenced completed rows. If a row fails or the current code does not support
its config, record the error, mark that row `failed` or `skipped`, and continue
with every row that does not depend on it.

| Order | Run name | Model | Training data and validation | Config path | Config changes from the preceding applicable run |
|---:|---|---|---|---|---|
| 1 | `powder_dev_2d_baselines_jun18_val15` | LookbackMean2D + LinearAR2D | June 18, three site segments; trailing 15% validation | `training/configs/powder_dev_2d_baselines.yaml` | Enable both 2D baselines; `val_fraction: 0.15`; horizons 1, 5, 15, 60 |
| 2 | `powder_vanilla_v0_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v0_jun18_val15.yaml` | Current one-layer architecture; recurrent dropout `0.0`; seed 42 |
| 3 | `powder_vanilla_v1_one_step_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v1_one_step.yaml` | From row 2, set `prediction_horizon: 1`; evaluate recursive rollout through 60 minutes |
| 4 | `powder_vanilla_v1_direct60_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v1_direct60.yaml` | From row 2, set `prediction_horizon: 60` |
| 5 | `powder_vanilla_v2_h64_l1_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v2_h64_l1.yaml` | Use selected V1 prediction strategy; hidden 64; one layer; dropout 0.0 |
| 6 | `powder_vanilla_v2_h128_l1_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v2_h128_l1.yaml` | From row 5, hidden 128 |
| 7 | `powder_vanilla_v2_h128_l2_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v2_h128_l2.yaml` | Hidden 128; two layers; dropout 0.1 |
| 8 | `powder_vanilla_v2_h256_l2_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v2_h256_l2.yaml` | Hidden 256; two layers; dropout 0.2 |
| 9 | `powder_vanilla_v3_lb015_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v3_lb015.yaml` | Use selected V2 model; lookback 15 |
| 10 | `powder_vanilla_v3_lb030_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v3_lb030.yaml` | From row 9, lookback 30 |
| 11 | `powder_vanilla_v3_lb060_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v3_lb060.yaml` | From row 9, lookback 60 |
| 12 | `powder_vanilla_v3_lb120_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v3_lb120.yaml` | From row 9, lookback 120 |
| 13 | `powder_vanilla_v3_lb240_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v3_lb240.yaml` | From row 9, lookback 240 |
| 14 | `powder_vanilla_v4_lr1e3_wd0_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v4_lr1e3_wd0.yaml` | Use selected V3 config; Adam; LR 0.001; weight decay 0; gradient clip 1.0 |
| 15 | `powder_vanilla_v4_lr3e4_wd0_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v4_lr3e4_wd0.yaml` | Adam; LR 0.0003; weight decay 0; gradient clip 1.0 |
| 16 | `powder_vanilla_v4_lr3e4_wd1e4_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v4_lr3e4_wd1e4.yaml` | Adam; LR 0.0003; weight decay 0.0001; gradient clip 1.0 |
| 17 | `powder_vanilla_v4_lr1e4_wd1e4_jun18_val15` | VanillaLSTM | Same as row 1 | `training/configs/powder_vanilla_v4_lr1e4_wd1e4.yaml` | Adam; LR 0.0001; weight decay 0.0001; gradient clip 1.0 |
| 18 | `powder_dev_4d_baselines_jun28_7site_val15` | LookbackMean4D + LinearAR4D | Combined June 28 seven-node map; trailing 15% validation | `training/configs/powder_dev_4d_baselines.yaml` | Enable both 4D baselines; fixed map geometry; `val_fraction: 0.15`; site and grid metrics |
| 19 | `powder_convlstm_c1_current_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c1_current.yaml` | Current ConvLSTM architecture and map representation; seed 42 |
| 20 | `powder_convlstm_c2_one_step_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c2_one_step.yaml` | From row 19, set `prediction_horizon: 1`; evaluate recursive rollout through 60 minutes |
| 21 | `powder_convlstm_c2_direct60_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c2_direct60.yaml` | From row 19, set `prediction_horizon: 60` |
| 22 | `powder_convlstm_c3_hc8_l1_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c3_hc8_l1.yaml` | Use selected C2 strategy; hidden `[8]`; decoder 8; dropout 0; kernel 3x3 |
| 23 | `powder_convlstm_c3_hc16_l1_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c3_hc16_l1.yaml` | Hidden `[16]`; decoder 16; dropout 0; kernel 3x3 |
| 24 | `powder_convlstm_c3_hc16_32_l2_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c3_hc16_32_l2.yaml` | Hidden `[16, 32]`; decoder 16; dropout 0.1; kernel 3x3 |
| 25 | `powder_convlstm_c3_hc32_64_l2_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c3_hc32_64_l2.yaml` | Hidden `[32, 64]`; decoder 32; dropout 0.3; retain current kernel unless config supports 3x3 |
| 26 | `powder_convlstm_c4_grid04x04_jun28_7site_val15` | ConvLSTM | Same seven June 28 nodes; trailing 15% validation | `training/configs/powder_convlstm_c4_grid04x04.yaml` | Use selected C3 model; rebuild map at 4x4; unique map cache name |
| 27 | `powder_convlstm_c4_grid06x06_jun28_7site_val15` | ConvLSTM | Same as row 26 | `training/configs/powder_convlstm_c4_grid06x06.yaml` | Rebuild map at 6x6; unique map cache name |
| 28 | `powder_convlstm_c4_grid10x10_jun28_7site_val15` | ConvLSTM | Same as row 26 | `training/configs/powder_convlstm_c4_grid10x10.yaml` | Rebuild map at 10x10; unique map cache name |
| 29 | `powder_convlstm_c5_adam_2e4_4e3_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c5_adam_2e4_4e3.yaml` | Use selected C4 config; Adam; LR 0.0002; weight decay 0.004 |
| 30 | `powder_convlstm_c5_adam_1e4_1e4_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c5_adam_1e4_1e4.yaml` | Adam; LR 0.0001; weight decay 0.0001 |
| 31 | `powder_convlstm_c5_adamw_3e4_1e4_jun28_7site_val15` | ConvLSTM | Same as row 18 | `training/configs/powder_convlstm_c5_adamw_3e4_1e4.yaml` | AdamW; LR 0.0003; weight decay 0.0001 |
| 32 | `powder_vanilla_final_all_prejul` | VanillaLSTM | June 18 + June 28 + June 30 as separate segments; trailing 15% validation | `training/configs/powder_vanilla_final_all_prejul.yaml` | Selected V4 settings; replace training paths with all pre-July 2D files |
| 33 | `powder_convlstm_final_jun28_7site` | ConvLSTM | Combined June 28 seven-node map; trailing 15% validation | `training/configs/powder_convlstm_final_jun28_7site.yaml` | Selected C5 settings; retain the selected grid and map representation |
| 34 | `powder_final_eval_jul03_vanilla` | VanillaLSTM evaluation | Row 32 checkpoint; July 3 seven sites as test only | Row 32 config plus July 3 evaluation paths | Evaluate horizons 1, 5, 15, 60; write aggregate, frequency, band, and site metrics |
| 35 | `powder_final_eval_jul03_convlstm` | ConvLSTM evaluation | Row 33 checkpoint; July 3 same seven-node map as test only | Row 33 config plus July 3 evaluation map | Reuse June 28 normalization and selected grid; report site and full-grid metrics |
| 36 | `powder_final_eval_jul03_baselines_2d` | 2D baseline evaluation | Selected row 1 baseline settings; July 3 seven sites as test only | `training/configs/powder_final_eval_jul03_baselines_2d.yaml` | Match row 34 inputs, target rows, lookback, and horizons |
| 37 | `powder_final_eval_jul03_baselines_4d` | 4D baseline evaluation | Selected row 18 baseline settings; July 3 seven-node map as test only | `training/configs/powder_final_eval_jul03_baselines_4d.yaml` | Match row 35 map, target rows, lookback, and horizons |

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

Do not modify source code. This runbook is config-only. The agent may:

1. Reuse an existing config and override only experiment settings.
2. Create a new config under `training/configs/`.
3. Reuse an existing analysis or plotting script.
4. Write generated tables and figures under the applicable run directory.

Do not edit Python files, model implementations, loaders, evaluators, plotting
code, analysis code, location metadata, annotations, or repository structure.
Do not add scripts. Do not refactor, rename, or clean up unrelated files. If an
experiment cannot run with existing code and config settings, record the exact
error, mark only that run `skipped`, and continue with the remaining independent
rows. Do not implement a fix and do not ask for approval to implement one.

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
leakage. Run July 3 evaluation only after selecting a configuration from
validation data drawn from the same dataset as training.

## Validation and Test Policy

Validation must always come from the same dataset used for training. Use a
chronological trailing holdout, not a different collection and not a random
sample. The default validation fraction is `0.15`.

For the first development stage:

```text
Training input: June 18 collection
Validation:     final 15% of the June 18 training input
Test:           July 3 collection, locked
```

The existing shared trainer already creates a chronological trailing validation
split with `val_fraction`. Use that behavior for the first baselines and model
experiments. Do not modify partition code before running the config-only
experiments.

When training expands to June 18, June 28, and June 30, those pre-July
collections together form the training dataset. Validation must still be a
chronological trailing 15% holdout from that same pre-July dataset. July 3
remains the separate test dataset.

Preserve source files as temporal segments. Windows must not cross site or
collection boundaries. The current segment-aware window generation should be
verified with a smoke test before each full run. If the existing trailing split
places validation windows in only the final concatenated site, record that
limitation and continue with the initial config-only experiments. Do not redesign
the partition system. Record the limitation in each affected run.

The current trainer selects checkpoints by validation MSE. Use that behavior
throughout this runbook. Report both validation MSE and evaluation MAE without
claiming they are the same selection criterion.

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
powder_dev_2d_baselines_jun18_val15
powder_dev_4d_baselines_jun28_7site_val15
```

The 2D run must include `lookbackmean2d` and `linearar2d` on June 18 with its
trailing 15% validation split. The 4D run must include `lookbackmean4d` and
`linearar4d` on all seven June 28 nodes with its trailing 15% validation split.
Evaluate the 4D baselines at physical site cells as well as over the full grid.

## VanillaLSTM Development Stage

### V0: Reproduce the current model

Create:

```text
training/configs/powder_vanilla_v0_jun18_val15.yaml
```

Run name:

```text
powder_vanilla_v0_jun18_val15
```

Use the current architecture and seed 42. Fix the current warning before
running: a one-layer PyTorch LSTM does not apply recurrent dropout. Set dropout
to `0.0` for one layer or use two layers. For V0, set dropout to `0.0` so the
architecture remains one layer.

Train on all June 18 site segments and use `val_fraction: 0.15` from that same
training input. Do not evaluate July 3 until V0 and the experiments below have
been selected using the June 18 validation holdout.

### V1: One-step versus direct 60-step prediction

Create two configs:

```text
training/configs/powder_vanilla_v1_one_step.yaml
training/configs/powder_vanilla_v1_direct60.yaml
```

Run names:

```text
powder_vanilla_v1_one_step_jun18_val15
powder_vanilla_v1_direct60_jun18_val15
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
powder_vanilla_v2_<suffix>_jun18_val15
```

If a candidate runs out of memory or produces non-finite loss, mark that row
`failed` and continue with the next row. Record candidates whose validation MAE
is worse than LookbackMean at every horizon. Do not increase capacity beyond
the listed rows when the training-validation gap grows.

### V3: Lookback sweep

Use the best V2 architecture. Test lookbacks 15, 30, 60, 120, and 240 minutes.
Keep the forecast horizons `[1, 5, 15, 60]`.

Name runs as:

```text
powder_vanilla_v3_lb015_jun18_val15
powder_vanilla_v3_lb030_jun18_val15
powder_vanilla_v3_lb060_jun18_val15
powder_vanilla_v3_lb120_jun18_val15
powder_vanilla_v3_lb240_jun18_val15
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
powder_vanilla_v4_<suffix>_jun18_val15
```

Use early stopping with patience 10 and a maximum of 100 epochs. Save the best
validation checkpoint, not the last checkpoint.

## ConvLSTM Spatial Data Policy

ConvLSTM training and test must use the same set of physical nodes. Use these
seven nodes for both datasets:

```text
cpg-nuc1
ebc-nuc1
guesthouse-nuc1
humanities-nuc1
madsen-nuc1
moran-nuc1
sagepoint-nuc1
```

Use the combined `20260628T0436Z` and `20260628T0437Z` collection as the
ConvLSTM training dataset. Align the seven nodes by `timestamp_utc`. Use the
chronological trailing 15% of this June 28 map sequence for validation. Use the
same seven nodes from `20260703T1839Z` only for final testing.

Do not include June 18 or June 30 in the main ConvLSTM run because those
collections have different node sets.

Start with the existing map builder and ConvLSTM implementation. Do not add
masks, new losses, raw-site target tensors, cache fingerprinting, or a new map
loader before running the config-only baselines and C1 reproduction. First
verify:

1. All seven expected files appear in the June 28 training map metadata.
2. All seven expected files appear in the July 3 test map metadata.
3. The training and test maps use identical grid dimensions and site indices.
4. June 28 normalization is reused for July 3 evaluation.
5. No July 3 values are used to fit normalization or select an epoch.
6. The CPG location resolves correctly. If it does not, skip the ConvLSTM rows,
   record the error, and continue the VanillaLSTM rows and reports.
7. Site-level metrics are reported for all seven nodes.

Treat full-grid metrics as secondary because most grid cells are interpolated.
Use the existing site-level evaluator as the primary comparison. Record that
site values come from mapped grid cells. Do not add masks, raw-site targets, or
new loss weighting in this runbook.

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
powder_convlstm_c1_current_jun28_7site_val15
```

Train on the seven-node June 28 map and use its chronological trailing 15% for
validation. Do not use July 3 during this stage. Record validation metrics for
all seven mapped site cells.

### C2: One-step versus direct 60-step prediction

Create:

```text
training/configs/powder_convlstm_c2_one_step.yaml
training/configs/powder_convlstm_c2_direct60.yaml
```

Run names:

```text
powder_convlstm_c2_one_step_jun28_7site_val15
powder_convlstm_c2_direct60_jun28_7site_val15
```

Keep architecture and loss fixed. Compare site-level MAE at horizons 1, 5, 15,
and 60. The current direct 60-step model is a much larger output problem; do not
assume it is preferable.

### C3: Smaller architecture sweep

The current ConvLSTM has enough capacity to overfit June 28. Test smaller
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
powder_convlstm_c3_<suffix>_jun28_7site_val15
```

### C4: Grid resolution

Compare these fixed grid sizes while keeping the same seven nodes:

```text
powder_convlstm_c4_grid04x04_jun28_7site_val15
powder_convlstm_c4_grid06x06_jun28_7site_val15
powder_convlstm_c4_grid10x10_jun28_7site_val15
```

Rebuild separate map caches for each grid. Include grid size in the cache name
so one experiment cannot load another experiment's cache.

### C5: Optimizer and regularization

Use the best architecture and grid with the existing input and normalization.
Test:

| Suffix | Optimizer | Learning rate | Weight decay |
|---|---|---:|---:|
| `adam_2e4_4e3` | Adam | 0.0002 | 0.004 |
| `adam_1e4_1e4` | Adam | 0.0001 | 0.0001 |
| `adamw_3e4_1e4` | AdamW | 0.0003 | 0.0001 |

Do not add another optimizer unless a later experiment explicitly requests it.
Name runs as:

```text
powder_convlstm_c5_<suffix>_jun28_7site_val15
```

Use a maximum of 150 epochs, patience 20, gradient clipping, and the best
validation checkpoint. Plot training and validation losses. Reject a candidate
with non-finite values or a widening training-validation gap without improved
site-level MAE.

## Expanded-Data Stage

Perform this stage only after selecting configurations from chronological
validation holdouts drawn from each model's training dataset.

### VanillaLSTM expanded training

Train on June 18, June 28, and June 30 as separate chronological segments. Use
a chronological trailing 15% holdout from this same combined pre-July training
dataset. Never let a window cross from one site or collection into another.

Run name: `powder_vanilla_final_all_prejul`.

### ConvLSTM final training

Keep the seven-node June 28 training dataset for the final ConvLSTM. It is the
only pre-July collection with the same seven nodes as the July 3 test dataset.
Use its chronological trailing 15% for validation. Do not add June 18 or June 30
to this run because their node sets differ.

Run name: `powder_convlstm_final_jun28_7site`.

## Final July 3 Evaluation

Evaluate the selected VanillaLSTM and ConvLSTM runs on all seven July 3 sites.
Also evaluate matching LookbackMean and LinearAR baselines. Use the four final
evaluation names in rows 34 through 37. Use existing reporting scripts; if one
report is unavailable, record the omission and produce every other report.

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
3. Difference from LookbackMean at each site and horizon.
4. Metrics by annotated frequency region.
5. Entropy-versus-MAE rectangle plots produced by the committed analysis and
   plotting scripts.

Use these final evaluation labels:

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
4. Confirm test files are disjoint from training files. Confirm validation is a
   chronological trailing 15% holdout from the same training dataset.
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

Set `status` to one of `planned`, `running`, `completed`, `failed`, `skipped`,
or `rejected`. Record the full error in `notes` for failed and skipped runs. Do
not silently rerun a failed name and replace its record. Continue with the next
independent table row after recording a failure.

## Selection Rules

Use these rules in order:

1. Reject data leakage, missing-site, missing-checkpoint, and non-finite runs.
2. Select by mean physical-site validation MAE across horizons.
3. Require the candidate to improve over its same-representation LookbackMean
   baseline at least at one target horizon.
4. Prefer the smaller model when validation differences are within 1%.
5. Report negative results. Do not hide experiments where LookbackMean wins.

Do not use full-grid ConvLSTM metrics as the primary selection criterion. The
grid contains interpolated cells, while the scientific target is forecasting
power at physical sites and evaluating spatial generalization.
