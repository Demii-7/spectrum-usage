# Result inventory and configuration matrix

What exists, in one place. Every table is computed by `build_master.py` from the
result JSONs in this directory; regenerate with
`python training/SpectrumLLM/repro/build_master.py`.

The two protocol axes that dominate reproduction quality, and the coverage we
hold on each:

* **split** -- how the article's "7:3" is read. `tslib` = 70/10/20 chronological
  (Time-Series-Library `Dataset_Custom`, 563 test windows at K=12);
  `literal` = last 30% all test, validation carved from the training span
  (851 test windows at K=12).
* **channel mode** -- `chan_indep` = Time-LLM's arrangement (channels folded
  into the batch); `all_channel` = the article's equation (one token spans all
  channels).

Grid cells are mean test MSE in raw PSD units over all horizons and seeds of
that cell, with the mean ratio to the published row in parentheses where one
exists. Where a cell holds more than one capacity (d_model/d_ff) the variants
are shown on separate lines. `--` = not run. Per-horizon, per-seed detail is in
`configuration_matrix.csv`; the full run list is `results_master.csv`.

**331 runs** in `results_master.csv` (281 primary, 20 sweep/diagnostic, 30 superseded lr 1e-3 campaign) | **120 primary configuration cells** in `configuration_matrix.csv` | **71 checkpoints** on disk in `CHECKPOINTS.tsv` (paths relative to the repository root; the 150 checkpoints of the first campaign exist only as SHA-256 hashes in `checkpoints_manifest.tsv` -- the weights were deleted).


## DLinear

| channels | split | all-channel | channel-independent |
|---|---|---|---|
| 4 | tslib | 0.3169 (0.937x) &middot; 5 Ks &middot; seed 2021 | -- |
| 4 | literal | -- | -- |
| 25 | tslib | 1.0234 (0.967x) &middot; 5 Ks &middot; seed 2021 | -- |
| 25 | literal | -- | -- |

## OFA (gpt2)

| channels | split | all-channel | channel-independent |
|---|---|---|---|
| 4 | tslib | -- | 0.2639 &middot; 5 Ks &middot; seed 0 |
| 4 | literal | -- | 0.2777 &middot; 5 Ks &middot; seeds 0, 1, 2 |
| 25 | tslib | -- | 0.8092 &middot; 5 Ks &middot; seed 0 |
| 25 | literal | -- | 0.7515 &middot; 5 Ks &middot; seeds 0, 1, 2 |

## Time-LLM (bert)

| channels | split | all-channel | channel-independent |
|---|---|---|---|
| 4 | tslib | -- | 0.3702 &middot; 5 Ks &middot; seed 0 |
| 4 | literal | -- | 0.2857 &middot; 5 Ks &middot; seeds 0, 1, 2 |
| 25 | tslib | -- | 0.9257 &middot; 5 Ks &middot; seed 0 |
| 25 | literal | -- | 0.7867 &middot; 5 Ks &middot; seeds 0, 1, 2 |

## Time-LLM (gpt2)

| channels | split | all-channel | channel-independent |
|---|---|---|---|
| 4 | tslib | 0.2992 (0.975x) &middot; 5 Ks &middot; seed 0<br>d64/ff128: 0.3052 (0.995x) &middot; 5 Ks &middot; seed 0 | 0.2837 (0.922x) &middot; 5 Ks &middot; seed 0 |
| 4 | literal | 0.3111 (1.013x) &middot; 5 Ks &middot; seeds 0, 1, 2, 3, 4 | 0.2871 (0.937x) &middot; 5 Ks &middot; seeds 0, 1, 2 |
| 25 | tslib | 1.1468 (1.373x) &middot; 5 Ks &middot; seeds 0, 1, 2<br>d64/ff128: 1.1966 (1.423x) &middot; 5 Ks &middot; seed 0 | 0.8245 (0.969x) &middot; 5 Ks &middot; seed 0 |
| 25 | literal | 1.0114 (1.209x) &middot; 5 Ks &middot; seeds 0, 1, 2 | 0.7419 (0.885x) &middot; 5 Ks &middot; seeds 0, 1, 2 |
| 151 | tslib | -- | -- |
| 151 | literal | 1.2430 &middot; 5 Ks &middot; seeds 0, 1, 2 | -- |

## Time-LLM (random)

| channels | split | all-channel | channel-independent |
|---|---|---|---|
| 4 | tslib | -- | -- |
| 4 | literal | 0.3157 &middot; 5 Ks &middot; seeds 0, 1, 2 | -- |
| 25 | tslib | -- | -- |
| 25 | literal | 1.1221 &middot; 5 Ks &middot; seeds 0, 1, 2 | -- |
| 151 | tslib | -- | -- |
| 151 | literal | 1.2053 &middot; 5 Ks &middot; seeds 0, 1, 2 | -- |

## Time-LLM (GPT-2) per horizon -- the four protocol cells

Mean test MSE (raw PSD) over seeds, with the ratio to the published row in parentheses.


### 4 channels

| K | tslib / all_channel | tslib / chan_indep | literal / all_channel | literal / chan_indep |
|---|---|---|---|---|
| 12 | 0.2648 (1.010x) | 0.2431 (0.928x) | 0.2717 (1.037x) | 0.2649 (1.011x) |
| 24 | 0.2781 (0.933x) | 0.2621 (0.879x) | 0.2869 (0.962x) | 0.2830 (0.949x) |
| 48 | 0.3145 (0.997x) | 0.2765 (0.877x) | 0.3069 (0.973x) | 0.2846 (0.902x) |
| 96 | 0.3345 (1.060x) | 0.3110 (0.985x) | 0.3557 (1.127x) | 0.2891 (0.916x) |
| 192 | 0.3141 (0.908x) | 0.3259 (0.942x) | 0.3340 (0.965x) | 0.3139 (0.907x) |

### 25 channels

| K | tslib / all_channel | tslib / chan_indep | literal / all_channel | literal / chan_indep |
|---|---|---|---|---|
| 12 | 1.0120 (1.641x) | 0.5814 (0.943x) | 0.8825 (1.431x) | 0.5304 (0.860x) |
| 24 | 1.0530 (1.387x) | 0.7258 (0.956x) | 0.9368 (1.234x) | 0.6633 (0.874x) |
| 48 | 1.1292 (1.319x) | 0.8543 (0.997x) | 0.9678 (1.130x) | 0.7543 (0.881x) |
| 96 | 1.1797 (1.218x) | 0.9334 (0.964x) | 1.0247 (1.058x) | 0.8449 (0.872x) |
| 192 | 1.4101 (1.352x) | 1.0278 (0.985x) | 1.2453 (1.194x) | 0.9870 (0.946x) |

## Hyperparameter coverage (primary runs)

* **backbone**: `bert` x40, `gpt2` x186, `none` x10, `random` x45
* **d_model**: `(not recorded)` x160, `32` x111, `64` x10
* **d_ff**: `(not recorded)` x160, `128` x10, `32` x111
* **patch_length**: `(not recorded)` x160, `16` x121
* **patch_stride**: `(not recorded)` x160, `8` x121
* **llm_layers**: `(not recorded)` x160, `6` x121
* **num_prototypes**: `(not recorded)` x160, `1000` x121
* **heads**: `(not recorded)` x160, `8` x121
* **prompt_mode**: `(not recorded)` x160, `static` x121
* **prompt_stats**: `(not recorded)` x10, `normalized` x15, `normalized (pre-flag)` x196, `raw` x60
* **lr**: `0.0001` x271, `0.005` x10
* **batch_size**: `32` x281
* **epochs_run**: `10` x273, `4` x1, `6` x2, `7` x1, `8` x3, `9` x1
* **space_trained_in**: `(not recorded)` x150, `raw` x121, `std` x10

## Known gaps

* Time-LLM (GPT-2), 4 channels, split `tslib`, mode `all_channel`: seed 0 only (the matching literal cells carry 3-5 seeds)
* Time-LLM (GPT-2), 4 channels, split `tslib`, mode `chan_indep`: seed 0 only (the matching literal cells carry 3-5 seeds)
* Time-LLM (GPT-2), 25 channels, split `tslib`, mode `chan_indep`: seed 0 only (the matching literal cells carry 3-5 seeds)
* Time-LLM (GPT-2), 151 channels, split `tslib`, mode `all_channel`: **not run**
* Time-LLM (GPT-2), 151 channels, split `tslib`, mode `chan_indep`: **not run**
* Time-LLM (GPT-2), 151 channels, split `literal`, mode `chan_indep`: **not run**

3 cells of the published-comparison grid remain unrun; the seed-0-only cells above need seeds 1 and 2 to match the three-seed convention of the literal campaign.


## Caveats

* **Checkpoint survival is partial.** Of 331 runs: **70 still have their weights** (12.2 GB, sha-verified in `CHECKPOINTS.tsv`); **26 were overwritten** by a later run that reused the same chunk id; **176 had their weights deleted** after being recorded (the first campaign's 150, re-evaluated bit-exact in `reeval_runs.json` before deletion, hashes in `checkpoints_manifest.tsv`; the 40 OFA/BERT checkpoints, hashes in `results_ofa_bert.json`); **59 never saved weights** (DLinear, protocol sweep, split probe, superseded campaign). `results_master.csv` marks this per row in `checkpoint_on_disk`.
* **Chunk ids encode neither the split nor the capacity**, so later runs silently overwrote earlier weights at the same paths: `res4_all_match` (d64/ff128) replaced `res4_all_dm32` (d32/ff32); the split study's tslib K=12 run replaced its own literal K=12 run; and all 15 surviving files reused first-campaign basenames whose weights had been deleted. Recorded metrics remain valid (computed before each overwrite); the overwritten weights cannot be re-evaluated. `CHECKPOINTS.tsv` names the true owner of every file (`owner_source`) and the first-campaign hash it replaced (`first_campaign_sha256_16`).
* `prompt_stats` differs across campaigns and is now a column in `results_master.csv`: split-study / `res4` / first-campaign runs used the pre-fix `normalized` convention (recorded or inferred, marked as such); the 2026-11-04 `tslib` all-channel campaign is the first to use the settled `raw` default. Cross-campaign comparisons carry that variable.
* `smoke_tslib.json` (1-epoch smoke test of the tslib split) was deleted in the 2026-11-04 consolidation; every configuration it covered has a full 10-epoch run.
* Numbers here are the **recorded** test values, consolidated and cross-checked against the source JSONs (`build_master.py` reconciles every run against its source file). For the surviving checkpoints they are additionally re-verified by reloading the weights -- see the next bullet.
* **Checkpoint re-evaluation** (`verify_checkpoints.py`, per-file detail in `checkpoint_verification.json`): 70 surviving checkpoints re-scored from their weights -- **46 reproduce their recorded test MSE exactly** (at the evaluation batch size the run used), 23 within 1%, 1 within ~1.4% but at no tried batch size (32/16/8/4/2/1). The residuals sit on the split-study N=25 channel-independent runs: their weights reload cleanly (parameter key counts, normalisation statistics and test window counts all match), so the drift is consistent with those runs having executed on different hardware, not with a data or record error. Recorded values stand.
