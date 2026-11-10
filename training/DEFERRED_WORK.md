# Deferred work

Work that is deliberately not done yet, and the reason. Each item has to be run or explicitly dropped with
a stated reason before the paper results are final; none of it should quietly disappear.

Lease ends: H100 cluster 2026-11-07 16:58 UTC; A100 cluster 2026-11-10 18:00 UTC. Everything must be in the
durable bucket (`s3://spectrum`) before then (target: A100 results exported by 2026-11-10 12:00 UTC).

## Weekend plan (user offline, from Nov 6)
The weekend to-do list, as set by the user on 2026-11-05 (do not add to it yourself):
1. **Priority 1: the remaining DSwinLSTM round-2 seeds** (seeds 40-44, 0.5 GPU each; the 16-candidate search finished
   2026-11-05 02:55 UTC and the replicate stage is PENDING on the H100 cluster).
2. **Priority 2: the models missing on 2400-2600 MHz, and only those** (not extra seeds for the models already
   there): ConvLSTM and ResidualConvLSTM (4D), VanillaLSTM and ResidualVanillaLSTM (1D). Run them at seed 42 with the
   600-800 hyperparameters, to match the other 2400-2600 runs (assumption: confirm with the user if a different
   seed count is wanted). DSwinLSTM on this band uses the round-2 winner only if it has exported, otherwise
   round 1, and the paper says which.

3. **Priority 3: extra seeds (40, 41, 43, 44) on 2400-2600 for the non-DSwinLSTM models** (11 models; DSwinLSTM extra
   seeds are not included, they cost about 27 GPU-h each).
   - CPU-only linear models (LinearAR and ResidualLinearAR in 1D/2D/4D): launched 2026-11-05 as
     `claude-band2400-linear-seeds-1` on the A100 cluster (24 runs, CPU only, so they do not compete for GPUs).
   - GPU models (AutoformerCSA, LSTMAttn, VanillaLSTM, ResidualVanillaLSTM, TCN in 2D; about 4.2 GPU-h per seed,
     17 GPU-h in total): launch only after priority 1 and 2 are submitted, because Ray does not prioritize and
     these would otherwise take GPU slots first.
   Hyperparameters inherited from 600-800, same settings as the existing seed-42 runs.

Do **not** pause or stop running jobs to make room (user decision). The other open items (scoring the 1D LSTMs and
the 4D ConvLSTM models at stride 1, rescoring per-receiver/pooled-6 and 2400-2600, seeds 40-44 on 2400-2600,
horizon-specific models) stay on the open list below but are not part of the weekend plan.

**Status at 2026-11-06 19:20 UTC** (checked against the bucket, `oct2026-band-runs/` and `checkpoints-oct2026-r2/`; every item below has its weights and `run_costs.json` in `s3://spectrum`):
- **Priority 1, DSwinLSTM round-2 seeds 40-44: done**, exported to `checkpoints-oct2026-r2/` (5 weight files). Final-range scoring is part of `claude-final-eval-v3-4d-2` (A100, still running).
- **Priority 2, the four missing 2400-2600 models (ConvLSTM, ResidualConvLSTM, 1D VanillaLSTM, 1D ResidualVanillaLSTM) at seed 42: done** (`claude-band2400-missing-1`).
- **Priority 3, extra seeds (40, 41, 43, 44) on 2400-2600:** the CPU-only linear models are **done** (all 5 seeds for LinearAR/ResidualLinearAR in 1D, 2D, 4D). The GPU models are **partial**: present are AutoformerCSA 40/41/42/44, VanillaLSTM 2D 40-43, ResidualVanillaLSTM 2D 41/42/44, LSTMAttn and TCN only seed 42. Resubmitted on the H100 cluster as `claude-h100-band2400-gpu-seeds-1` (all 20 runs; about 7 duplicate seeds that already exist and will be overwritten by an equivalent retrain). Missing only: AutoformerCSA 43; LSTMAttn 40/41/43/44; TCN 40/41/43/44; VanillaLSTM 2D 44; ResidualVanillaLSTM 2D 40/43.
- Not in the plan and not run: DSwinLSTM extra seeds on 2400-2600 (43 and 44 exist from an earlier job; 40 and 41 do not).

**Timing risk:** the H100 lease ends Friday 2026-11-07 16:58 UTC, before the weekend, and the DSwinLSTM replicate is
still PENDING there (its GPUs are fully claimed). Anything not started on the H100 cluster by Friday must be
restarted on the A100 cluster (lease ends 2026-11-10 18:00 UTC) from the search results, which are in the durable
backup `oct2026-minio-backup-newcluster/` (the `dswinlstm_i-search` experiment, last backed up 2026-11-05 03:48 UTC).
Everything must be in the bucket before the leases end (H100 2026-11-07 16:58 UTC, A100 2026-11-10 18:00 UTC).

## Models missing from the 2400-2600 MHz band run
One seed per model on 2400-2600 was queued with the inherited 600-800 hyperparameters, but these models were
left out because their searches have not exported a winner yet:
- ConvLSTM (4D), ResidualConvLSTM (4D)
- VanillaLSTM (1D), ResidualVanillaLSTM (1D)
- DSwinLSTM: the queued run uses the round-1 winner. If the round-2 search produces a better configuration,
  rerun it on 2400-2600 with that winner (or state the mismatch).

Waiting on: H100 searches (expected exported about 2026-11-05).

## Evaluation of late checkpoints on the final test range
Once exported, score on the full final range (2026-10-27T20:00Z to 2026-11-03T20:00Z, stride 1) with
`final_eval`: round-2 DSwinLSTM, ConvLSTM, ResidualConvLSTM, 1D VanillaLSTM, 1D ResidualVanillaLSTM.

## Stride-1 rescoring for consistency
The 2400-2600, per-receiver and six-receiver runs score the test range at each config's selected stride, while
the 600-800 final evaluation uses stride 1. Rescore every checkpoint from those runs at stride 1 with
`final_eval` so all reported numbers use the same protocol.

## H100 per-receiver and six-receiver models
Trained with the preliminary test cutoff (2026-10-30T18:00Z). Rescore the checkpoints on the final range, and
compute the pooled-vs-per-receiver comparison on identical test windows (`final_eval --test-sites` for the
pooled models). Outputs: `oct2026-per-receiver/`, `oct2026-pooled6/`.

## Inference benchmark
Existing inference timings come from training runs on shared (fractional) GPUs, mostly H100. Run a clean
benchmark (one model at a time on an otherwise idle GPU, same batch size, warm-up, median) for every exported
checkpoint, so the cost table has comparable numbers. Plan: do it on the H100 cluster once its campaign has
drained (expected about Nov 5-6, lease ends Nov 7 16:58 UTC), since the H100s are the main training GPU type
for the cost table. Not written yet.

## Reporting decision: rolled-out and direct models (user, 2026-11-05)
Report **both** rolled-out (one-step model fed back) and direct (trained for the horizon) results, in the 15-minute
POWDER experiments (and ElectroSense, where both already exist). Status:
- 15-minute POWDER direct: LSTM family, AutoformerCSA and 1D LSTM done (237 records); the four linear models
  (LinearAR/ResidualLinearAR, 1D and 2D) launched 2026-11-05 as `claude-p15-direct-linear-1` (CPU, 160 runs).
  TCN has no direct form, so it is reported rolled-out only (say so in the paper).
- 1-minute POWDER: report rolled-out models only (user decision, 2026-11-05). No direct 15/60-minute models are
  needed there; the paper should state that the 1-minute models are one-step models rolled out.
- Direct models reuse the one-step model's hyperparameters (no separate search); say so.

## Horizon-specific models (15 and 60 minutes) - NOT NEEDED for 1-minute results (user, 2026-11-05)
Train separate models directly for the 15- and 60-minute horizons (current models are one-step and rolled
out autoregressively, except Autoformer-CSA and DSwinLSTM-I, which output 60 steps). Reuse each model's
selected hyperparameters; no new search. `fixed_train --horizon 15 60` now does this (code `c626a57`, needs
syncing); linear and reference models are skipped because they are one-step by design (they stay rolled out).
Start with 2D and 1D, add 4D only if a trend appears.

## TimeRAN (and MOMENT) experiments
The paper's TimeRAN results (1D, per receiver-frequency series; head-only and full fine-tuning; plus the MOMENT
backbone without TimeRAN pretraining) were run on the old data, and there are no checkpoints in `checkpoints/`.
Redo on the October split. Open points:
- The pretrained weights (`TimeRAN_small.pth`, downloaded with `gdown`) must be present on the cluster; the
  repo does not ship them. Not yet checked on the A100 workers.
- TimeRAN is excluded from the Ray tuning registry, so it has no search. The other models got a 16-trial
  search; if TimeRAN is run with fixed settings that disparity must be stated or addressed.
- The existing configs (`timeran_powder_one_seed.yaml`, `moment_powder_one_seed.yaml`) use June data and
  need re-pointing; their transfer settings (head-only vs full fine-tune) need the same level-handling
  treatment as the rest of the transfer re-run.

## TimesFM-3 (zero-shot, used as a 2D model)
Frozen Google TimesFM-3 (330M, weights under the TimesFM Non-Commercial License v1.0; code Apache 2.0), prediction
only, no training and no search. Preliminary test on the A100 cluster (150 windows, humanities, CPU) showed it
runs and beat a last-60-minute mean by about 8% at 1 min, 6% at 15 min and 4-5% at 60 min when forecasting all
200 bins in one call. Not yet comparable to our results: different windows, quick baselines, one receiver.
To do:
- Run it through the real evaluation (same test windows, all metrics, per-bin files, per-receiver scoring) so it
  can be compared with LinearAR and LookbackMean on identical windows, with window-level confidence intervals.
- Choose the channel grouping on the validation range, then freeze it before scoring the test range: all 200
  bins in one call (the model accepted 200 targets, though it is reportedly trained for up to 32), contiguous
  32-bin blocks, or overlapping blocks (50% overlap, step 16, center-weighted or plain averaging).
- Same 60-minute lookback as every other model; a longer-context variant, if run, is a separate labelled result.
- Weights (`google/timesfm-3.0-pytorch`, 1.32 GB, sha256 a7592b0a...f17da) are only on one worker (`/tmp/tfm3`);
  copy to the bucket and all workers. Package: `timesfm[torch]`, imported as `timesfm3`.
- Needs a GPU slot for the full test range; CPU is about 0.7 s per 200-bin window.
- State the non-commercial license in the paper.

## MOMENT backbone (not TimeRAN-pretrained)
Loaded from Hugging Face (`AutonLab/MOMENT-1-{small,base,large}`) via `momentfm`; the A100 workers can reach
Hugging Face. The pretrained weights have no forecasting head, so true zero-shot forecasting is not built in.
Feasibility job: `claude-moment-1`.

## Spectral-context ablation (full context vs region only)
Rerun on the new data with each 2D model's selected hyperparameters: 13 non-noise regions x 2 conditions x
5 seeds per model (LinearAR is unchanged by construction). Code: `training/ray/spectral_context_fixed.py`.
Region boundaries and the noise-floor calibration ranges are reused from the 600-800 plan
(`evaluation/analysis/plan_regions_600_800.csv`); confirm they still hold on the new runs.

## Spatial ablation
Evaluation-only on frozen 4D checkpoints (`training/ray/spatial_ablation.py`). Waiting on exported 4D
checkpoints (ConvLSTM, ResidualConvLSTM, round-2 DSwinLSTM) and on the decision about the new 4/5/6-node
sets.

## Queued, waiting on other jobs
- DSwinLSTM round-1 seeds 43 and 44 and LookbackMean 4D on 600-800: queued on the A100 cluster, starts after
  the 4D final evaluation finishes (job `claude-a100-dswin-seeds-1`).

## Data and records
- The A100 cluster holds the final data (run 2 ends 20:05-20:13 UTC on 2026-11-03). Record file checksums for
  the data used in the final evaluation.
- Final data was not synced into a separate directory on the H100 cluster; its run-2 files are a snapshot
  from 2026-11-03 18:53-19:00 UTC, so H100 results use the preliminary test cutoff only.
