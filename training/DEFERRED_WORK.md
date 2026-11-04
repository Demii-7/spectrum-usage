# Deferred work

Work that is deliberately not done yet, and the reason. Each item has to be run or explicitly dropped with
a stated reason before the paper results are final; none of it should quietly disappear.

Lease ends: H100 cluster 2026-11-07 16:58 UTC; A100 cluster 2026-11-10 18:00 UTC. Everything must be in the
durable bucket (`s3://spectrum`) before then (target: A100 results exported by 2026-11-10 12:00 UTC).

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
benchmark on an idle A100 (one model at a time, same batch size, warm-up, median) for every exported
checkpoint, so the cost table has comparable numbers. Not written yet.

## Horizon-specific models (15 and 60 minutes)
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
