# Campaign status and findings

**Latest state: section 39.30 (2026-11-12 02:50 UTC: strong models, parallelism, TCN seed 42); before that section 39 (2026-11-11 15:17 UTC, 39.10: status, ETAs, TSS tuning, ElectroSense table); section 39 earlier parts (2026-11-11 14:09 UTC: results processed: TSS-LCD, 1-minute table, transfer regimes, kriging); section 38 (2026-11-11 03:37 UTC: all forks finished, queues, open decisions); section 37 (2026-11-11 03:03 UTC: overnight setup, H100 rebalancing, queues, missing code); section 36 (2026-11-11 01:43 UTC: transfer coverage and jobs, TSS-LCD 2400-2600 and tuning); section 35 (2026-11-11 01:10 UTC: cluster status, Time-LLM finals relaunched, TimeRAN facts, open items); section 34 (01:30 UTC: Part 1 per-horizon 15-minute table, tuned vs inherited, latest numbers); section 33 (2026-11-10 23:45 UTC: TSS-LCD 15-minute and causal runs, STS variants, external-transfer and neighbour analyses); section 32 (23:10 UTC: cluster status and Part 1/2 readiness); section 31 (20:10 UTC); section 30 (18:25 UTC); sections 26-29 hold the earlier updates (26 and 27: the 13:20 update and the transfer-evaluation design; 28: 16:20; 29: 17:30). `OVERVIEW.md` maps the paper's six questions to the evidence.**

Snapshot: **2026-11-08, about 02:30 UTC** (section 22 holds the detailed weekend results, analyzed at that time; section 21 the new direct-model tuning; sections 10, 12, 14.1, 15.1, 18, 19, 20 were refreshed earlier on 11-07/11-08) (the H100 lease ended 2026-11-07 16:58 UTC; sections 10, 12, 14.1, 15.1, 18, 19 and 20 were refreshed from the bucket and the A100 cluster at 23:30 UTC on 11-07, the rest is as of 2026-11-06 19:40 UTC). One section per campaign. Each section has a table of what is done and what is not, then the findings so far.
Findings are provisional until the paper's final runs are in; items I did not re-verify for this snapshot are marked *(unverified)*. Planned-but-not-run work is in each
section and collected again in section 17; risks and open decisions are in section 20.

**Overnight 11-06 to 11-07 (now analyzed in section 22):** all H100 jobs finished before the lease ended: 15-minute pooled models rerun with weights (514 of 514 runs, 514 weight files), OFA/Time-LLM tuning on ElectroSense (228 records, none failed, 100 finals with weights), TSS-LCD full schedule on POWDER (10 of 10 runs, 30 weight files), the 2400-2600 extra seeds for the five GPU models (now seeds 40-44 for all of them). On the A100: the 4D final evaluation, the per-receiver and pooled comparison, the 12 OFA 15-minute runs (no weights) and the tuned 15-minute finals for every model except the 1D LSTMs finished; the 1D LSTM finals are running after the automatic restart (section 18).

**What changed since the 15:30 snapshot of 11-06:** the savgol-off TSS-LCD retrains finished and were rerun with weights saved (15.5); all 16 BERT reruns finished (14.3); the
tuned Informer/Autoformer/LightTS baselines finished (13.2); an equal-budget tuning run for OFA and Time-LLM was written and queued (14.1); checkpoint saving was added
to the runners (19); an inference benchmark was written (10); the weekend plan was checked against the bucket (17); and the H100 cluster was loaded with enough queued
work to stay busy until its lease ends (18).

## 0. Conventions and infrastructure

| item | value |
|---|---|
| POWDER 1-minute metric | MAE in dB, mean over horizons 1/15/60 min, pooled over bins and receivers (test = 2026-10-27T20:00Z to 2026-11-03T20:00Z) |
| POWDER 15-minute metric | MAE in dB, mean over horizons 4/12/24/48 steps (1-12 h), lookback 96 steps |
| ElectroSense metric | **raw-unit MSE** (the data's own units), mean over horizons 12/24/48/96/192, lookback 192; the SpectrumLLM paper reports raw MSE, not standardized |
| Seeds | 40-44 for final results (tuning/search uses seed 42) |
| Tuning protocol | equal budget: 16 candidates per (dataset, model), selected on validation, then 5 seeds on test; test data never used for selection |
| Rolled-out vs direct | one-step model fed back (rolled out) vs model trained for the horizon (direct). Reporting decision: report both at 15 minutes and on ElectroSense; **1-minute is rolled-out only** |
| Clusters | H100 "mixed-gpu", **rebuilt 2026-11-07** (Ray dashboard tunnel :8266 via `training/ray/docker/h100-ssh.conf`, 6 hosts, 6 GPU units; the four `ray-new-*` hosts end **2026-11-09 13:40 UTC**, the two old hosts (`ray-worker-1`, head `ray-worker-2`) end 2026-11-10 18:00 UTC; the old cluster expired 2026-11-07 16:58 UTC). A100 "tacc" (tunnel :8267, 4 GPU units, 1024 CPUs): lease ends **2026-11-10 18:00 UTC** (the user gave "6 p.m." with no timezone; if that is US Eastern it is 23:00 UTC; planning uses 18:00 UTC). Everything must be in the durable bucket `s3://spectrum` before then |
| Sync | `training/ray/scripts/sync-cluster-code.py --revision <commit>` (syncs committed code to both clusters) |
| Caveat | Ray gives no job priority, so queue order is effectively random. Both clusters' GPUs are fully claimed; CPU is mostly idle |

---

## 1. POWDER 1-minute, 600-800 MHz, four core receivers (October campaign)

Receivers: humanities, madsen, moran, sagepoint (pooled into one model per architecture). Train 2026-10-15 to 10-22, validation
10-25 to 10-27, final test 10-27 to 11-03. Lookback 60 min, one-step models rolled out to 15 and 60 min (AutoformerCSA and
DSwinLSTM-I output 60 steps).

### 1.1 Hyperparameter tuning (October search, 16 candidates per model)

| model (representation) | search | state |
|---|---|---|
| LookbackMean 1D/2D/4D | none (parameter-free) | done |
| LinearAR, ResidualLinearAR (1D, 2D, 4D) | 16 candidates | done, exported, 5 seeds |
| VanillaLSTM, ResidualVanillaLSTM, LSTMAttn (2D) | round 1 and round 2 | done (round-2 checkpoints exist for all three) |
| AutoformerCSA, TCN (2D) | round 1 | done |
| VanillaLSTM 1D, ResidualVanillaLSTM 1D | round 1 | done (campaign finished 11-02 to 11-04), 5 seeds trained |
| ConvLSTM, ResidualConvLSTM (4D) | round 1 | done, 5 seeds trained |
| DSwinLSTM-I (4D) | round 1 (3 seeds exported) and **round 2** | round 2 search finished 11-05 02:55 UTC; all five final-stage seeds finished and were **exported on 11-06 at 14:07 UTC** (`checkpoints-oct2026-r2/4d/dswinlstm_i/`); not yet scored on the final range |
| TimeRAN / MOMENT | **no search** (excluded from the tuning registry) | not run on October data |

### 1.2 Seed runs and final evaluation (stride 1, final test range)

Test MAE in dB, mean over 5 seeds; lookback mean is the reference.

| model | scored? | MAE (dB) | vs lookback mean (0.950) |
|---|---|---|---|
| LookbackMean 1D / 2D | yes | 0.950 / 0.949 | 0 |
| ResidualLinearAR 2D | yes | 0.924 | -2.7% |
| ResidualLinearAR 1D | yes | 0.926 | -2.5% |
| LinearAR 1D / 2D | yes | 0.928 / 0.930 | -2.3% / -2.1% |
| TCN 2D | yes | 0.930 | -2.1% |
| ResidualVanillaLSTM 2D (round 2 / round 1) | yes | 0.944 / 1.056 | -0.6% / +11% |
| AutoformerCSA 2D | yes | 0.956 | +0.7% |
| LSTMAttn 2D (round 1 / round 2) | yes | 1.084 / 1.094 | +14% / +15% |
| VanillaLSTM 2D (round 1 / round 2) | yes | 1.103 / 1.128 | +16% / +19% |
| VanillaLSTM 1D, ResidualVanillaLSTM 1D | **no** (checkpoints exist, not scored at stride 1) | - | - |

### 1.3 4D models and DSwinLSTM

| item | state |
|---|---|
| LinearAR 4D, ResidualLinearAR 4D | 5 seeds trained; final evaluation partly done (5 and 3 seeds scored) |
| ConvLSTM, ResidualConvLSTM | 5 seeds trained; **still not scored on the final range** (the 4D evaluation job's model list never included them) |
| DSwinLSTM-I round 1 | 3 seeds trained and scored; the job `claude-a100-dswin-seeds-1` (seeds 43-44 and LookbackMean 4D) shows SUCCEEDED; its outputs were not re-verified *(unverified)* |
| DSwinLSTM-I round 2 | trained and exported (11-06 14:07 UTC); **final-range scoring not done** (the 4D job started before the export and scored only round 1) |
| 4D final evaluation job `claude-final-eval-v3-4d-2` | **finished 11-07** (13 rows, all ok): ResidualLinearAR 4D 0.719 dB, LinearAR 4D 0.728, DSwinLSTM-I round 1 (3 seeds) 0.918; see 22.5 for what the 4D numbers mean |

### 1.4 Per-receiver and pooled-6 models

| item | state |
|---|---|
| Per-receiver models (2D, one model per receiver, six receivers: the four core + law73 + guesthouse) | trained (`oct2026-per-receiver/`), preliminary test cutoff 2026-10-30T18:00Z, each at its selected stride |
| Pooled-6 models (2D, one model on all six) | trained (`oct2026-pooled6/`), seed 42, same caveat |
| Both versions (4-site and 6-site training) | kept |
| Rescoring on the final range at stride 1, identical test windows for the pooled-vs-per-receiver comparison | **not done** |

### 1.5 Findings so far (1 minute)

- **Flat.** The best learned model (ResidualLinearAR 2D) is only 2.7% better than the lookback mean. Most neural models are no
  better, and the plain LSTMs are 14-19% worse.
- **A level pathway matters.** Models that predict a correction to the lookback mean (the residual family, plus TCN and
  AutoformerCSA, which have a separate level path) stay near the baseline; plain LSTM/LSTMAttn fall far behind. Round-2
  search recovered ResidualLSTM (1.056 to 0.944) but not LSTMAttn or VanillaLSTM.
- **Linear beats neural** here, and the residual linear model is the most stable under distribution shift (see section 2).
- Headroom (gap between causal mean and a two-sided oracle) is small at one minute, which is why little helps; see sections 3 and 4.

---

## 2. POWDER 1-minute, 2400-2600 MHz

Same four receivers and protocol, but one seed per model with the **600-800 hyperparameters inherited** (no band-specific
search), scored at each model's selected stride (not the stride-1 final protocol).

| model | seed 42 | seeds 40, 41, 43, 44 | note |
|---|---|---|---|
| LookbackMean 1D/2D/4D | done | n/a | |
| LinearAR, ResidualLinearAR (1D/2D/4D) | done | **done** (`claude-band2400-linear-seeds-1`, 24 runs; all 5 seeds present) | means over 5 seeds not yet tabulated |
| VanillaLSTM, ResidualVanillaLSTM, LSTMAttn, AutoformerCSA, TCN (2D) | done | **partial** (checked against the bucket 19:20 UTC): AutoformerCSA 40/41/44, VanillaLSTM 40/41/43, ResidualVanillaLSTM 41/44 present; LSTMAttn and TCN none | all 20 runs resubmitted on the H100 cluster as `claude-h100-band2400-gpu-seeds-1` (about 7 are duplicates that overwrite equivalent runs); 13 seeds were truly missing |
| ConvLSTM, ResidualConvLSTM (4D) | **done** (11-06) | not done | were missing; run as `claude-band2400-missing-1` |
| VanillaLSTM 1D, ResidualVanillaLSTM 1D | **done** (11-06) | not done | were missing |
| DSwinLSTM-I | done with the round-1 winner | seeds 43 and 44 present (from an earlier job, round-1 winner), 40 and 41 not run | the round-2 winner is exported; a rerun on this band with it is not planned (about 27 GPU-h per seed), so the paper must state the mismatch |
| Stride-1 rescoring of the band | - | - | **not done** |

**Comparability warning (found 2026-11-08, from the job commands; not yet re-scored to confirm).** `fixed_train` scores to a default test end of **2026-10-30T18:00Z** unless `--test-end` is given. The seed-42 job (`claude-a100-band2400-s42-1`), the A100 seeds-43-44 job and the H100 GPU-seed job were launched with `--test-end 2026-11-03T20:00Z`; the **extra linear seeds (40, 41, 43, 44; `claude-band2400-linear-seeds-1`) and the four "missing" models (ConvLSTM, ResidualConvLSTM, 1D VanillaLSTM, 1D ResidualVanillaLSTM; `claude-band2400-missing-1`) were launched without it**, so they were scored on the shorter, earlier range. The numbers agree: ResidualLinearAR 2D is 0.744 at seed 42 and 0.776 at seeds 40/41/43/44 with a seed spread of 0.000. Tables that mix these runs compare different test windows. Fix: rescore every 2400-2600 checkpoint on one range at stride 1 (see 22.5).

Seed-42 test MAE (dB), lookback mean 0.639 (4D) / 0.768 (2D) / 0.772 (1D):

| model | MAE | model | MAE |
|---|---|---|---|
| LinearAR 4D | 0.610 | TCN 2D | 0.740 |
| ResidualLinearAR 4D | 0.614 | ResidualLinearAR 2D | 0.744 |
| ResidualConvLSTM 4D | 0.647 | ResidualLinearAR 1D | 0.749 |
| ConvLSTM 4D | 0.676 | ResidualVanillaLSTM 1D | 0.775 |
| DSwinLSTM-I 4D (round 1) | 0.696 | VanillaLSTM 1D | 0.796 |
| LSTMAttn 2D | 0.790 | VanillaLSTM 2D | 0.795 |

**Findings.** As flat as 600-800: the best learned model is 3-5% better than the lookback mean, ResidualConvLSTM
(0.647) is level with the mean, and the plain recurrent and attention models are worse. Single-seed differences of a few
percent are within noise. In the transfer evaluations (paper draft) the residual linear model and the lookback mean
generalize best and several neural models fail by more than 20 dB.

---

## 3. Forecastability atlas (headroom across sites, bands and datasets)

How much do spectrum measurements differ across sites and bands, and what does that mean for forecastability? Measures:
headroom (causal lookback mean vs a two-sided oracle), Fano/entropy predictability, leave-one-day-out daily-profile R2,
and a realized-gain proxy (per-bin ridge AR on residuals). A quality rule keeps only runs with at least 48 h of data and at
most 20% missing.

| item | state |
|---|---|
| Atlas script (`evaluation/analysis/forecastability_atlas.py`, 6 unit tests) | done |
| Atlas run over the runs mentioned in the paper draft, the October POWDER data and the other testbeds | done; outputs in the bucket under `oct2026-results/atlas/` |
| ElectroSense traces compiled for the atlas and uploaded to the bucket | done; the atlas run on them is *(unverified)* |
| Cross-run stability and cross-site analyses | requested; not re-verified for this snapshot *(unverified)* |
| A figure linking atlas headroom / realized gain to the model results | **not done** |
| Core-band decision (600-800 vs 2400-2600) | discussed; no written conclusion |

**Findings so far.**
- Headroom is small on 1-minute POWDER and large on 15-minute POWDER and ElectroSense; the model results in sections 1, 2,
  12 and 13 follow it (about 3%, 35% and 45-65% below the lookback mean).
- From the paper draft (older June/July data): the power-difference entropy of a frequency region is strongly associated
  with forecast error (rho = 0.77), but higher entropy does not give the neural models a larger advantage over the
  lookback mean.
- The numbers themselves live in the atlas outputs and are not reproduced here.

## 4. Level diagnostics (why learned corrections barely beat the baseline at 1 minute)

CPU-only work from a separate session, started 2026-11-06 (plan file `~/.claude/plans/come-up-with-a-streamed-coral.md`, memory note `project_level-diagnostics`). **The facts in this section come
from that session's notes and its committed writeup; I did not re-read the outputs for this document** *(unverified)*.

| item | state |
|---|---|
| Code `training/ray/level_diagnostics.py` (commit `a54a1c0`, `031af6c` on `integrate`) | **local only, not pushed**; synced to the mixed-gpu cluster with `--revision HEAD`, so a later sync from `origin/integrate` would remove it |
| Smoke job, full job (`claude-leveldiag-full-1`, 4 receivers x 2 bands) and sampling job (`claude-leveldiag-sampling-1`) | done; outputs copied to `runs/claude/level-diagnostics/{full-1,sampling-1}/` |
| Writeup with results, derived tables and scripts | committed (`5c3d766`): `evaluation/results/level_diagnostics/README.md` |
| ElectroSense aggregation check | done: the API serves only aggregated data (aggTime/aggFreq/aggFun AVG or MAX, non-owner floor 60 s / 100 kHz); the SpectrumLLM files' noise statistics match POWDER 15-minute averages, not samples (authors unconfirmed) |
| Open | real-model checkpoint audit, GBM confidence intervals vs the strong baseline, ElectroSense strong-baseline rerun, entropy diagnostic (primary citation Ding et al. 2015) |

**Findings (from the other session).** Local-level fits (first difference rho1 about -0.48, rho2/3 about 0); the best simple predictor is a 15-30 minute median; MSE-ridge corrections are at best equal to it; an MAE-loss
gradient-boosting probe is 2-10% better than the best simple predictor (no confidence interval yet), which points to a loss/metric mismatch (MSE training, MAE scoring). With validation-selected baselines the averaged-target gain
of ridge is only 0-3% (an earlier 8-14% was against a mismatched dB-mean baseline). Averaging beats decimation for averaged targets; for point targets the 1-minute history beats both. White noise is about 66-85% of the best
point-target MSE; the random-walk floor is invalid beyond about 15 minutes.

## 5. Spectral-context ablation (full spectrum context vs region only)

Does giving a model the whole spectrum help, compared with only the region being predicted? Rerun on the new data with each 2D
model's selected hyperparameters: 13 non-noise regions x 2 conditions x 5 seeds per model (LinearAR is unchanged by
construction). Code: `training/ray/spectral_context_fixed.py`; region boundaries and noise-floor ranges reused from
`evaluation/analysis/plan_regions_600_800.csv`.

| item | state |
|---|---|
| Ablation cells, 1 minute | **restarted 2026-11-08 ~14:30 UTC** (166 of 650 done at 23:30; per-cell median 31-39 minutes; resubmitted with a cap on cells in flight as `claude-spectral-1min-3`) at the user's request on the A100 cluster (`claude-spectral-1min-2`, resumes from the 121 finished cells; 529 left; median 41 minutes per cell in the first batch, so about 360 task-hours) |
| Ablation cells, **15 minutes** (new; same 13 regions x 2 conditions x 5 seeds x 5 models = 650 cells, each model's inherited 1-minute 600-800 configuration, lookback 96, horizons 4/12/24/48 rolled out; `--resolution 15min`) | **started 2026-11-08 ~14:30 UTC** (output `oct2026-spectral-context-15min/`; 72 of 650 done at 23:30, none failed, so the 15-minute path works; about 2 minutes per cell, so about 20 task-hours for the rest; resubmitted with a cap as `claude-spectral-15min-2`). With both running, my rough estimate is 30-35 hours on the 16 A100 GPU slots, which fits the lease (ends 2026-11-10 18:00 UTC) with little slack; some cells may be cut |
| (superseded) Ablation cells | paused at 121 of 650 (529 left) at the user's request, to free the clusters for ElectroSense and 15-minute POWDER |
| Control conditions (to show the comparison can detect an effect) | proposed, **not added** |
| Analysis of the 121 completed cells | **not done** |
| Confirmation that the region boundaries still hold on the new runs | not done |
| Earlier (older-data) analysis for ResidualLinearAR 2D | exists: `evaluation/results/tables/spectral_context_residuallinearar2d.csv`; not summarized here |

**Findings so far.** None from the completed cells. The user's conclusion is that "does spectral context help" cannot be
evaluated while the models already saturate without it (see sections 1 and 2: the gain over the lookback mean is only a few
percent); a test that could show an effect needs held-out receivers or hours-long horizons.

## 6. Spatial ablation (4D models, frozen checkpoints)

Evaluation-only study of how much the 4D (spatial) context helps, on frozen 4D checkpoints
(`training/ray/spatial_ablation.py`).

| item | state |
|---|---|
| Code | done |
| Run on the October 4D checkpoints | **not run**: waiting for the exported 4D checkpoints (ConvLSTM, ResidualConvLSTM, round-2 DSwinLSTM) and for the decision on the new 4/5/6-node receiver sets |
| 4D models available | LinearAR 4D, ResidualLinearAR 4D, ConvLSTM, ResidualConvLSTM (5 seeds each, trained); DSwinLSTM round 1 (3 seeds), round 2 training |

**Findings so far.** None. In the 1-minute results the 4D linear models were the best on 2400-2600 (0.610 and 0.614 vs the
lookback mean 0.639), a small gain that the spatial ablation would test.

## 7. Transfer evaluations (new receiver, new bands, new sites)

Freeze the models trained on the core data and score them under shift: a new collection point (guesthouse), new bands
(800-1000, 2400-2600, 3500-3700, 5725-5925 MHz), new sites (COSMOS, ARA, AERPAW) and combined site-and-band shifts.

| item | state |
|---|---|
| Transfer tables in the paper draft | from the **older (June/July) data and models** |
| Re-run on the October models | not confirmed *(unverified)*; the 2400-2600 band and held-out receivers (guesthouse, law73) exist as separate runs (sections 2 and 1.4) |
| Level handling for transfer (the zero-shot level problem) | discussed; applied in the re-run plan *(unverified)* |

**Findings so far (paper draft, older data).** Several neural sequence models fail catastrophically under shift: on the
COSMOS site-and-band condition AttnLSTM, LSTM and TCN reach horizon-averaged MAEs above 20 dB, against under 0.1 dB for the
lookback mean and ResidualLinearAR; similar failures occur for the 4D ConvLSTM and DSwinLSTM. The residual linear model
generalizes most consistently and the lookback mean is nearly as strong. Models without a persistence or residual pathway are
the ones that fail.

## 8. TimeRAN and MOMENT (foundation-model transfer)

| item | state |
|---|---|
| TimeRAN (1D, per receiver-frequency series; head-only and full fine-tuning) on the October data | **not run**; the paper's results are from the old data and no checkpoints exist |
| MOMENT backbone without TimeRAN pretraining | **not run**; feasibility job `claude-moment-1` was planned |
| Pretrained weights on the clusters | `TimeRAN_small.pth` (downloaded with `gdown`) not checked on the A100 workers; MOMENT is loaded from Hugging Face (`AutonLab/MOMENT-1-{small,base,large}`) |
| Tuning | none: TimeRAN is excluded from the Ray tuning registry, so if run with fixed settings the disparity (the other models got a 16-trial search) must be stated |
| Configs | `timeran_powder_one_seed.yaml` and `moment_powder_one_seed.yaml` use June data and need re-pointing |

**Findings so far (paper draft, older data).** TimeRAN avoids catastrophic transfer failures (its reversible instance
normalization acts as a level-preserving, residual-like pathway), but its performance is essentially matched by the MOMENT
backbone, which gives little evidence that RAN-specific pretraining helps spectrum forecasting. Not yet re-tested.

## 9. TimesFM-3 (zero-shot, used as a 2D model)

Frozen Google TimesFM-3 (330M, weights under the TimesFM Non-Commercial License v1.0; state this in the paper), prediction
only, no training and no search.

| item | state |
|---|---|
| Preliminary test (A100 cluster, 150 windows, humanities, CPU, all 200 bins in one call) | done |
| Full evaluation on the real test windows with all metrics, per-bin files, per-receiver scoring and window-level confidence intervals | **not run** |
| Channel grouping (all 200 bins; contiguous 32-bin blocks; overlapping blocks), chosen on the validation range and then frozen | not done |
| Longer-context variant | optional, would be a separate labelled result |
| Weights `google/timesfm-3.0-pytorch` (1.32 GB) on all workers | only on one worker (`/tmp/tfm3`); copy to the bucket and the other workers |

**Findings so far (preliminary only).** It beat a last-60-minute mean by about 8% at 1 minute, 6% at 15 minutes and 4-5% at
60 minutes. Not comparable with our results yet: different windows, quick baselines and one receiver.

## 10. Inference-cost benchmark

| item | state |
|---|---|
| Timings from the training runs (shared fractional GPUs, mostly H100) | exist in the per-run cost records; not comparable across models |
| Benchmark script (`training/ray/inference_benchmark.py`, tests in `tests/test_inference_benchmark.py`): one checkpoint per model (seed 42 if present), whole GPU per task, CUDA-synchronized timers around the pipeline's `forecast` (includes the roll-out for one-step models), batch sizes 1 and 32, 3 warm-up calls, median; refuses to run unless the cluster is idle | **written and smoke-tested** on the H100 cluster (shared, so marked `idle=false`; output `inference-benchmark/smoke/`) |
| Extended to all checkpointed POWDER models (11-07): `oct`, `band` (2400-2600), `p15` (15-minute pooled, rolled-out and direct) and `p15tuned` (tuned 15-minute finals), the last two rebuilding their exact config from the weights' `.meta.json`; manifest at 23:30 UTC: **158 models** (20 + 17 + 102 + 19); CPU smoke test on the A100 passed for every source (not a timing run); runbook `training/INFERENCE_BENCHMARK.md` | **ready**, waiting for the new H100 cluster; tests in `tests/test_inference_benchmark.py` (3) |
| **Full benchmark run finished 2026-11-08 00:50 UTC (34 minutes)** on the idle new H100 cluster: job `claude-bench-h100-1`, **159 models, 318 rows, all ok, `idle=True`, all on NVIDIA H100**, output `s3://spectrum/inference-benchmark/h100/inference_benchmark_20261108T0050Z.csv` (results not yet analyzed beyond a first look: at batch 32 the median cost per receiver spectrum is about 0.08 ms for 15-minute 2D models, 0.45 ms for 1-minute 2D models, 8-9 ms for 4D models, and 45 ms for 1D models, which are scored one bin at a time; 4D windows number about 496, i.e. one window per time step). Originally launched 00:22 UTC with 158 models, batch sizes 1 and 32, one GPU per model, output `inference-benchmark/h100/`. Two changes after the smoke test: 1D models are scored one (receiver, bin) series per window (about 800 per time window), so their windows are thinned 20x, and every row has `spectra_per_window` (1D = 1/200, 2D = 1, 4D = 4 receivers, an assumption to confirm from the 4D window counts) and `ms_per_spectrum` so models of different representations are comparable. The tuned 15-minute 1D LSTM finals were still training on the A100 and are **not** in this run; add them with `--sources p15tuned --models vanillalstm1d residualvanillalstm` once they are in the bucket |
| **Caveat on the full run:** the six linear models (LinearAR and ResidualLinearAR in 1D, 2D, 4D) have `device: cpu` in their configs, so the benchmark timed them on the H100 hosts' **CPUs** (their peak GPU memory is 0.0 MB), while every other model ran on the GPU. **Decision (user, 2026-11-08): keep as is and mark these rows with \* in every timing table.** Timings per model: section 22.6 |
| H100 smoke test on the new cluster (11-08, idle GPUs, 8 models, `inference-benchmark/smoke-h100/`) | first rows are real timings: for example the 600-800 2D VanillaLSTM takes 10.3 ms per window at batch 1 and 0.41 ms per window at batch 32 |
| Manifest of checkpoints (`--manifest-only`, October campaign only, 11-06) | **done**: 20 (prefix, representation, model) groups with weights in `checkpoints-oct2026*` (1D 4, 2D 7 + 3 round-2 copies, 4D 5 + DSwinLSTM round 2); the three lookback-mean baselines have no weights (nothing to load) |
| Clean benchmark run on an idle GPU for every exported checkpoint | **not run**; planned for the new H100 cluster next week. Needs: code sync, checkpoint and data sync, an empty cluster (or submit first), one GPU per task |
| Not covered by this script | TSS-LCD (own pipeline), Time-LLM/OFA, TimeRAN, MOMENT, TimesFM-3; 15-minute and ElectroSense models (no weights for runs before commit de1b97e) |
| TSS-LCD cost | 53 s to sample 9,888 test windows on a shared H100 (section 15); diffusion sampling makes it the cost outlier |

**Findings so far.** None.

## 11. Horizon-specific (direct) models at 15 and 60 minutes, 1-minute POWDER

| item | state |
|---|---|
| `fixed_train --horizon 15 60` (code `c626a57`) | built; skips linear and reference models |
| Direct models for the 1-minute results | **not needed**: the 1-minute results are reported rolled-out only (user decision, 2026-11-05) |

**Findings so far.** None; the paper should state that the 1-minute models are one-step models rolled out. Direct models are
reported at 15 minutes (section 12) and on ElectroSense (section 13).

## 12. POWDER 15-minute (block-averaged, lookback 96)

Data: 1-minute traces block-averaged in linear power to 15 minutes (a block needs >= 80% valid minutes), gaps of up to 16
blocks interpolated. Horizons 4/12/24/48 steps. Four core receivers pooled unless stated.

### 12.1 Table

| item | done | not done |
|---|---|---|
| Baselines (lookback mean etc.) | done | |
| Rolled-out models, **inherited** (600-800 1-minute) hyperparameters, 5 seeds, both bands | done: LinearAR, ResidualLinearAR (1D, 2D), VanillaLSTM, ResidualVanillaLSTM, LSTMAttn, AutoformerCSA, TCN (2D), VanillaLSTM 1D, ResidualVanillaLSTM 1D | |
| Direct (horizon-specific) models, inherited hyperparameters | done: LSTM family, AutoformerCSA, 1D LSTM and the four linear models; all rerun with weights saved (22.4) | TCN has no direct form (rolled-out only); **direct vs rolled-out now compared: direct is better for almost every model (22.4)** |
| Per-receiver vs pooled (`sitecmp2`) | partial: per-receiver 213 and pooled 55 records written | **finished 11-07** (260 usable records); compared on identical windows in 22.4 |
| **Tuning** (16 candidates, then 5 seeds) | trials 16/16 for LinearAR, ResidualLinearAR, VanillaLSTM, ResidualVanillaLSTM, LSTMAttn, TCN (2D) and the linear 1D models, both bands; AutoformerCSA 15/16 | **no tuned finals yet for any model**; the 1D LSTM trials are the slow part (`claude-tune-p15-1d-lstm`; past trials of these models took a median of 3.4 h (VanillaLSTM 1D) and 4.3 h (ResidualVanillaLSTM 1D), maximum 10-14 h; at 19:20 UTC 16 were running and 8 pending on the A100 cluster) |
| OFA and Time-LLM (SpectrumLLM paper models) on 15-minute POWDER | runner built, setup done, smoke test passed (see 14) | 12 OFA runs (`claude-llm-ofa-p15-1`) pending on the A100 cluster since 2026-11-05 21:46 UTC, none finished (they wait behind the tuning trials); Time-LLM not launched; these runs predate the weight-saving patch, so they will keep no weights |
| TSS-LCD at 15 minutes | not planned yet | |

**Test-range mismatch between the two 15-minute result sets (found 2026-11-08).** Splits are chronological and do not overlap (train 10-15 14:34 to 10-22 13:13, validation 10-25 13:55 to 10-27 13:55, test from 10-27 20:00; lookback 96 steps), so there is no train-to-test leak.
But the **tuned finals** (`tuning/powder15`, from `tune_fixed`) were scored on test 10-27 20:00 to **10-30 18:00** (the preliminary cutoff, about 3 days), while the **inherited-hyperparameter runs** (`powder15-results*`, from `powder15.py`) use 10-27 20:00 to **11-03 20:00** (7 days). Tuned and inherited numbers are therefore
on different windows and must not be compared directly. Fix (not yet done, user asked about it): rescore the tuned finals' saved checkpoints on the full range with their sidecar configs (evaluation only, no retraining).

### 12.2 Results with inherited hyperparameters (rolled out, test MAE dB, 5 seeds)

| model | 600-800 | 2400-2600 |
|---|---|---|
| LookbackMean | 1.238 | 0.987 |
| ResidualLinearAR 1D | 0.800 | 0.657 |
| LinearAR 1D | 0.800 | 0.664 |
| ResidualLinearAR 2D | 0.801 | 0.681 |
| LinearAR 2D | 0.935 | 0.760 |
| ResidualVanillaLSTM 1D | 0.999 | 0.740 |
| LSTMAttn 2D | 1.036 | 0.916 |
| ResidualVanillaLSTM 2D | 1.139 | 0.950 |
| VanillaLSTM 1D | 1.201 | 0.948 |
| TCN 2D | 1.212 | 1.137 |
| VanillaLSTM 2D | 1.286 | 0.999 |
| AutoformerCSA 2D | 1.307 | 1.210 |

### 12.3 Per-receiver vs pooled (older, incomplete table; the final comparison on identical windows is in 22.4)

| model | 600-800 per-receiver / pooled | 2400-2600 per-receiver / pooled |
|---|---|---|
| ResidualLinearAR 2D | 0.809 / 0.801 | 0.666 / 0.681 |
| LinearAR 2D | 1.018 / 0.935 | 0.866 / 0.760 |
| LSTMAttn | 1.147 / 1.036 | 1.042 / 0.916 |
| ResidualVanillaLSTM | 0.988 / 1.100 | 0.864 / 0.846 |
| VanillaLSTM | 1.303 / 1.290 | 1.209 / 0.999 |

### 12.4 Findings so far (15 minute)

- **There is headroom at this resolution**: the linear and residual models are about 35% better than the lookback mean on both
  bands, against about 3% at one minute.
- **Linear beats neural again**, and neural models with a level pathway (ResidualLSTM 1D) come closest; AutoformerCSA,
  TCN and the plain LSTMs are at or worse than the baseline. These neural numbers are **untuned**; tuning has not finished.
- **Pooled vs per-receiver**: for the residual linear model they are equal within noise (0.801 vs 0.809 and 0.681 vs 0.666).
  For the other models pooling is better in most cells, but the runs are incomplete and the two are not yet scored on identical
  windows, so no conclusion yet.

---

## 13. ElectroSense-Madrid (the SpectrumLLM paper's data; spec4 / spec25 / spec151, 2,872 rows, 15-minute)

Protocol follows the Time-Series-Library loader the paper's baselines use: 70/10/20 chronological split, lookback 192,
horizons 12/24/48/96/192, normalization fit on train; metric is raw MSE (and standardized MSE is also stored). Our models are
trained one-step and rolled out, or directly per horizon.

### 13.1 Table

| item | done | not done |
|---|---|---|
| Trivial baselines (lookback mean, persistence, last value, seasonal naive/mean) | done | |
| Library baselines with **default** hyperparameters (DLinear, PatchTST, Informer, Reformer, Autoformer, LightTS, TimesNet) | 3 seeds each; PatchTST on spec151 and TimesNet are incomplete (missing-package failures, then TimesNet rerun stopped as too slow) | TimesNet (spec4, spec25) rerun; PatchTST default on spec151 |
| Library baselines **tuned** (16 candidates, validation-selected, 5 seeds) | PatchTST and DLinear done, all 3 datasets (150 finals) | Informer, Autoformer, LightTS on CPU (job `claude-tsl3-tune-cpu-2`): **done, all 288 trials and all 225 finals** (results in 13.2); other library models (Reformer, TimesNet) not tuned |
| Our models, inherited hyperparameters, rolled out and direct | done, 5 seeds, all models | |
| Our models, **tuned** (16 candidates, then 5 seeds) | all trials done; finals done for linear models (both representations, all datasets), 1D LSTMs on spec4/spec25, and most 2D LSTM cells | finals pending for AutoformerCSA and TCN (all datasets); LSTM-family finals on spec151 partial and 1D LSTMs on spec151 not started (`claude-tune-es-1` still has 12 final runs going on the H100 cluster) |
| ResidualLinearAR variants (seasonal baseline, window normalization) | done for spec4, spec25; spec151 only 2D seasonal and 2D seasonal+norm | spec151 1D variants and 2D norm |
| Loss-units test (standardized vs raw-unit loss) | done | |
| Collaborator's SpectrumLLM reproduction (Time-LLM, OFA, DLinear) | done, see section 14; **run in the paper's fixed regime, not under our equal-budget tuning protocol** | matched-budget tuning of OFA and Time-LLM: queued (14.1) |

**CORRECTION (2026-11-10 02:10 UTC): the "5 seeds" of the tuned Time-Series-Library baselines are one run.** The finals of the tuned DLinear, PatchTST, Informer, Autoformer and LightTS (seeds 40-44; `electrosense-results/tuned-tsl/`, `tuned-tsl-2/`) are bit-identical across the five seeds (PatchTST spec25 horizon 192: 1.068412 for seeds 40, 41, 42, 43 and 44; DLinear spec4 horizon 12: 0.271111 for all five). The seed passed to the library is not applied: the library seeds itself with a fixed value (the reproduction notes: `fix_seed = 2021`), and the label in the job only names the file. So every statement in this document and in earlier results tables that these baselines are "means over 5 seeds" or have a seed sd is wrong: they are single runs, with no seed variation to report (standard error undefined). The untuned Reformer repeats (seeds 2021-2023, `rerun-a/`) differ by 0.1-0.4%, which looks like GPU non-determinism rather than seeding; the reproduction README measures real seed-to-seed variation of 2.4% sd of the MSE (single cells have swung by 20%), so a one-seed number can be off by that much. **Fix (not done):** make the seed effective (patch `run.py` to read it from the environment or a flag when TSL is set up with `tsl_runs setup`), then rerun the finals with the tuned parameters: 5 models x spec4/spec25 x horizons 12/48/192 x 5 seeds = 150 runs (DLinear and PatchTST about 45 s each, the other three slower on CPU). The Time-LLM and OFA reproduction script does seed torch and numpy per run, so the multi-seed run in progress (`claude-final-es-repro-1`, `es-repro-seeds/`) gives real seeds. The aggregator `evaluation/analysis/es_reproduction_table.py` reports mean, sample sd, standard error and n per cell for the whole table.

### 13.2 Results (raw MSE, mean over the 5 horizons)

Reference points:

| model | spec4 | spec25 | spec151 |
|---|---|---|---|
| LookbackMean | 0.804 | 1.493 | 1.750 |
| Paper: PatchTST / DLinear | 0.317 / 0.342 | 0.873 / 1.060 | n/a |
| Paper: Time-LLM (GPT2) / OFA | 0.307 / 0.299 | 0.849 / 0.846 | n/a |
| Collaborator (TSLib split): OFA / Time-LLM (GPT2, channel-independent) | 0.264 / 0.284 | 0.809 / 0.825 | n/a |
| Library PatchTST, default hyperparameters | 0.321 | 0.831 | incomplete |
| Library PatchTST, **tuned** | 0.311 | 0.849 | 0.901 |
| Library DLinear, default / **tuned** | 0.411 / 0.319 | 1.251 / 1.026 | 1.380 / 1.113 |
| Library Autoformer / LightTS / Informer / Reformer (defaults) | 0.650 / 0.438 / 0.703 / 0.632 | 1.403 / 1.326 / 2.658 / 1.974 | 1.615 / 1.460 / 3.263 / 3.089 |
| Library Autoformer / LightTS / Informer, **tuned** (5 seeds) | 0.434 / **0.329** / 0.588 | 1.328 / **1.093** / **2.864 (worse than default)** | 1.462 / **1.282** / 2.927 |

Ours, rolled out with inherited hyperparameters vs tuned (number of seeds in brackets where fewer than 5):

| model | spec4 inherited / tuned | spec25 inherited / tuned | spec151 inherited / tuned |
|---|---|---|---|
| ResidualLinearAR 2D | 0.275 / 0.277 | 1.022 / 1.013 | 1.074 / 1.102 |
| ResidualLinearAR 1D | 0.311 / 0.302 | 0.965 / 0.969 | 1.105 / 1.110 |
| LinearAR 2D | 0.425 / 0.306 | 2.287 / 1.304 | 1.730 / 1.451 |
| LinearAR 1D | 0.343 / 0.320 | 1.104 / 1.032 | 1.214 / 1.120 |
| ResidualVanillaLSTM 1D | 0.679 / 0.516 | 1.416 / 1.101 | not run |
| VanillaLSTM 1D | 0.885 / 0.520 | 1.817 / 1.697 | not run |
| ResidualVanillaLSTM 2D | 0.803 / 0.748 | 1.454 / pending | 1.698 / 1.749 (1) |
| VanillaLSTM 2D | 0.880 / 0.848 | 2.400 / 2.439 (4) | 2.955 / 2.943 (2) |
| LSTMAttn 2D | 0.799 / pending | 2.522 / 2.204 (3) | 2.879 / 2.900 (4) |
| AutoformerCSA 2D, TCN 2D | 0.829, 0.846 / pending | 1.548, 1.860 / pending | 1.899, 2.397 / pending |

Direct (horizon-specific) vs rolled out, inherited hyperparameters (ResidualLinearAR): direct is worse: spec4 2D 0.305 vs 0.275,
spec25 1D 1.074 vs 0.965, spec151 2D 1.211 vs 1.074.

ResidualLinearAR variants (inherited hyperparameters; mean horizon MSE):

| variant | spec4 1D / 2D | spec25 1D / 2D |
|---|---|---|
| plain | 0.311 / 0.275 | 0.965 / 1.022 |
| + window-std normalization | 0.313 / 0.276 | 0.963 / 1.024 |
| + seasonal baseline | 0.312 / 0.298 | 0.959 / 1.115 |
| + both | 0.314 / 0.298 | 0.957 / 1.115 |

Loss-units test (shared-weight 1D ResidualLinearAR, best-epoch selection; standardized loss vs raw-unit loss):
spec4 0.337 vs 0.322, spec25 0.987 vs 1.022, spec151 0.982 vs 1.045.

### 13.3 Findings so far (ElectroSense)

- **There is large headroom here** (45-65% below the lookback mean), unlike 1-minute POWDER.
- **At 4 channels our ResidualLinearAR 2D (0.275) beats every number in the paper** (best paper entry 0.299) and ties the
  collaborator's best reproduction (OFA 0.264, 0.278 under their other split).
- **At 25 channels it does not**: ours 0.965 vs the paper's best 0.846-0.873 and the collaborator's 0.81-0.83. The gap is
  largest at 192 steps (about 1.30x). At 151 channels PatchTST (tuned, 0.901) also beats ours (about 1.07).
- **What does not explain the 25-channel gap** (each tested): hyperparameter tuning (about 1% for ResidualLinearAR; tuned
  PatchTST is no better than its defaults), rolled-out vs direct form (direct is worse), the seasonal baseline and window
  normalization variants (no help, 2D seasonal hurts), and the loss units (raw-unit loss is no better). The remaining
  explanation is model class: nonlinear patch models (PatchTST, OFA, Time-LLM) beat a linear model on more channels. A small
  nonlinear residual head was proposed as the direct test and **has not been built**.
- **Tuning matters a great deal for the plain linear and the LSTM models** (LinearAR 2D on spec25 2.29 to 1.30; 1D LSTMs by
  20-40%) but not for ResidualLinearAR, which is already near its ceiling. LSTM models remain far behind the linear ones.
- **Head-to-head at 4 channels** (raw MSE): ours ResidualLinearAR 2D 0.275 vs paper OFA 0.299 / Time-LLM 0.307 / PatchTST 0.317 (we beat all of them), vs collaborator's
  Time-LLM 0.284 (we are 3% better) and OFA 0.264 (we are 4% worse). The reproductions of OFA and Time-LLM are untuned (fixed regime), as are the paper's; our model was
  tuned (no gain). A matched-budget tuning of the LLM models is queued (14.1) so the comparison is like for like.
- **Tuned vs default library baselines**: tuning helps 8 of 9 cells for Informer/Autoformer/LightTS (up to -33%), but Informer on 25 channels got worse (+8%). None of
  the three comes near ResidualLinearAR on 4 channels (best: LightTS 0.329 vs 0.275).
- **DLinear needs tuning to be a fair baseline**: the library default underfits (lr 1e-4); tuned it improves 20-40%, but
  ResidualLinearAR is still ahead of it on 4 and 25 channels.

---

**Tuned Informer / Autoformer / LightTS (complete; raw MSE, mean over 5 horizons; defaults 3 seeds, tuned 5 seeds):**

| model | spec4 default / tuned | spec25 default / tuned | spec151 default / tuned |
|---|---|---|---|
| LightTS | 0.438 / **0.329** (-25%) | 1.326 / **1.093** (-18%) | 1.460 / **1.282** (-12%) |
| Autoformer | 0.650 / **0.434** (-33%) | 1.403 / 1.328 (-5%) | 1.615 / 1.462 (-9%) |
| Informer | 0.703 / **0.588** (-16%) | 2.658 / 2.864 (**+8%**) | 3.263 / 2.927 (-10%) |

Tuning helps in 8 of 9 cells. The exception is Informer on 25 channels, where the validation-selected candidate was worse on test than the library default (the
winner was picked on the mean of horizons 12 and 192 only; not diagnosed). LightTS is the best of the three everywhere but it is still behind tuned PatchTST at 25
channels (1.093 vs 0.849) and far behind ours at 4 channels only by 0.329 vs 0.275. These runs keep no weights.

## 14. SpectrumLLM (Time-LLM and OFA): reproduction and use

### 14.1 Collaborator's reproduction on ElectroSense (`training/SpectrumLLM/repro/`)

| item | state |
|---|---|
| Time-LLM (GPT-2 and BERT backbones, our port `models/SpectrumLLM.py`), OFA (`models/OFA.py`), DLinear, paper regime, 4 / 25 / 151 channels, horizons 12-192, both split readings (`literal` and `tslib`) | done: about 291 primary runs in `results_master.csv` (331 rows with superseded and sweep rows) |
| Channel-independent vs all-channel modes | done (channel-independent is the paper's arrangement of Time-LLM) |
| Random-backbone control (all-channel Time-LLM) | done on 4, 25 and 151 channels |
| Channel-independent runs on 151 channels | **not run** |
| **Equal-budget tuning of OFA and Time-LLM (GPT-2) on ElectroSense** (`training/ray/tune_llm_es.py`, 4 unit tests; 16 candidates per (dataset, arm) = the reproduction's defaults + 15 seeded draws over lr 5e-5..1e-3, backbone layers 3/6, patch length 8/16/32 and stride, and for Time-LLM d_model 16/32/64 and d_ff 32/64/128; validation-selected on the mean of horizons 12 and 192 at seed 42; finals = winner x horizons 12/24/48/96/192 x seeds 40-44; spec4 and spec25 only; 128 trials, then 100 finals; winners written to `winners.json` before any test-scored run) | **finished** on the H100 cluster as `claude-llm-tune-es-2` (228 records, 0 failed, 100 final-run weight files; winners in `llm-tuning-electrosense/winners.json`; **results in 22.3: tuned OFA 0.279 / 0.817 and Time-LLM 0.275 / 0.838 (spec4 / spec25)**); it was (resubmitted at 0.25 GPU per run after the first submission, `-es-1`, sat pending because the free capacity was fragmented; `-es-1` was stopped with no task started). Results under `llm-tuning-electrosense/`; the finals save weights, the search trials do not. Cost estimate from the BERT reruns: 4 channels 3-5 min per run, 25 channels 17-54 min, about 65 GPU-h in all. `transformers` and GPT-2 were installed on all three H100 hosts for this (`claude-llm-setup-h100-1`). Until it finishes the LLM models are untuned, so the comparison with our tuned models is not matched. spec151 is not included (the reproduction never ran channel-independent at 151) |
| Checkpoints | stored as trainable parameters only (frozen GPT-2 rebuilt from the Hugging Face id); on the bucket and in `checkpoints/repro/` |
| Pretrained GPT-2 weights | not stored with the checkpoints; downloaded from Hugging Face |

**Findings.** The paper's tables are reproduced and slightly beaten (collaborator OFA on 25 channels: 0.75-0.81 vs the paper's
about 0.85). The advantage comes from channel-independent training: the same Time-LLM in all-channel mode is much worse
(25 channels: 1.01-1.16). The language-model backbone adds little: a random-weight backbone is within 2% of GPT-2 at 4
channels (0.316 vs 0.311) and about 10% worse at 25 channels (1.122 vs 1.011, all-channel), and the non-LLM PatchTST is
as good as OFA. The paper's numbers fit raw (not standardized) MSE. One BERT run at 192 steps looks unstable (2.3x at 4 channels); section 14.3 shows it is a collapse to the lookback mean.

**Tuning disparity.** The reproduction (and the paper) use one fixed setting (lr 1e-4, 10 epochs, patch 16/8, d_model 32, 6 layers). The collaborator's sweeps varied the learning rate (1e-4, 5e-4, 1e-3; worth about 6% on OFA) and epochs (10 to 50, no change), but not under our 16-candidate validation-selected protocol. Our own models, PatchTST, DLinear, Informer, Autoformer and LightTS do get that protocol, so until 14.1's run finishes the LLM models are the only untuned strong baselines.

### 14.2 SpectrumLLM on POWDER (our experiments)

| item | state |
|---|---|
| Runner `training/ray/llm_powder.py` (channel-independent, direct per horizon, 1-minute and 15-minute, same split and metrics as our models) | built, tested, synced |
| `transformers` 4.44.2 (isolated under `/workspace/pylibs`) and GPT-2 weights on all 7 GPU nodes | done (approved by the user) |
| Smoke tests, 15-minute horizon 4, tiny training on CPU | OFA 1.163 dB in 257 s; Time-LLM 1.281 dB in 86 min (pipeline check only) |
| OFA, 15-minute, both horizons x 3 seeds (12 runs, `claude-llm-ofa-p15-1`, A100) | **queued, not started** (no GPU slot) |
| Time-LLM, 15-minute | not launched |
| OFA and Time-LLM, 1-minute | not launched |
| Hyperparameter tuning for these models | none planned: they use the reproduction's regime (lr 1e-4, patch 16/8, d_model 32, 6 GPT-2 layers); must be stated as untuned |

**Findings.** None yet on POWDER. Cost: from the collaborator's 25-channel runs, a channel-independent Time-LLM run is about
2 GPU-hours and an OFA run about 8x cheaper; with 200 channels per window the POWDER cost will be higher (not measured).

---

### 14.3 Time-LLM with BERT at horizon 192: diagnosis and reruns

The collaborator's BERT runs at K=192 were worse than the paper in some cells (TSLib split: 0.794 at 4 channels, 1.551 at 25; paper 0.344 and 1.031). Raw MSE,
horizon 192 only, channel-independent.

| item | state |
|---|---|
| Rescoring six saved checkpoints (3 diverged, 3 controls) three ways: as recorded, one window per batch, and with no padding at all (`training/ray/spectrumllm_diagnose.py`) | **done** (CPU) |
| Reruns with the current code (`training/ray/spectrumllm_rerun.py`): seeds 1-5 at 4 channels and 1-3 at 25, learning rates 1e-4 and 3e-4 (16 runs) | **done** (`claude-bert-rerun-n4-1`, `-n25-1`); these runs kept no weights (they started before the artifact patch) |
| Choosing a remedy (learning rate or seed policy) on validation data only | **done as far as the data allow: no clear remedy** (see below) |
| Telling the collaborator | not done |

All 16 reruns (test MSE raw, horizon 192; **C** = collapsed, i.e. best validation loss not at least 3% below the validation lookback-mean baseline of 0.511 at 4 channels and 0.593 at 25):

| setting | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean (all) | mean (not collapsed) |
|---|---|---|---|---|---|---|---|
| 4 channels, lr 1e-4 | 0.312 | 0.364 | 0.288 | **0.795 C** | 0.293 | 0.410 | 0.314 |
| 4 channels, lr 3e-4 | 0.325 | 0.296 | 0.325 | 0.296 | 0.309 | 0.310 | 0.310 |
| 25 channels, lr 1e-4 | 1.029 | 1.078 | **1.556 C** | | | 1.221 | 1.054 |
| 25 channels, lr 3e-4 | 1.034 | 1.026 | **1.552 C** | | | 1.204 | 1.030 |

Paper: 0.344 (4 channels), 1.031 (25 channels).

**Findings.**
- **Not a padding artifact**: the diverged BERT checkpoints score identically as recorded, per window and with no padding (0.794 / 0.794 / 0.794 at 4 channels; 1.552 in all three at 25).
- **They collapsed to the lookback mean**: test MSE 0.794 vs 0.793 (4 channels), 1.552 vs 1.541 (25 channels), 1.389 vs 1.384 (literal split, seed 0). The error is spread evenly over windows (the top 5% carry 5-6% of it, as in healthy runs), so it is a failure to learn, not a spike.
- **It is detectable on validation alone** (validation equals the lookback-mean baseline for collapsed runs, 30-40% below it for healthy ones).
- **Healthy reruns match the paper**: 4 channels 0.29-0.36 (paper 0.344), 25 channels 1.03-1.08 (paper 1.031).
- **Collapse rate**: 3 of 16 reruns (2 of 8 at lr 1e-4, 1 of 8 at 3e-4); 3 of the original 8 collapsed. 25-channel seed 3 collapses at **both** learning rates (1.556, 1.552);
  4-channel seed 4 collapsed at 1e-4 but trained normally at 3e-4. So it depends on the seed, and a higher learning rate helped in one of the two cases.
- **No clear remedy on validation data**: on the healthy runs, validation prefers 1e-4 slightly (4 channels 0.322 vs 0.328; 25 channels 0.491 vs 0.498), while 3e-4 had one fewer
  collapse. These differences are within noise at this sample size. The defensible report is: all 16 runs plus the original ones, with the collapse rate (about 1 in 5 to 1 in 3), and
  not only the seeds that match the paper.

## 15. TSS-LCD

### 15.1 Table

| item | state |
|---|---|
| Collaborator's reproduction of the authors' results (two campaigns, `training/TSS-LCD/repro/results_master.csv`, 37 arms) | done |
| In-repo integration (a paper-faithful rewrite, `training/TSS-LCD/model.py`) | exists but **never run in the October benchmark**; open decisions in `INTEGRATION_NOTES.md` (dead duplicate model, config-schema trap, checkpoint format, masking default) |
| Authors' released model wrapped into our pipeline with the two smoothing switches (`models/tsslcd_authors/`, `training/ray/tsslcd_authors.py`, 800 columns = 4 receivers x 200 bins) | built and tested (10 unit tests); smoke and reduced run done |
| Reduced schedule (20/20/60 epochs), seed 40, savgol off and centered, full test range | **done** (`claude-tsslcd-reduced-1`) |
| Authors' full schedule (300/300/2000 epochs), 5 seeds (40-44), savgol off and centered, full test range at stride 1 | **finished** on the H100 cluster as `claude-tsslcd-full-2` (10 of 10 runs ok, 30 weight files in `tsslcd-authors-full/weights/`, **results in 22.2: against raw measurements 1.164-1.166 dB (21-23% worse than the lookback mean), with or without the savgol filter**; 10 runs at 0.25 GPU each, about 1.6 h per run on a shared GPU, about 16 GPU-h; `-full-1` was stopped with no task started); results `tsslcd-authors-full/`; weights saved. My choice to run the full schedule (the earlier request was for a reduced timing run) |
| Causal savgol setting | implemented, not run |
| TSS-LCD at 15 minutes, held-out-receiver or other transfer tests | not planned |

### 15.2 Timing (shared H100 GPU, reduced run)

| stage | seconds per epoch | full-schedule estimate |
|---|---|---|
| context autoencoder (75.8M parameters) | 5.9 | 300 epochs, about 30 min |
| future autoencoder (2.9M) | 0.21 | 300 epochs, about 1 min |
| diffusion (0.5M, re-encodes each window every epoch) | 1.9 | 2000 epochs, about 63 min |
| inference, 9,888 test windows | 53 s total | |
| **one run** | about 5 min reduced | **about 1.6 h full** |

### 15.3 Results of the reduced run (test MAE in dB, mean over t+1, t+15, t+60; **not final**)

| savgol | eval smoothing | scored vs raw measurement | scored vs the model's own target |
|---|---|---|---|
| off | off | 1.692 | 1.692 |
| off | centred 6:3 / trailing 6:3 | 1.681 / 1.682 | 1.545 / 1.545 |
| centered | off | 1.551 | 1.272 |
| centered | centred 6:3 / trailing 6:3 | 1.537 / 1.538 | 1.257 / 1.258 |

For comparison on the same test range: lookback mean 0.950 dB, ResidualLinearAR 0.924 dB.

### 15.4 Findings so far

- **The authors' headline is leak-dependent** (collaborator's analysis): it reproduces (MSE about 0.15-0.19 vs the paper's
  0.1541) only under evaluations that leak: a random split of pooled windows (0.155), or the first commit's overlapping split
  (0.164-0.186; the two released files overlap, 55% of test windows also appear in training). On a chronological split the
  same code scores 0.45-0.88; persistence scores about 0.006. The savgol filter alone makes the task about 660x easier
  (persistence 4.25 raw to 0.0064 smoothed), and eval-time averaging is about 19% of the headline. There is no evidence the
  authors shuffled windows; the random split is the collaborator's hypothesis test.
- **On raw (unsmoothed) targets the authors' model is dominated by the trivial predictor** in every configuration the
  collaborator tried (MSE 7.3-39 vs persistence 4.14).
- **On POWDER (reduced schedule)** it is far worse than the lookback mean (1.55-1.69 dB vs 0.95). Training on the filtered
  series helps even when scored against the raw measurement (1.69 to 1.55), eval-time smoothing changes little, and a
  centred and a trailing eval window give nearly the same numbers. These are one seed at 5% of the authors' epochs, so they
  may change with the full schedule.

---

### 15.5 The authors' first commit (d2d4f29) on their own data: savgol and eval-time averaging on and off

Protocol (documented in `training/TSS-LCD/repro/run_firstcommit_eval.py` and the authors' first-commit `train.py`): the authors' recovered 6838 x 750
series, whole-series min-max scaling, window 50 / future 10, train windows `range(int(N*0.8))`, test windows `range(int(N*0.555), N)` (55% of the test
windows are also training windows), batch 256, 300/300/2000 epochs, first-step MSE in dBm, 10% test-time context masking. Runner:
`training/ray/tsslcd_firstcommit.py`; the authors' files are in the bucket under `datasets/tsslcd/`; results under `tsslcd-firstcommit-results/`.

| trained with savgol | eval-time averaging (w6/w3) | scored against | MSE, seeds 0 / 1 / 2 | state |
|---|---|---|---|---|
| on (authors') | on | smoothed target (authors' protocol) | 0.164 / 0.168 / 0.195 (collaborator's stored); 0.163 / 0.170 / 0.191 (my rescoring) | done |
| on | off | smoothed target | 0.201 / 0.205 / 0.224 (stored); 0.203 / 0.208 / 0.221 (rescoring) | done |
| on | on (predictions only) | raw measurement | 2.560 / 2.555 / 2.581 | done (rescoring) |
| on | off | raw measurement | 2.596 / 2.579 / 2.618 | done (rescoring) |
| persistence on the raw series | | raw measurement | 4.247 | done |
| **off** | on (w6/w3), scored against the smoothed (w3) target | the authors' protocol minus the filter | 1.385 / 1.379 / 1.370 (mean **1.378**) | done (retrained, `claude-tsslcd-fc-off-1`) |
| **off** | off | raw measurement (= the model target, as nothing was filtered) | 3.591 / 3.570 / 3.648 (mean **3.603**) | done |
| **off** | on (predictions only) | raw measurement | 2.853 / 2.841 / 2.834 | done |
| on (retrained, as a check of my harness) | on | smoothed target | 0.193 / 0.152 (stored arms: 0.164 / 0.168) | done, 2 seeds (`claude-tsslcd-fc-on-1`) |
| on (retrained) | off | smoothed target | 0.227 / 0.195 | done |
| on (retrained) | on / off | raw measurement | 2.580 / 2.537 and 2.620 / 2.574 | done |
| **third retrain, with weights saved** (`claude-tsslcd-fc-weights-1`, 6 runs, `tsslcd-firstcommit-results-w/`, 18 weight files): savgol on, eval averaging on, smoothed target | on | smoothed target | 0.184 / 0.163 / 0.184 | done |
| same retrain, savgol on, no averaging / raw target with averaging | off / on | smoothed / raw | 0.219 / 0.204 / 0.215 and 2.575 / 2.548 / 2.578 | done |
| same retrain, **savgol off**, averaging on (w6/w3 target) | on | smoothed (w3) target | 1.361 / 1.356 / 1.416 | done |
| same retrain, savgol off, no averaging / raw target with averaging | off / on | raw | 3.514 / 3.523 / 3.655 and 2.829 / 2.817 / 2.882 | done |

**Findings so far.** The collaborator's numbers reproduce within about 2% when I rescore their checkpoints, so the harness is faithful. Turning off the
eval-time averaging raises the error by about 20-24% (0.164 to 0.201) while still scoring against the smoothed target. Scored against the raw
measurement instead, the filter-trained models get about 2.6, which is 13-16 times worse than on the smoothed target but still 39% better than
persistence (4.25).

**Answer to "what does it come out to without each piece"** (smoothed target where the filter is on; first-step MSE, dBm squared):
- **Without eval-time averaging (filter on):** 0.20-0.23 (about +20%).
- **Without the savgol filter (eval-time averaging on):** **1.38**, about 8 times the headline. Almost all of the headline comes from the filter on the data.
- **Without both:** **3.60**, only 15% better than persistence (4.25). Eval-time averaging matters little when the filter is on but a lot when it is off, since it
  is itself a smoother.
- **Against raw measurements** the filter-trained models reach about 2.6 and the filter-free models about 2.8 with prediction averaging (3.6 without).
- **Retraining is not bit-reproducible, but the conclusions are stable**: five retrains of savgol-on give 0.152-0.193 (mean 0.175) against the stored 0.164-0.195, so the
  seed-to-seed spread is about +-0.02; the savgol-off numbers are tighter in relative terms (1.36-1.42 with averaging, 3.51-3.66 without). The second retrain (with weights) agrees
  with the first within that spread in every cell.

### 15.6 Does the savgol filter's effect depend on the prediction horizon? (authors' AERPAW data, 2026-11-08)

Question: the filter helps the raw-measurement score on the authors' data but not on POWDER; is that a horizon effect? Test: the six saved first-commit retrains (savgol off and on, seeds 0-2; `tsslcd-firstcommit-horizons/`, `python -m training.ray.tsslcd_firstcommit horizons`) scored at **all ten future steps** (1 step = 1 minute) against the raw measurement, MSE in dBm squared, mean of 3 seeds:

| step | savgol off, no averaging | off, averaging | savgol on, no averaging | on, averaging | persistence | on vs off (no averaging) | on vs off (averaging) |
|---|---|---|---|---|---|---|---|
| 1 | 3.56 | 2.84 | 2.60 | 2.57 | 4.25 | -27.0% | -9.7% |
| 2 | 3.64 | 2.92 | 2.63 | 2.59 | 4.74 | -27.8% | -11.1% |
| 5 | 3.75 | 3.02 | 2.68 | 2.65 | 5.09 | -28.5% | -12.4% |
| 10 | 3.85 | 3.04 | 2.83 | 2.78 | 5.30 | -26.5% | -8.5% |

(steps 3, 4, 6-9 lie in between; ratios stay between -26% and -29% without averaging and -8% and -12% with it.)
**Finding.** The filter's benefit against raw measurements is **flat across steps 1-10**, so the horizon (within 10 minutes) does not explain why the filter helps on the authors' data and not on POWDER (where it makes no difference from t+1 to t+60, 22.2). Against raw measurements the models' error grows only about 9% from step 1 to 10 (persistence's grows 25%); against the model's own smoothed target the savgol-on error roughly doubles over the same steps (0.177 at step 1, 0.241 at step 5, 0.347 at step 10, authors' protocol), which the raw score hides under the noise floor. The dataset difference remains unexplained.

**The authors' plotting code** (`test.py::visualise`, github.com/Xlab2024/TSS-LCD at `b156d0a007`, read from a scratch copy, not run, not committed): (a) the plotted "true" curve is the **savgol-smoothed** series, because the test CSV is passed through `preprocess_multi_frequency` (Savitzky-Golay window 51, order 2, `mode='interp'`, along time) before windowing; (b) it contains the eval-time moving-average block (`np.convolve(..., mode='same')`) but with `window1 = window2 = 1`, i.e. **no eval-time averaging at that commit** (comment: "simple smoothing, add to the parser if needed"); the earlier first commit used 6 on predictions and 3 on ground truth; (c) it plots only the **first** future step and three hard-coded frequency bins (100, 350, 500), writes `gt_mat.csv` and `pred_matours.csv`, and computes **no metrics**; (d) the scaler is fit on the test CSV itself, and the filter is applied to each CSV separately. So the figure compares predictions with a smoothed truth, which flatters the visual match relative to the raw measurements.

## 16. Reproduction-style and cross-cutting results

| item | state / finding |
|---|---|
| Does the paper's SpectrumLLM table reproduce on raw MSE? | yes (section 14) |
| Does a level pathway matter? | yes, strongly: residual and decomposition models avoid the failures of plain LSTMs, both in distribution and under shift (sections 1, 2) |
| Is headroom larger at coarser resolution? | yes: about 3% at 1 minute, about 35% at 15 minutes, 45-65% on ElectroSense (all vs the lookback mean) |
| Is a pooled model better than per-receiver? | inconclusive (incomplete, section 12.3) |
| Do our models beat the paper's models? | at 4 channels yes (0.275 vs the paper's best 0.299; the collaborator's untuned OFA reproduction reaches 0.264); at 25 and 151 channels no (section 13) |
| Is the TSS-LCD headline attributable to the smoothing? | yes: savgol filter on the data gives most of it (0.15-0.19 with, 1.38 without), eval-time averaging adds about 20% on top; with neither the model is only 15% better than persistence (section 15.5) |
| Does BERT collapse at horizon 192? | sometimes (3 of 16 reruns); healthy runs match the paper (section 14.3) |

### Tuning status by model family (equal budget = 16 candidates, validation-selected, finals on seeds 40-44)

| family | 1-minute POWDER | 15-minute POWDER | ElectroSense |
|---|---|---|---|
| Our models (linear, LSTM, TCN, AutoformerCSA, ConvLSTM, DSwinLSTM) | done (October search; DSwinLSTM round 2) | trials done, **finals not started**, 1D LSTM trials still running | trials done; finals mostly done |
| PatchTST, DLinear | not run | not run | tuned, done |
| Informer, Autoformer, LightTS | not run | not run | tuned, done |
| Reformer, TimesNet, other library models | not run | not run | defaults only (TimesNet incomplete) |
| OFA, Time-LLM (SpectrumLLM paper) | not run | smoke only, untuned (planned regime) | **untuned**; matched-budget tuning queued |
| TSS-LCD (authors' model) | authors' schedule, no tuning | not planned | not applicable |
| TimeRAN, MOMENT, TimesFM-3 | not run on October data | not run | not run |

---

## 17. Planned or deferred, not yet run (consolidated)

**Weekend plan** (user offline from Nov 6; the list was set by the user, I do not add to it; do not pause running jobs). Status checked against the bucket at 19:20 UTC:

| priority | item | status |
|---|---|---|
| 1 | DSwinLSTM round-2 seeds 40-44 | **done**, exported to `checkpoints-oct2026-r2/` (5 weight files). Final-range scoring is part of `claude-final-eval-v3-4d-2` (A100, running) |
| 2 | Missing 2400-2600 models (ConvLSTM, ResidualConvLSTM, 1D VanillaLSTM, 1D ResidualVanillaLSTM), seed 42 | **done** (`claude-band2400-missing-1`) |
| 3 | Extra seeds (40, 41, 43, 44) on 2400-2600, non-DSwinLSTM models | **done** (checked 11-07 23:30 UTC): CPU linear models and the five GPU models all have seeds 40-44 on 2400-2600 (the GPU seeds ran on the H100 cluster as `claude-h100-band2400-gpu-seeds-1`) |

**Other open work**

| item | notes |
|---|---|
| Matched-budget tuning of OFA and Time-LLM on ElectroSense | **finished** (`claude-llm-tune-es-2`); results not analyzed |
| **15-minute POWDER models rerun with weights saved** (pooled, inherited hyperparameters, both bands, seeds 40-44; the first runs kept metrics only) | **finished**: 514 of 514 runs ok with 514 weight files (last write 06:26 UTC on 11-07), prefix `powder15-results-w/`; jobs ran on the H100 cluster: `claude-p15w-rollout-1` (114 runs), `claude-p15w-direct-1` (240 runs, LSTM family and AutoformerCSA), `claude-p15w-direct-linear-1` (160 CPU runs). First 102 runs succeeded, all with weights. Remaining after the H100 lease: re-queue on the A100. The per-receiver and pooled comparison jobs (`sitecmp2`, A100) and the tuned finals are not covered |
| TSS-LCD authors' full schedule, 5 seeds, savgol off and centered | **finished** (`claude-tsslcd-full-2`, 10 of 10 ok); results not analyzed; the causal savgol setting is implemented, not run |
| Inference-cost benchmark on an idle GPU | script extended to all checkpointed POWDER models (158), smoke-tested, runbook `training/INFERENCE_BENCHMARK.md`; **runs on the new H100 cluster once it is up** (section 10) |
| Score VanillaLSTM 1D and ResidualVanillaLSTM 1D at stride 1 | cheap, evaluation only, not queued |
| Read the 4D final evaluation (ConvLSTM, ResidualConvLSTM, round-2 DSwinLSTM, remaining seeds) | job `claude-final-eval-v3-4d-2` finished (126 objects under `oct2026-results/final-eval-v3/`); results not yet read or tabulated |
| Rescore per-receiver, pooled-6 and 2400-2600 at stride 1 on the final range | `final_eval` may not accept the band-run prefixes; not checked |
| 15-minute: tuned finals, direct vs rolled-out comparison, per-receiver vs pooled on identical windows | tuning still running on the A100 |
| ElectroSense: remaining tuned finals (AutoformerCSA, TCN, parts of the LSTM family) and 1D LSTMs on spec151 | `claude-tune-es-1` running (12 final runs) |
| A nonlinear residual head to test the 25-channel gap | not built |
| **TODO (added 2026-11-08): tuned-median baseline** | Add the validation-tuned 15-30 minute median (lookback length chosen on validation, as in the level diagnostics, section 4) as a baseline in every 1-minute and 15-minute table. Reason: it beats the 60-minute lookback mean by 3-8% at 1 minute, more than ResidualLinearAR 2D's 2.7%, so the current reference baseline is not the strongest simple predictor. Evaluation only, CPU. **Consistency check done 2026-11-09:** the level-diagnostics numbers are *not* directly comparable with the model tables. The diagnostics score a plain direct 60-minute mean per (receiver, band) cell; the pipeline scores `lookbackmean` as a one-step model rolled out autoregressively, and the two differ by about 1% (600-800 mean over horizons 1/15/60: 0.958 plain vs 0.950 in the tables; 2400-2600: 0.778 vs 0.768). So the median is recomputed through the pipeline itself (`training/ray/median_study.py` part A, `lookbackmean2d` with `statistic`/`window`; K grid swept on validation and test, K selected on validation only, one K for all horizons and one per horizon; 1D and 2D are identical for a per-bin median). **Sweep done (section 24.3: the tuned median beats the rolled-out lookback mean by 3.2% and 4.4% at 1 minute and matches ResidualLinearAR 2D and the best direct learned models; at 15 minutes only a per-horizon window works, +17% and +11%).** Still to do: add the median row to the main model tables |
| **TODO (added 2026-11-08): median-anchored and MAE-loss ResidualLinearAR** | ResidualLinearAR with the tuned median as its anchor instead of the 60-minute mean, plus an MAE-loss variant (1D and 2D, both bands, 1 and 15 minutes, CPU). Reason: swapping in another linear smoother is only a reparametrization for a linear model, but the median is nonlinear and ignores spikes, which matches the asymmetric leftover structure (dips persist, spikes revert) and the MAE-loss gradient-boosting gain in section 4. Separates "anchor" from "loss". Expected gain at most a few percent. Neural residual models with the new anchor are out of scope before the deadline. Must finish before the A100 lease ends (2026-11-10 18:00 UTC) if run there. **Code written** (`median_study.py` part B; model option `baseline: median` + `baseline_window`, training option `train.loss: mae`, defaults unchanged, tests in `tests/test_median_variants.py`): variants `ref` (mean anchor, MSE, retrained here so the comparison is like for like), `mae`, `median`, `median_mae`; 1D and 2D, both bands, 1 and 15 minutes, seeds 40-42, hyperparameters inherited from the existing winners, K for the median anchor taken from the validation sweep; 1-minute 1D test windows thinned 10x (memory), reference included. Host memory is reserved per task (8-30 GB) and at most 4 tasks run at once. **Running (`claude-median-full-1`, 64 runs; first result 24.3: median anchor plus MAE loss 1.8% better than the reference for the 1D 1-minute 2400-2600 model, each change alone slightly worse)** |
| **TODO (added 2026-11-09): level-shift response test (why TCN does well in distribution but fails under transfer)** | Hypothesis from the code, untested: `models/TemporalConvNet.py` has no residual connections, but `_leveled_init` (on by default) sets every convolution to a uniform averaging filter with zero bias, so with kernel 2 and dilations up to 32 (receptive field 64 >= lookback 60) the untrained network outputs a moving average of the input, i.e. starts as the lookback mean. The tuned October TCN has 713 parameters (section 22.6), possible only in channel-independent mode, so it plausibly stays a reweighted per-bin smoother: hence 0.930 / 0.740 dB at 1 minute, level with the linear models. The ReLUs after each layer act on z-scored inputs; the network can stay linear inside the training range (positive biases), but under transfer with frozen training statistics, z-scores far outside that range are clipped, so the forecast cannot follow a shifted level (the >20 dB transfer failures). Paper point: a structural level pathway (residual, per-window normalization, decomposition) carries any level; a level pathway made by initialization works only inside the training range. **Test** (evaluation only, minutes of CPU per model, on the cluster): for existing checkpoints of every model, add an offset c dB (e.g. c = -20..+20) to the whole input window and record the mean change in the forecast; plot forecast shift vs c. Expected: residual / RevIN / decomposition models track c exactly (slope 1); TCN tracks c near 0 and saturates for large offsets, more sharply for negative c; plain LSTMs saturate early. Also gives the cost/model table a "level-invariant: by construction / measured" column. Not started |
| **TODO (added 2026-11-09): OFA and its language-model controls on ElectroSense through `tune_direct`** | **Minimal version launched 2026-11-09 15:32 UTC; RESULT IN (15:50): the gap is a pipeline artifact, see 17.1** (`claude-es-check-1`, A100 cluster, output `direct-tuning-es-check/electrosense/spec25/{ofa,ofa_rand,ofa_scratch}/`). Why: on ElectroSense the cheap models (RevIN-MLP 0.309 / 1.085 raw MSE) are 11% / 32% behind OFA (0.279 / 0.817), unlike POWDER where RevIN-MLP matches OFA (section 24.11), but the two were run through different pipelines (cheap models: `tune_direct --resolution electrosense`; OFA and Time-LLM: `tune_llm_es.py`, the collaborator's code with the TSL loader). Signs that `tune_direct` is handicapped on ElectroSense: our PatchTST is 28% / 7% behind TSL PatchTST, and our direct ResidualLinearAR is 17-18% behind the rolled-out one. **Design (reduced from 16 candidates x 2 datasets x 5 seeds x 5 horizons x 3 arms, about 600 runs, to 18 runs):** spec25 only (the large gap; spec4 is 11% and cannot separate pipeline from model); **no search**: all three arms use the winning OFA settings of the other pipeline (learning rate 1.59e-4, 3 layers, patch 32/32), so any difference between `tune_direct` OFA and the reference is the pipeline, not tuning (new option `tune_direct --fixed-params`); horizons 12 and 192 only (the tuning horizons; per-horizon references exist for every model); seeds 40-42; `ofa_rand` and `ofa_scratch` with the same settings (an equal-configuration comparison, disclosed as not separately tuned). **Reading** (reference per horizon 12 / 192, raw MSE: OFA 0.581 / 1.029; RevIN-MLP 0.787 / 1.480): `ofa` near 0.58 / 1.03 means the pipelines agree and the gap is real; `ofa` near 0.79 / 1.48 means `tune_direct` is handicapped and the cheap-model numbers on ElectroSense are suspect; if the gap is real, `ofa_rand` and `ofa_scratch` near `ofa` mean pretraining is not the cause (on POWDER it was at most 1.3%), clearly worse means it is. Not run: spec4, the 16-candidate search, horizons 24/48/96, Time-LLM. Result in (17.1): in the article's window-averaged convention RevIN-MLP matches OFA and Time-LLM on ElectroSense too, so "the simplest model matches the language-model models" holds on both datasets, with the caveats stated there. |
| **TODO (added 2026-11-09): explain the PatchTST implementation gap on ElectroSense** | `models/PatchTST.py` through `tune_direct` scores 1.083 (spec25) / 0.333 (spec4) raw MSE against 0.849 / 0.311 for the tuned TSL PatchTST, the same architecture. Compare the two on spec25 at one setting: lookback and window stride, train/validation/test windowing at the split edges, scaler fit and scope, loss in raw vs standardized units, direct per-horizon vs multi-output head, epochs and early stopping. Whatever explains it probably also explains the direct ResidualLinearAR gap (item above) and may change every `tune_direct` ElectroSense number. CPU/small GPU; debugging, not a sweep. Not started |
| OFA / Time-LLM on POWDER (15-minute, then 1-minute) | OFA 15-minute queued on the A100 (no result yet); Time-LLM and 1-minute not launched |
| Investigate why tuned Informer on 25 channels is worse than its default | not done |
| Tell the collaborator about the BERT collapse findings | not done |
| TimeRAN and MOMENT on October data; TimesFM-3 full evaluation | deferred |
| TimesNet reruns (spec4, spec25); PatchTST default on spec151 | stopped earlier; rerun when GPUs are free |
| Spectral-context ablation (529 cells left), spatial ablation | paused at the user's request / waiting for checkpoints |
| Atlas: link headroom and realized gain to the model results (figure) | not done |
| Five-seed 1-minute runs on 2400-2600 for DSwinLSTM | about 27 GPU-h per seed; out of scope |
| Weights for TSL library baselines and `loss_units.py` | not patched (section 19) |
| Paper edits: anonymize author block, Ethics appendix, AI-use disclosure, regenerate `.bbl`, update tables to the four-node set, add the new findings (TSS-LCD smoothing, BERT collapse, tuned baselines) | not done |

**Deferred (2026-11-09, at the user's request): why do OFA and Time-LLM beat the cheap models on ElectroSense but not on POWDER?** Full plan in 17.1; not started.

### 17.1 Deferred: the ElectroSense versus POWDER gap between the language-model models and the cheap models

**Question.** On POWDER 15-minute data RevIN-MLP matches or slightly beats OFA and Time-LLM (0.744 vs 0.749 and 0.752; 0.617 vs 0.622 and 0.626), but on ElectroSense it loses clearly (raw MSE, mean over horizons: spec4 0.309 vs OFA 0.279 and Time-LLM 0.275; spec25 1.085 vs 0.817 and 0.838). RevIN-MLP, OFA, Time-LLM and PatchTST are all channel-independent with weights shared by every bin, so this is not a weight-sharing difference (it is for the LinearAR 2D family, which has separate weights per bin). Results: sections 24.1 and 24.11.

**What the data already say.** (1) The gap is at every horizon, not only long ones: on spec25 the cheap models trail OFA by 22-44% at all five horizons including horizon 12 (RevIN-MLP 0.787 vs 0.581), on spec4 by 2-18%. (2) It is not clearly about language models: our own PatchTST (no LLM) is 33% behind OFA on spec25 (1.083 vs 0.817), whereas the TSL library's PatchTST is within 4% (0.849), so implementation or training recipe varies as much as backbone type. (3) The direct ResidualLinearAR is 17-18% worse than the rolled-out one on ElectroSense and 2-3% better on POWDER 15 minutes. (4) Time span is not smaller on ElectroSense: 2,872 rows = about 30 days (about 21 training days, 3 validation, 6 test) against 7 training days on POWDER, but ElectroSense has 4 or 25 series against 800 on POWDER, so far fewer training windows (about 7k or 45k against 456k; the 800 POWDER series are highly correlated, so the effective gap is smaller); lookback and horizons are longer (192 and up to 192 steps, against 96 and up to 48). "Data-starved" is a hypothesis, not a measurement. (5) The ElectroSense test set is only 6 days (about 190 heavily overlapping windows at horizon 192), so the seed sds understate the uncertainty. (6) POWDER 15 minutes is nearly saturated: all tuned nonlinear models are within 1.5% (0.744-0.755), against a 33-75% spread on ElectroSense. (7) Comparability hole: the ElectroSense OFA and Time-LLM numbers (section 22.3) came from a different runner (`tune_llm_es`) than the cheap models (`tune_direct`, ElectroSense mode); the protocols were meant to match but were never verified end to end.

**Experiments (each isolates one explanation).**

| # | Question | What to run | Reading of the outcome | Cost and readiness |
|---|---|---|---|---|
| 1 | Is the gap a pipeline artifact? | OFA on spec4 and spec25 through `tune_direct --resolution electrosense` (same runner and 16-candidate protocol as the cheap models) | reproduces 0.279 / 0.817: the gap is real; lands near RevIN-MLP: pipeline artifact | runner supports it now; minutes per trial |
| 2 | Does the pretrained backbone matter when data is small? | the same runner with `ofa_rand` and `ofa_scratch` | pretrained clearly better than both: pretraining helps on ElectroSense (unlike POWDER, where it adds at most 1.3%); all equal: it does not | runner supports it now; minutes per trial |
| 3 | Sharing or normalization among the linear models? | a plain shared-weight linear model without normalization next to NLinear and DLinear | separates weight sharing from normalization (the doc's 24.1 comparison of NLinear with LinearAR 2D mixes them) | about 15 lines of model code plus wiring; CPU; trivial |
| 4 | Channel count or sample count? | shrink POWDER 15 minutes to one receiver with 4 or 25 bins, then RevIN-MLP, OFA and PatchTST | OFA's lead appears at 4-25 channels: channel or sample count; no lead: something specific to ElectroSense | needs 1-2 hours of code to subset bins and receivers; cheap to run |
| 5 | Why is our PatchTST 28% behind the TSL one? | our PatchTST with the TSL recipe (learning-rate schedule, early stopping, settings) | tells whether the gap is training recipe | optional, after 1 and 2 |

**Where and when.** The A100 cluster (small GPU shares fit beside the ablations; the workers already have the language-model setup; the lease ends 2026-11-10 18:00 UTC) or any cluster with free GPUs. Experiments 1-3 together are roughly 10-20 GPU-hours on small shares (a rough guess; OFA's trial time on ElectroSense has not been measured). Suggested order: 1 and 2 first (no code needed), write 3 and 4 while they run, 5 last. A long-horizon POWDER test is not worth running (the gap exists at every horizon).

**CORRECTION, cause found (2026-11-10 01:30 UTC): the two pipelines report different quantities, not different quality.** The collaborator's script (`run_paper_regime.py`, the article and Time-Series-Library convention) scores the **whole forecast window**: the MSE is averaged over steps 1..H. Our evaluation (`evaluation_integrated.py`, `batch_forecast[:, horizon - 1]`) scores the **single endpoint step t+H**. Error grows with the step, so the window average is lower: 0.579 / 1.018 against 0.780 / 1.429 for the *same predictions* of the other pipeline's OFA at horizons 12 / 192 (spec25, raw MSE), and `tune_direct`'s OFA with identical settings scores 0.792 / 1.443 (endpoint), within 1.5%. The bisect (`es_bisect.py`, same OFA settings, one switch at a time inside the other script) had already excluded the loss units (`--space std`: 0.586 / 1.068), the test-window convention (`--split literal`: 0.533 / 0.988; our split borders are in fact the TSL ones, so both score the same 563 windows at horizon 12), batch size (32: 0.789 / 1.476 in `tune_direct`) and pretraining (random-init and from-scratch arms equal pretrained OFA). **Consequences:** (a) `tune_direct` is not handicapped; the sentences above that say so, and the three "consequences" (a)-(c) of the earlier paragraph, are superseded by this one. (b) In a common (endpoint) convention OFA equals the cheap models on spec25 (0.78 / 1.43 against RevIN-MLP 0.787 / 1.480), so there is **no gap**, and pretraining does not matter there. (c) **Every ElectroSense comparison across the two groups is invalid**: the OFA and Time-LLM numbers (22.3), the paper's numbers and the TSL library baselines (13.2; `tune_tsl.py`) are window-averaged, while our own models (`tune_fixed`, `tune_direct`, including the rolled-out ResidualLinearAR 0.275 / 1.022) are endpoint. The statements in 22.3 that OFA and Time-LLM beat ResidualLinearAR by 13-15% at 25 channels and that tuned PatchTST is 1-4% behind them are therefore not supported. (d) **Fix:** add window-averaged metrics (mean over steps 1..H) to our evaluation for direct and rolled-out models, rerun the ElectroSense evaluations for our models in both conventions, and rerun the OFA and Time-LLM finals with the endpoint field now written by the script (fast: about 36 s per final). POWDER is not affected (all POWDER numbers are endpoint, within one pipeline). **RevIN-MLP in the article's convention (2026-11-10 01:45 UTC; `training/ray/es_window_rescore.py`, output `es-window-rescore/`; all 50 saved checkpoints rescored at steps 1..H and averaged, no retraining; the endpoint value of the same evaluation reproduces the value recorded at training time to 9e-8 for every checkpoint).** Window-averaged raw MSE, mean of 5 seeds, horizons 12 / 24 / 48 / 96 / 192 and the mean over horizons:

| Dataset | Model | 12 | 24 | 48 | 96 | 192 | mean |
|---|---|---:|---:|---:|---:|---:|---:|
| spec4 | RevIN-MLP (window-averaged) | 0.254 | 0.274 | 0.280 | 0.280 | 0.334 | **0.284** |
| spec4 | RevIN-MLP (endpoint, as recorded before) | 0.291 | 0.297 | 0.299 | 0.297 | 0.361 | 0.309 |
| spec4 | OFA, tuned (22.3) | 0.247 | 0.263 | 0.271 | 0.291 | 0.325 | 0.279 |
| spec4 | Time-LLM, tuned (22.3) | 0.246 | 0.270 | 0.272 | 0.280 | 0.307 | 0.275 |
| spec25 | RevIN-MLP (window-averaged) | 0.592 | 0.726 | 0.836 | 0.926 | 1.051 | **0.826** |
| spec25 | RevIN-MLP (endpoint, as recorded before) | 0.787 | 0.934 | 1.091 | 1.132 | 1.480 | 1.085 |
| spec25 | OFA, tuned (22.3) | 0.581 | 0.722 | 0.823 | 0.930 | 1.029 | 0.817 |
| spec25 | Time-LLM, tuned (22.3) | 0.594 | 0.736 | 0.844 | 0.954 | 1.063 | 0.838 |

**So on ElectroSense, as on POWDER, RevIN-MLP matches the language-model models in the article's convention:** 1.8% and 3.3% behind OFA and Time-LLM on spec4, 1.1% behind OFA and 1.4% ahead of Time-LLM on spec25 (OFA seed sds 0.003-0.014). Caveats: the OFA and Time-LLM rows come from the other pipeline (tuned, 5 seeds) and RevIN-MLP from `tune_direct` (tuned, 5 seeds), now in one convention; the ElectroSense test set is only 6 days, so differences of 1-3% are within what the test period can move. The earlier 11% / 32% "gap" (24.11) is fully explained by the convention. **Result of the minimal check (2026-11-09 15:50 UTC; `claude-es-check-1`, spec25, same OFA winner settings in both pipelines: learning rate 1.59e-4, 3 layers, patch 32/32; raw MSE; seeds 40-42; `direct-tuning-es-check/`):** OFA through `tune_direct` scores **0.792 (sd 0.002) at horizon 12 and 1.443 (sd 0.029) at horizon 192**, against 0.581 / 1.029 for the same model and settings in the other pipeline (`tune_llm_es`, 22.3), and against RevIN-MLP 0.787 / 1.480 and PatchTST 0.802 / 1.417 in `tune_direct`. So **OFA in `tune_direct` equals the cheap models; the pipeline, not the model, produces the 36% gap.** The random-init control (`ofa_rand`, 2 of 3 seeds so far) is the same as pretrained OFA (0.800 / 1.435), consistent with pretraining mattering little inside `tune_direct`; `ofa_scratch` and the third `ofa_rand` seed were still running. **Consequences:** (a) the section 24.11 statement that the cheap models trail OFA on ElectroSense is retracted; neither pipeline's cheap-versus-LLM comparison is currently valid. (b) All ElectroSense results from `tune_direct` (the cheap models, our PatchTST and its ablations, the direct ResidualLinearAR) are handicapped by the same cause, which also explains the unexplained PatchTST gap to TSL (28%) and the direct-versus-rolled-out gap (17-18%). (c) The cause is some difference in how the two pipelines train or score, not yet isolated. Differences found by reading `tune_llm_es.py` / `run_paper_regime.py`: batch 32 with micro-batches (ours: 4 for the language models, 8-64 for others), a raw-unit loss (ours: z-scored), a fixed 10 epochs without early stopping (ours: early stopping with patience 5 on a 287-row validation set), and the TSL loader's test windows (it starts the test set `seq_len` rows earlier, so about twice as many windows, starting in the validation period; ours keeps inputs and targets inside the test partition: at horizon 192 about 190 windows against about 380). **Next step (bisect, one factor at a time with the OFA settings fixed):** batch 32 (`claude-es-check-bs32-1`, launched 15:51, same 6 runs; prefix `direct-tuning-es-check-bs32`), then raw-unit loss, then no early stopping, then the TSL test-window convention; whichever closes the gap tells which setting `tune_direct` must adopt for ElectroSense, after which the cheap models must be rerun there. **Until then:** the paper must not extend "the language model contributes nothing" or "the simplest model matches OFA" beyond POWDER, and the ElectroSense comparison should carry the runner caveat.

## 18. Job and cluster state (2026-11-07 23:30 UTC)

**New H100 cluster (rebuilt, up since 2026-11-07 night; idle since 00:50 UTC on 11-08, benchmark finished)**: 6 hosts, 6 H100s, 288 CPUs, head `h100-head` (`ray-worker-2`, 129.114.27.86); dashboard tunnel `ssh -F training/ray/docker/h100-ssh.conf -fN -L 8266:localhost:8265 -L 9002:localhost:9001 h100-head`.
Leases: `ray-new-1..4` end **2026-11-09 13:40 UTC**, `ray-worker-1` and the head end 2026-11-10 18:00 UTC. Checked 2026-11-08: all 6 workers on code `e596b8f` (checksum-verified on every host); the restored MinIO holds 4,087 objects (matches the backup); data synced
to all six (211 objects, 4.1 GB: POWDER 106, ARA 96, COSMOS 6, AERPAW 3, i.e. every testbed in the paper draft). The inference benchmark smoke test ran there on idle GPUs (section 10).

**Old H100 cluster ("mixed-gpu")**: the lease ended 2026-11-07 16:58 UTC; the dashboard is unreachable. Everything it ran finished before that (see the overnight paragraph at the top) and is in
`s3://spectrum`. A **new H100 cluster is being brought up**; the first job on it should be the inference benchmark (section 10), which needs idle GPUs.

**A100 "tacc"** (4 of 4 GPUs allocated, i.e. at capacity; 32 of 1024 CPUs; lease ends 2026-11-10 18:00 UTC, about 2.8 days)

| job | what | state |
|---|---|---|
| `claude-tune-p15-1d-lstm-2` | 15-minute tuned finals for the four 1D LSTM groups (VanillaLSTM 1D and ResidualVanillaLSTM 1D, both bands), weights saved | running since 18:57 UTC on 11-07; at 01:40 UTC on 11-08 the two 2400-2600 groups are done (10 of 20 finals) and the ten 600-800 finals are still running (GPU 2.5 of 4 in use). Earlier estimate: Past 1D LSTM runs took 3.4-4.3 h (max 10-14 h), so my estimate is **finished between about 03:00 and 09:00 UTC on 11-08**, longer if one of the slow runs lands in the tail |

Finished on the A100 since the last snapshot: `claude-p15-1dlstm-watcher-1` (stopped the old 1D LSTM driver at 18:56 UTC once all four groups had 16 trial records and resubmitted it with the patched code as `-lstm-2`, as the user approved),
`claude-tune-p15-2` (tuned finals for every other group with weights; TCN has 4 of 5 per band, one final failed in each), `claude-final-eval-v3-4d-2` (4D evaluation; 126 objects, last write 02:21 UTC; results not yet read),
`claude-p15-sitecmp2-site` and `-pooled` (per-receiver and pooled comparison, 312 records), `claude-llm-ofa-p15-1` (12 OFA runs ok, no weights).
After the 1D LSTM finals the A100 is idle until its lease ends. It is not an H100, so it should not be used for the timing benchmark; it is free for other work.

The SSH tunnel to the A100 dashboard was down and had to be reopened (`ssh -F training/ray/docker/tacc-ssh.conf -o ServerAliveInterval=30 -fN -L 8267:localhost:8265 -L 9003:localhost:9001 ray-tacc-1`).

## 19. Checkpoints and artifacts in the bucket

Checked 2026-11-06 by listing `s3://spectrum` and reading each runner; counts refreshed 2026-11-07 23:30 UTC: `powder15-results-w/` 514 weight files, `tuning/powder15/` 89 weight files (growing), `llm-tuning-electrosense/` 100 (10.8 GB), `tsslcd-authors-full/` 30 (3.3 GB), `oct2026-band-runs/` 70, `tsl-checkpoints/` 650 (78.8 GB). Helper: `training/ray/artifacts.py` (`save_artifact`, `save_run`; tests in `tests/test_artifacts.py`), added in commit `de1b97e`
and synced to all seven hosts.

| what | weights saved | where |
|---|---|---|
| POWDER 1-minute 600-800 MHz campaign and DSwinLSTM round 2 | yes, with `run_costs.json` and `MANIFEST` | `checkpoints-oct2026/` (78 files), `checkpoints-oct2026-r2/` (20) |
| POWDER 1-minute 2400-2600 MHz, per-receiver, pooled-6 | yes (via `fixed_train`) | `oct2026-band-runs/` (54 weight files and 54 `run_costs.json` at 19:20 UTC, growing), `oct2026-per-receiver/` (48), `oct2026-pooled6/` (8) |
| Collaborator's reproductions | yes (only about 70 of 331 SpectrumLLM runs kept weights, by their own inventory) | `checkpoints/repro/` (312) |
| Earlier Ray tuning campaign | yes | `oct2026-minio-backup-newcluster/` (4,087 objects; last write 14:48 UTC) |
| POWDER 15-minute tuned finals | `claude-tune-p15-2` (patched): weights saved; 1D LSTM finals: only after `claude-tune-p15-1d-lstm` is restarted |
| POWDER 15-minute pooled models | **being rerun with weights**: `powder15-results-w/weights/` (102 files at 20:0x UTC, 514 runs expected); the original `powder15-results/` has none. Not covered: per-receiver/pooled comparison runs and tuned finals |
| ElectroSense (ours, tuned finals), POWDER Time-LLM/OFA, BERT reruns, TSS-LCD | **no for runs started before commit `de1b97e`** (metrics JSON only); **yes for runs started after it**: `<output prefix>/weights/...` with a `.meta.json` sidecar (resolved config, seed, code revision, host, sha256) | e.g. `tsslcd-firstcommit-results-w/weights/` (18 files), `tsslcd-authors-full/weights/`, `llm-tuning-electrosense/weights/` |
| TSL library baselines (PatchTST, DLinear, Informer, Autoformer, LightTS, plus the other defaults), ElectroSense | **recovered 2026-11-06** from the H100 worker containers (`/workspace/tsl/...`) by `training/ray/recover_tsl_checkpoints.py` (job `claude-tsl-recover-2`): all **650** finished default and tuned-final runs have a checkpoint, 78.8 GB, `tsl-checkpoints/<setting>/checkpoint.pth` + `meta.json`; search trials were not copied (their hyperparameters are in the trial records) | `tsl-checkpoints/` |
| `loss_units.py` | no (never saves weights) | |
| Finished runs whose scratch directory was deleted (15-minute POWDER pooled before the rerun, ElectroSense ours/tuned finals, POWDER Time-LLM/OFA smoke, BERT reruns, TSS-LCD reduced run and the first retrains, tuned 15-minute trials) | **not recoverable**: each ran inside a temporary directory that is removed when the run ends (checked 2026-11-06: no scratch directories of finished runs remain on any of the seven hosts); only a rerun brings weights back | |
| Tuning search trials (16 per model) | no, by design: only the winner's finals are saved; the search parameters are in the trial records | |
| Inference benchmark output | n/a | `inference-benchmark/` (smoke rows only so far) |
| **Per-bin metrics** (per frequency bin, per horizon: MAE, RMSE, MSE, MAPE linear and dBm) | **only the October final evaluation saved them** (123 files under `oct2026-results/final-eval-v3/**/per-bin/`). Every other evaluation (15-minute pooled and tuned runs, 2400-2600 band runs, ElectroSense LLM tuning, the `tune_direct` finals running now) kept only the aggregate metrics per horizon, because the per-bin table was written to the job's scratch directory and deleted with it (checked 11-08). Fix: `training/ray/rescore.py` now uploads the per-bin test table for every checkpoint it evaluates (`rescored/per-bin/<path>.csv`, 600 rows per 1-minute model = 200 bins x 3 horizons); `tune_direct.py` does too for any finals started after commit `4990771`. Per-bin results can be regenerated for any model **whose weights were saved**; models without weights (pre-patch ElectroSense, BERT, original 15-minute runs) cannot be | `rescored/per-bin/` |

**Jobs that predate the patch and will not save weights**: on the A100, the OFA, 15-minute tuning, per-receiver/pooled and 4D jobs; on the H100, `claude-tune-es-1`. Finished pre-patch results (BERT reruns,
the first TSS-LCD retrains, tuned library baselines, 15-minute and ElectroSense runs) have metrics only.

**Backups.** The H100 backup loop copies the Ray tuning state (MinIO prefix `spectrum-usage/ray/oct2026-core4*`) hourly to the durable bucket. There is no backup loop on the A100, and none is needed: a dry-run listing at 20:4x UTC found **0 objects** in its local MinIO, so nothing there lives only on the cluster (the A100 jobs write straight to `s3://spectrum`). Host-disk outputs are not backed up (the one known case, the level-diagnostics outputs on the H100
head, was copied to `runs/claude/level-diagnostics/`).

## 21. Direct-model tuning on POWDER (OFA, Time-LLM, PatchTST, DLinear, nonlinear residual head)

Started 2026-11-08 at the user's request, after the finding that models which beat ResidualLinearAR on ElectroSense (PatchTST, OFA, Time-LLM) had hardly been tried on POWDER. Goal: tune each with the **same budget as our core models**
(16 candidates per model, validation-selected, then seeds 40-44 on the final test range), 15-minute first, then 1-minute, with inference timing.

| item | state |
|---|---|
| Models wired into the pipeline: `models/PatchTST.py` (channel-independent, RevIN, patching), `models/ResidualNonlinearAutoregressive.py` (ResidualLinearAR plus an MLP correction head on each channel's deviations, last layer zero-initialised so it starts as the linear model), DLinear (existing port, now registered); 3 tests on the cluster | done |
| Runner `training/ray/tune_direct.py` (19 tests): trials seed 42 at a short and a long horizon (15-minute 4 and 48 steps, 1-minute 1 and 60), winner by mean validation objective written to `winners.json` before any test run, finals = winner x horizons (15-minute 4/12/24/48; 1-minute 1/15/60) x seeds 40-44, test range **2026-10-27T20:00Z to 2026-11-03T20:00Z** (not the preliminary 10-30 cutoff), all five metrics per horizon, weights and config uploaded per final; resumable (rerun the same command to redo failed runs) | done |
| Smoke test, all 5 models, trials + a final + weights (`direct-tuning-smoke/`) | done; `transformers` and GPT-2 had to be installed on the six new hosts first (`claude-llm-setup-newh100-1`) |
| 15-minute, cheap models: DLinear, PatchTST, residual head (192 trials then finals) | `claude-direct15-small-1` **running** (134 of 192 trials done, 0 failed at 02:15 UTC) |
| 15-minute OFA (64 trials then 40 finals) | `claude-direct15-ofa-1` **running** |
| 15-minute Time-LLM (64 trials then 40 finals; training windows thinned 2x as the user chose) | first submission (`-timellm-1`) failed with CUDA out-of-memory (one task needs 43-49 GB, two shared a 93 GB GPU) and was stopped; resubmitted as `claude-direct15-timellm-2` with a 0.6 GPU share, memory cap per task and peak-memory recording. **Cost is the open risk**: with one Time-LLM per GPU, 104 runs at a few hours each may not fit before the `ray-new-*` hosts expire (2026-11-09 13:40 UTC); decision pending the first measured run times |
| 1-minute, cheap models: DLinear, PatchTST, residual head (training and test windows thinned 15x, disclosed) | `claude-direct1min-small-1` **running** since 02:2x UTC (first trials 7-10 s each, no failures) |
| **Simple-baseline search (user request 2026-11-09): which cheap models match OFA/PatchTST?** Same protocol as the other direct models (16 candidates, validation-selected, seeds 40-44, all five metrics, weights and per-bin tables saved). Models (`tune_direct.py`, tests in `tests/test_tune_direct.py` and `tests/test_direct_models.py`, all passing, the latter on a worker): `residuallinearar2d` and `linearar2d` **tuned in direct mode** (the clean control for the MLP head; the registry's space: ridge alpha 1e-5 to 10, learning rate 1e-5 to 1e-2, batch 32/64/128), `nlinear` (linear with last-value normalization, channel-shared), `revinmlp` (per-window normalization + one MLP shared across channels: the two ingredients PatchTST and OFA share, without patching or attention), and PatchTST ablations with PatchTST's own 16 candidates and one ingredient removed: `patchtst_noinorm` (no per-window normalization), `patchtst_joint` (each patch spans all channels), `patchtst_nopatch` (one token per time step). Jobs: `claude-simple15-1` (15-minute POWDER, both bands, H100), `claude-simple1min-rl-1` (1-minute ResidualLinearAR direct), `claude-simple-es-1` (ElectroSense spec4 and spec25, A100, also adds the residual head, PatchTST and DLinear through the same pipeline so every ElectroSense number is comparable; metric raw MSE as in the paper). The tuner now supports `--resolution electrosense` and `--max-parallel`. Output `direct-tuning/<15min|1min|electrosense>/<band or dataset>/<model>/` | 
| **Time-LLM progress at 2026-11-08 23:30 UTC** (`claude-direct15-timellm-3`, `claude-direct1min-timellm-2`; six tasks running, one per H100) | 15 minutes: **47 of 64 trials done** (600-800 22 of 32, 2400-2600 25 of 32), median 38 minutes per trial, none failed; 1 minute: **40 of 64** (20 and 20 of 32), median 92 minutes, none failed; no finals yet (each band's finals launch when its 32 trials finish). Remaining work about 125 task-hours (15 minutes: 17 trials + 40 finals, about 39 h; 1 minute: 24 trials + 30 finals, about 85 h) against about 140 task-hours of capacity (6 GPUs until 11-09 13:40 UTC, then 2 until 11-10 18:00). Queue order is first come, first served, so the 15-minute finals start only after the remaining 1-minute trials are dispatched (about 04:00-05:00 UTC), and my estimate is that everything completes around **2026-11-10 00:00-06:00 UTC**, with about 10% slack; tasks on the four expiring nodes at 13:40 are retried from scratch on the survivors (a few task-hours lost) |
| **Loss if killed near the end (11-08)**: a run killed mid-way loses its elapsed time (results are saved only when a run finishes); finished runs are never lost. The runner originally waited for **all** trials before computing any winner, so a deadline during the trials would have left no test results; it now **pipelines each (model, band) group** (trials, then winner, then finals) and retries tasks lost to a dying node up to 3 times. Time-LLM was restarted on the new runner at 11-08 03:30 UTC (no trial had finished): `claude-direct15-timellm-3`, `claude-direct1min-timellm-2`. The OFA and cheap-model jobs still use the older runner (they finish within hours) | 
| **Language-model control (user approved 2026-11-08)**: does the pretrained GPT-2 matter? Two OFA variants with the **same architecture** (`models/OFA.py`: `backbone_init: random`, `backbone_trainable`), tuned with **exactly OFA's 16 candidates**, validation-selected, finals seeds 40-44, 15-minute, both bands: `ofa_rand` (random init, only layer norms and positional embeddings train, OFA's rule) and `ofa_scratch` (random init, every backbone weight trains, i.e. a transformer trained from scratch), compared with GPT-2 `ofa` and the tuned PatchTST. Jobs `claude-direct15-ofa-rand-1`, `claude-direct15-ofa-scratch-1` (queued 11-08 ~04:00 UTC; prefix `direct-tuning/15min/<band>/ofa_rand|ofa_scratch/`). This replaces the collaborator's control (a 256-wide untrained transformer, not capacity-matched, all-channel mode), and the SpectrumLLM paper itself contains no such ablation (read 11-08: case study only, no backbone ablation; its LLM-vs-PatchTST margin is 2-6% in its own tables, single run, no intervals). The extra OFA-size runs (about 1.5 h of cluster time) delay the start of Time-LLM. 1-minute controls not queued (little headroom there) | 
| **H100 capacity timeline for Time-LLM** (corrects an earlier note that treated 2026-11-09 13:40 UTC as the end for all six GPUs) | `ray-new-1..4` end 2026-11-09 13:40 UTC; the head (`ray-worker-2`, where the drivers run) and `ray-worker-1` end **2026-11-10 18:00 UTC**. Capacity is about 204 H100-hours (6 GPUs, about 34 h) plus about 56 (2 GPUs, about 28 h) = about 260. Rough Time-LLM need (scaled from a smoke test, to be refined): about 87 H100-hours at 15 minutes and about 173 at 1 minute. The A100s cannot host Time-LLM (43-49 GB needed, 40 GB available). Tasks running on the four expiring nodes at the deadline are retried once on the survivors from scratch. 15-minute Time-LLM should fit; 1-minute is borderline |
| 1-minute OFA and Time-LLM | **queued 11-08 ~02:50 UTC** to keep the H100s busy until the lease ends: `claude-direct1min-ofa-1`, `claude-direct1min-timellm-1` (64 trials each, then finals; the finals phase starts only after all trials of a model finish, so on the Time-LLM side finals may be cut by the lease; finished runs are saved) |
| Rescoring on one test range with per-bin tables (`training/ray/rescore.py`, output `rescored/` and `rescored/per-bin/`; restarted 11-08 ~03:10 UTC as `-p15tuned-3`, `-band-3`, `-oct-missing-2` to add the per-bin upload). **Still to run once training ends: `--sources direct` (OFA, Time-LLM, PatchTST, DLinear, residual-head finals; their per-bin tables do not exist yet because the running H100 drivers use the earlier code) and `--sources p15` (inherited 15-minute runs)**: tuned 15-minute finals (`claude-rescore-p15tuned-2`), all 2400-2600 checkpoints (`claude-rescore-band-2`), the 25 missing October checkpoints plus duplicates (`claude-rescore-oct-missing-1`; 1-minute 1D LSTMs thinned 10x) | **running on the A100 cluster** since 11-08 ~02:40 UTC (first attempt hit the numpy mismatch, see 22.7) |
| 4D lookback-mean baseline, both bands, seed 42, stride 1, final range (`claude-lookbackmean4d-600_800`, `-2400_2600`, output `oct2026-lookbackmean4d/`) | **running on the A100 cluster** |
| Rescoring our tuned 15-minute finals (ResidualLinearAR 2D etc.) on the full test range, so the baseline is on the same windows | **not started**; proposed, not yet approved |
| Inference timing of every new final (`inference_benchmark.py --sources direct`, needs idle GPUs) | code ready and tested; run after training |

Output prefix `direct-tuning/<resolution>/<band>/<model>/{trials,finals,weights,winners.json}` in `s3://spectrum`.
Disclosed deviations from our core-model protocol: training-window thinning (Time-LLM 2x at 15 minutes; all models 15x at 1 minute), 1-minute test windows thinned 15x, and for the LLM models a smaller batch (4 windows of 200 channels) and 10 epochs.

## 22. Weekend results, 2026-11-06 to 2026-11-08 (analyzed from `s3://spectrum` at 02:30 UTC on 11-08)

Everything in this section was read from result files in the bucket; where a number mixes test ranges or protocols that is stated. MAE is in dB unless noted.

### 22.1 Headline findings

1. **TSS-LCD (authors' model, full 300/300/2000-epoch schedule, 800 POWDER columns, 5 seeds) is worse than the lookback mean on raw measurements** (1.152-1.166 dB vs 0.950), and the savgol filter makes no difference there (1.164 without, 1.166 with). Training the full schedule helps a lot compared with the reduced run (1.69 / 1.55 to 1.16), but it never beats a trivial predictor. 22.2.
2. **Tuned OFA and Time-LLM on ElectroSense tie ResidualLinearAR at 4 channels and beat it by 13-15% at 25 channels** (OFA 0.279 / 0.817, Time-LLM 0.275 / 0.838, vs ResidualLinearAR 2D 0.277 / 1.013). Tuning itself moved little: the best candidate's validation error is only 1-5% below the default's, and the validation-chosen OFA is 5.7% worse on test than the collaborator's untuned OFA (0.264). 22.3.
3. **At 15 minutes, direct models beat rolled-out ones for almost every model** (inherited hyperparameters, 5 seeds): by 2-3% for ResidualLinearAR, 5-11% for the 2D LSTMs, and 15-29% for the 1D LSTMs. The best direct models are the 1D ResidualLinearAR (0.775 / 0.642), the 1D ResidualVanillaLSTM (0.769 / 0.649) and ResidualLinearAR 2D (0.788 / 0.658). 22.4.
4. **OFA, untuned, on 15-minute POWDER 600-800 beats the direct ResidualLinearAR 2D at every horizon, by 4.2-5.9%** (3 seeds, A100). This is the first direct evidence that the models that win on ElectroSense also win on POWDER. 22.4.
5. **Pooled vs per-receiver on identical windows (15 minutes):** ResidualLinearAR 2D is a tie (+1.0% and -2.2% for per-receiver); pooling is better for LinearAR 2D (+9% and +14% for per-receiver), LSTMAttn (+13%, +7%) and VanillaLSTM 2D in 2400-2600 (+9%); the **1D ResidualVanillaLSTM is better per receiver** (-20% and -6%). 22.4.
6. **Tuned 15-minute finals: no nonlinear model beats ResidualLinearAR 2D; only the 1D linear models do (by 2-5%)**; the 2D LSTMs and AutoformerCSA are 30-93% worse even after tuning. 22.4.
7. **4D (1-minute, 600-800):** the linear 4D models are best on the 4D map metric (ResidualLinearAR 4D 0.719, LinearAR 4D 0.728, DSwinLSTM-I round 1 0.918), but 4D MAE is over map cells and is **not comparable** with the 2D/1D numbers; the 4D lookback mean was not scored in this run. 22.5.
8. **A comparability hazard in the 2400-2600 results:** the extra linear seeds and the four "missing" models were scored to 10-30 18:00, the rest to 11-03 20:00. Rescoring on one range is needed before any cross-model table. 22.5.
9. **Inference time (H100, idle GPUs, 159 models):** per receiver spectrum at batch 32 the fastest GPU models take 0.17-0.46 ms (AutoformerCSA, VanillaLSTM, ResidualVanillaLSTM 2D), 4D models 1.9-11 ms, and the 1D models 14-88 ms because they score one bin at a time (200 calls per spectrum). The six linear models were timed on CPU, not GPU. 22.6.
10. **Operational:** one Time-LLM task needs 43-49 GB of GPU memory at the pipeline's batch size, so two cannot share an H100; the processed POWDER files kept growing after Nov 3 (pure appends, existing results unaffected); 78.8 GB of TSL checkpoints were recovered from worker disks before the old cluster expired. 22.7.

### 22.2 TSS-LCD, authors' code, full schedule, POWDER 800 columns (`tsslcd-authors-full/`, 5 seeds each, 2.1-2.4 h per run on a shared GPU)

Test MAE (mean over t+1, t+15, t+60 minutes), full final test range at stride 1 (9,888 windows); lookback mean 0.950, ResidualLinearAR 2D 0.924.

| savgol on the data | eval-time averaging | scored vs raw measurement | scored vs the model's own target (smoothed by the same savgol) |
|---|---|---|---|
| off | none | **1.164** (sd 0.006) | 1.164 (the target is the raw series) |
| off | centred 6:3 / trailing 6:3 | 1.152 / 1.153 | 0.916 / 0.917 (target smoothed by the 3-sample average) |
| centered (51, 2) | none | **1.166** (sd 0.006) | 0.755 (sd 0.007) |
| centered | centred 6:3 / trailing 6:3 | 1.156 / 1.157 | 0.739 / 0.740 |

By horizon (savgol off, no averaging, vs raw): t+1 1.148, t+15 1.171, t+60 1.174. Reduced run for comparison (5% of the epochs, one seed): 1.692 (off) and 1.551 (centered) vs raw.
Cost: context autoencoder (75.8M parameters) 11.8 s per epoch (3,533 s for 300), future autoencoder 0.42 s per epoch, diffusion 3.17 s per epoch (6,342 s for 2,000), inference 66 s for 9,888 windows; weights are in `tsslcd-authors-full/weights/`.
**Findings.** (a) Against raw measurements the model is 22-23% worse than the lookback mean in every configuration; the savgol filter on the training data does not help the raw-measurement score, unlike in the reduced run. (b) The only scores that look good are those against a smoothed target (0.74-0.92), and even those are not clearly better than the lookback mean on the raw series. (c) Eval-time averaging of the predictions helps by about 0.01 dB against raw measurements and by 0.016 dB against the filtered target (savgol centered: 0.755 to 0.739). The jump from 1.164 to 0.916 with savgol off is a change of target (the averaged ground truth), not a model gain.

### 22.3 OFA and Time-LLM, equal-budget tuning on ElectroSense (`llm-tuning-electrosense/`, 16 candidates, validation-selected at horizons 12 and 192, finals 5 horizons x 5 seeds)

| dataset | model | winner (candidate; lr, layers, patch/stride, d_model, d_ff) | validation (std units), winner vs default | **test raw MSE, mean over horizons** | by horizon 12/24/48/96/192 |
|---|---|---|---|---|---|
| spec4 | OFA | 11; 1.59e-4, 3 layers, 32/32 | 0.2704 vs 0.2845 | **0.279** (seed sd 0.007) | 0.247 / 0.263 / 0.271 / 0.291 / 0.325 |
| spec4 | Time-LLM | 14; 2.93e-4, 6 layers, 8/8, 16, 32 | 0.2706 vs 0.2757 | **0.275** (sd 0.005) | 0.246 / 0.270 / 0.272 / 0.280 / 0.307 |
| spec25 | OFA | 11; 1.59e-4, 3 layers, 32/32 | 0.3979 vs 0.4020 | **0.817** (sd 0.009) | 0.581 / 0.722 / 0.823 / 0.930 / 1.029 |
| spec25 | Time-LLM | 5; 2.40e-4, 6 layers, 32/16, 64, 32 | 0.4007 vs 0.4085 | **0.838** (sd 0.009) | 0.594 / 0.736 / 0.844 / 0.954 / 1.063 |

References (raw MSE, mean over horizons): ResidualLinearAR 2D 0.275 / 0.277 tuned on spec4 and 1.022 / 1.013 on spec25; paper OFA 0.299 / 0.846, paper Time-LLM 0.307 / 0.849; collaborator's untuned OFA 0.264 / 0.809 and Time-LLM 0.284 / 0.825; tuned PatchTST 0.311 / 0.849; lookback mean 0.804 / 1.493.
Search spread: validation means over the 16 candidates range 0.270-0.309 (OFA spec4), 0.271-0.308 (Time-LLM spec4), 0.398-0.409 and 0.401-0.418 (spec25), so the choice mattered little.
**Findings.** (a) At 4 channels tuned Time-LLM ties ResidualLinearAR (0.275 vs 0.275-0.277) and OFA is 1% behind; at 25 channels both beat ResidualLinearAR by 13-15% and tuned PatchTST by 1-4% (OFA 0.817 vs PatchTST 0.849). (b) Tuning gains are small, and a validation-chosen setting did not beat the collaborator's untuned regime on test (OFA 0.279 vs 0.264, about 2 seed standard deviations), so tuning under our protocol did not improve on the collaborator's untuned regime. (c) The default candidate was not separately scored on test, so the test-level gain from tuning is not measured.

### 22.4 POWDER 15-minute (test 2026-10-27 20:00 to 11-03 20:00 unless stated)

**Inherited hyperparameters, pooled four receivers, 5 seeds, rolled-out vs direct** (`powder15-results-w/`, all 514 runs ok, weights saved). MAE mean over horizons 1/3/6/12 h; lookback mean 1.238 (600-800) and 0.987 (2400-2600).

| model | 600-800 rolled-out | 600-800 direct | 2400-2600 rolled-out | 2400-2600 direct |
|---|---|---|---|---|
| ResidualLinearAR 1D | 0.801 | **0.775** (-3.3%) | 0.657 | **0.642** (-2.3%) |
| LinearAR 1D | 0.801 | 0.773 (-3.5%) | 0.663 | 0.652 (-1.8%) |
| ResidualLinearAR 2D | 0.803 | 0.788 (-1.8%) | 0.680 | 0.658 (-3.2%) |
| LinearAR 2D | 0.936 | 0.846 (-9.6%) | 0.760 | 0.674 (-11.3%) |
| ResidualVanillaLSTM 1D | 0.998 | **0.769** (-22.9%) | 0.760 | 0.649 (-14.6%) |
| VanillaLSTM 1D | 1.201 | 0.856 (-28.7%) | 0.959 | 0.785 (-18.2%) |
| LSTMAttn 2D | 1.048 | 0.980 (-6.5%) | 0.953 | 0.882 (-7.5%) |
| ResidualVanillaLSTM 2D | 1.135 | 1.073 (-5.5%) | 0.948 | 0.890 (-6.2%) |
| VanillaLSTM 2D | 1.284 | 1.142 (-11.0%) | 0.997 | 1.072 (**+7.5%**) |
| AutoformerCSA 2D | 1.306 | 1.184 (-9.3%) | 1.238 | 1.014 (-18.2%) |
| TCN 2D (rolled-out only) | 1.241 | - | 1.132 | - |

Direct is better in 19 of 20 cells; the largest gains are for the 1D LSTMs, which are poor when rolled out. The direct 1D ResidualVanillaLSTM (0.769) matches the direct 1D ResidualLinearAR (0.775) and edges ResidualLinearAR 2D (0.788) at 600-800 with untuned inherited settings.

**OFA on 15-minute POWDER, 600-800, untuned, 3 seeds** (A100, `powder-llm-results/`, 0.52-1.0 h per run), MAE per horizon (steps; 4 = 1 h, 48 = 12 h):

| horizon | OFA | ResidualLinearAR 2D direct | ResidualLinearAR 1D direct | ResidualVanillaLSTM 1D direct | lookback mean |
|---|---|---|---|---|---|
| 4 | **0.585** | 0.612 (OFA -4.4%) | 0.597 | 0.584 | 1.134 |
| 12 | **0.718** | 0.763 (-5.9%) | 0.758 | 0.743 | 1.220 |
| 24 | **0.817** | 0.853 (-4.2%) | 0.843 | 0.846 | 1.360 |
| 48 | **0.873** | 0.924 (-5.5%) | 0.901 | 0.905 | 1.238 |

OFA has no 2400-2600 runs and no weights saved (it predates the patch). The 4-step gap is within about 1.5 seed standard deviations; the longer horizons are clearer.

**Per-receiver vs pooled on identical test windows** (`sitecmp2`, inherited hyperparameters, rolled-out, 4 receivers x 5 seeds; pooled model scored on each receiver vs that receiver's own model):

| model | 600-800 pooled | per-receiver | 2400-2600 pooled | per-receiver |
|---|---|---|---|---|
| ResidualLinearAR 2D | 0.802 | 0.809 (+1.0%) | 0.680 | 0.666 (-2.2%) |
| LinearAR 2D | 0.936 | 1.018 (+8.8%) | 0.759 | 0.866 (+14.0%) |
| LSTMAttn 2D | 1.036 | 1.173 (+13.2%) | 0.915 | 0.975 (+6.5%) |
| ResidualVanillaLSTM 2D | 1.139 | 1.182 (+3.8%) | 0.949 | 0.979 (+3.1%) |
| VanillaLSTM 2D | 1.286 | 1.266 (-1.5%) | 0.998 | 1.085 (+8.7%) |
| ResidualVanillaLSTM 1D | 0.999 | **0.796 (-20.3%)** | 0.740 | **0.698 (-5.6%)** |

For the 2D models pooling usually helps or ties (the more parameters per receiver the more it helps); the 1D residual LSTM, which is shared across bins, gains from per-receiver training.

**Tuned finals** (`tuning/powder15/`; 5 seeds; **test range 10-27 20:00 to 10-30 18:00, not comparable with the tables above**; compare only with each other). MAE mean over horizons; RL2D = ResidualLinearAR 2D.

| model | 600-800 | vs RL2D | 2400-2600 | vs RL2D |
|---|---|---|---|---|
| ResidualLinearAR 1D | **0.678** | -3.6% | **0.567** | -5.1% |
| LinearAR 1D | 0.687 | -2.3% | 0.570 | -4.7% |
| LinearAR 2D | 0.703 | 0.0% | 0.601 | +0.5% |
| **ResidualLinearAR 2D** | 0.703 | - | 0.598 | - |
| TCN 2D (4 seeds) | 0.730 | +3.8% | 0.602 | +0.7% |
| ResidualVanillaLSTM 1D | running | - | 0.606 | +1.4% |
| VanillaLSTM 1D | running | - | 0.693 | +16% |
| VanillaLSTM 2D | 0.915 | +30% | 0.941 | +57% |
| LSTMAttn 2D | 0.925 | +32% | 1.019 | +70% |
| ResidualVanillaLSTM 2D | 1.004 | +43% | 1.155 | +93% |
| AutoformerCSA 2D | 1.314 | +87% | 1.137 | +90% |

No nonlinear tuned model beats ResidualLinearAR 2D; the best are the 1D linear models by 2-5% (above the seed spread of 0.01-0.02). Tuning helped the LSTMs strongly (2D VanillaLSTM at 600-800: 61% worse than ResidualLinearAR 2D untuned, 30% tuned) but did not close the gap. The 600-800 1D LSTM finals (10 runs) were still training on the A100 at 02:30 UTC.

### 22.5 POWDER 1-minute

**600-800, final range, stride 1** (`oct2026-results/final-eval-v3/`, 5 metrics each: MAE, MSE dB^2, RMSE, MAPE in linear power and in dBm): ResidualLinearAR 2D 0.924 (RMSE 1.819, MAPE dBm 0.83%), ResidualLinearAR 1D 0.926, LinearAR 1D 0.928, LinearAR 2D 0.930, TCN 2D 0.930, ResidualVanillaLSTM 2D round 2 0.944 (round 1 1.056), AutoformerCSA 2D 0.956, LSTMAttn 2D 1.084 / 1.094 (rounds 1 / 2), VanillaLSTM 2D 1.103 / 1.128. Lookback mean 0.950. **Not scored yet (25 checkpoints):** ConvLSTM x5, ResidualConvLSTM x5, DSwinLSTM round 2 x5, 1D VanillaLSTM x5, 1D ResidualVanillaLSTM x5.

**4D models** (same job; the models use the four core receivers (humanities, madsen, moran, sagepoint) on a 10x10 map (`oct2026_core4_*`), and the aggregate error averages **all 100 map cells**, of which only four are receiver locations (the rest are interpolated), so it is not comparable with the 2D/1D numbers above; per-receiver rows (error at the cell nearest each receiver) are computed separately by the evaluator and would be the comparable measure; the 4D lookback mean for the same range is being scored, `oct2026-lookbackmean4d/`): ResidualLinearAR 4D **0.719** (seed sd 0.000, RMSE 1.404, MAPE dBm 0.64%), LinearAR 4D 0.728 (RMSE 1.404), DSwinLSTM-I round 1 **0.918** (3 seeds, RMSE 1.652). The 4D linear models are 20% better than the 4D DSwinLSTM.

**2400-2600** (`oct2026-band-runs/_results`, MAE mean over 1/15/60 minutes; **ranges mixed, see the warning in section 2**): 4D LinearAR 0.635 and ResidualLinearAR 0.636 (seeds 40/41/43/44, short range), ResidualConvLSTM 0.647 and ConvLSTM 0.676 (seed 42, short range), TCN 2D 0.743, ResidualVanillaLSTM 1D 0.754 (5 seeds), ResidualVanillaLSTM 2D 0.761, ResidualLinearAR 2D 0.776 (short range; 0.744 at seed 42 on the full range), ResidualLinearAR 1D 0.779, LSTMAttn 2D 0.779, LinearAR 1D 0.780, LinearAR 2D 0.781, AutoformerCSA 2D 0.782, 1D VanillaLSTM 0.796 (short range), VanillaLSTM 2D 0.798. Seed spreads are 0.000-0.011. The GPU models now have seeds 40-44; 15 seed-42 checkpoints have no per-run record under `_results/`.
Per-receiver (48) and pooled-6 (8) models remain scored on the preliminary 10-30 cutoff.

### 22.6 Inference time (H100, one model per idle GPU, 159 models, batch sizes 1 and 32; `inference-benchmark/h100/inference_benchmark_20261108T0050Z.csv`)

Median milliseconds per **receiver spectrum** (1D models need 200 calls per spectrum; 4D windows count as one spectrum each, an assumption; 4D windows number about 496 per run, i.e. one per time step). 1-minute 600-800, test windows thinned (stride 20; 1D 400):

| model | ms per spectrum, batch 1 | batch 32 | parameters | device |
|---|---|---|---|---|
| AutoformerCSA 2D | 5.3 | **0.17** | 159,188 | GPU |
| VanillaLSTM 2D (r1) | 10.1 | 0.40 | 194,760 | GPU |
| ResidualVanillaLSTM 2D | 13.8 | 0.46 | 8,520 | GPU |
| LinearAR 2D* / ResidualLinearAR 2D* | 2.4 / 3.9 | 0.27 / 0.39 | 12,200 | CPU* |
| LSTMAttn 2D | 25.1 | 1.0 | 487,752 | GPU |
| TCN 2D | 26.2 | 1.3 | 713 | GPU |
| DSwinLSTM-I 4D | 62 | 1.9 | 308,348 | GPU |
| ResidualConvLSTM 4D / ConvLSTM 4D | 184 / 339 | 5.9 / 10.8 | 589,416 / 3,907,016 | GPU |
| LinearAR 1D* / ResidualLinearAR 1D* | 368 / 608 | 13.6 / 21.6 | 61 | CPU* |
| LinearAR 4D* / ResidualLinearAR 4D* | 15.0 / 16.5 | 28.3 / 41.9 | 1,220,000 | CPU* |
| VanillaLSTM 1D / ResidualVanillaLSTM 1D | 2,139 / 2,709 | 73 / 88 | 361 / 17,217 | GPU |

15-minute pooled, 600-800, horizon 12, batch 32 (ms per spectrum): ResidualVanillaLSTM 2D 0.013, VanillaLSTM 2D 0.015, LinearAR 2D* 0.059, LSTMAttn 0.109, AutoformerCSA 0.178, LinearAR 1D* 0.32, VanillaLSTM 1D 1.3, ResidualVanillaLSTM 1D 1.7. **\* Linear models (LinearAR and ResidualLinearAR, 1D/2D/4D) are timed on CPU** (their configs set `device: cpu`, so they ran on the H100 hosts' CPUs); every other model ran on the GPU. The user accepted this as is (2026-11-08); mark every timing table that includes them with the asterisk. For the same architecture the two bands time identically (median ratio 1.00-1.01); tuned finals differ between bands because the tuning chose different hyperparameters (for example 520,392 vs 114,376 parameters for the 2D VanillaLSTM, a 5x time difference). All rows `idle=True`, GPU NVIDIA H100. The OFA, Time-LLM, PatchTST, DLinear and residual-head finals are not timed yet (section 21).

### 22.7 Process and infrastructure findings

- **New H100 cluster** (6 hosts, up 11-07 night): code synced and checksum-verified on all hosts, restored MinIO has all 4,087 objects, POWDER, ARA, COSMOS and AERPAW data (211 objects, 4.1 GB) on every host. `ray-new-1..4` expire 2026-11-09 13:40 UTC.
- **Data kept growing after Nov 3:** 14 POWDER files (all receivers, both bands, run `20261025T1355Z`, and `web-nuc1/20261030T1823Z`) grew by about 12 MB each; all 14 were verified to be pure appends, so existing results (test range ending 11-03 20:00) are unaffected.
- **GPU memory:** one Time-LLM task needs 43-49 GB at batch 4 x 200 channels; two sharing a 93 GB H100 ran out of memory. Time-LLM now gets a 0.6 GPU share and every task is memory-capped to its share.
- **Weights:** all 15-minute pooled models were rerun with weights saved (514), TSS-LCD full (30) and the ElectroSense LLM finals (100) save weights, and the TSL baselines' 650 checkpoints (78.8 GB) were recovered from the old workers' disks; pre-patch ElectroSense, BERT and 15-minute (original) runs have metrics only.
- **`ray-tacc-4` has a different numpy from the other A100 workers** (1.25.2 vs 2.1.3), although all four containers run the same image (`spectrum-ray:2.54.0-tacc-844647d4fa55`, created 11-03). Cause: per-user `pip install --user` of `gdown`, `timesfm` 3.0.2 and `momentfm` 0.1.4 inside that one container on 2026-11-04 between 00:10 and 00:41 UTC (the TimeRAN, TimesFM-3 and MOMENT preliminary tests); `momentfm` pins numpy 1.25.2, transformers 4.33.3 and tokenizers 0.13.3, and these live in `/root/.local`, which shadows the image's packages for every Python process there. Effect: checkpoints pickled under numpy 2 cannot be loaded on that host (the first rescoring jobs failed there with `No module named 'numpy._core'`); the rescoring job now probes node numpy versions and avoids the odd one. Not changed: the container was left as it is. It is lost if the container is recreated; a proper fix is to move such installs to an isolated directory (as `/workspace/pylibs` does for transformers) or into the image.
- **A100:** a 1D LSTM tuning driver was restarted automatically by a watcher once its trials finished, so its finals save weights; its tunnel and the H100's dropped when the PC was closed and were reopened.

## 20. Risks and open decisions

| item | detail |
|---|---|
| Lease deadlines | The old H100 lease ended 2026-11-07 16:58 UTC with all its jobs finished. A100 ends 2026-11-10 18:00 UTC if "6 p.m." was UTC; 23:00 UTC if Eastern (unconfirmed). The new H100 cluster is being brought up |
| Work lost at the H100 deadline | none known: every H100 job finished and its results and weights are in the bucket |
| Local-only work | my commits (`de1b97e`, `c7adfeb`, `029b464` and the earlier unpushed ones) are not pushed; pushing would also publish the other session's `level_diagnostics` commits (`a54a1c0`, `031af6c`, `5c3d766`). This document is untracked |
| Level-diagnostics code on the clusters | synced with `--revision HEAD`; a later sync from `origin/integrate` would remove it |
| A100 backup | not needed (its MinIO is empty, checked) |
| Duplicate runs | the band-seed job redoes about 7 seeds that exist; they overwrite equivalent runs (retraining is not bit-reproducible, so those seeds' numbers will change slightly) |
| Data collection is still running | the processed POWDER files for run `20261025T1355Z` (all receivers) and `web-nuc1/20261030T1823Z` grew by about 12 MB each between 11-03 and 11-07; all 14 changed files were verified to be pure appends (the old file is an exact prefix of the new one), so every existing result, whose test range ends 11-03 20:00, is unaffected. The A100 hosts still hold the 11-03 copies |
| Pre-patch results | metrics only; the 15-minute pooled models are being rerun for weights (user decision, 2026-11-06); the TSL library baselines were recovered from worker disk; ElectroSense (ours), BERT and the per-receiver comparison are not recoverable and not rerun |
| Untuned LLM baselines | until 14.1 finishes, the "our model vs OFA/Time-LLM" comparison is not matched |
| Fragmented fractional GPUs | 0.5-GPU requests can starve while 0.25 slices are scattered; new fill jobs are submitted at 0.25 GPU |
| **A100 job drivers were killed by an out-of-memory event on 2026-11-08 ~03:03-03:08 UTC** | My rescoring jobs (1-minute 4D and 1D checkpoints at stride 1, launched three at a time with no memory reservation) pushed the 250 GB head node to 237.8 GB (95%); Ray's monitor killed the most recently scheduled workers, which were the job supervisors, so **every driver on that node failed**: the 1D LSTM tuning finals (about 8 hours of in-flight 600-800 finals lost; 10 of 20 finals, the 2400-2600 ones, had finished and were kept), the two 4D lookback-mean jobs, and all rescoring jobs. Fix: the 1D LSTM job was resubmitted at 03:12 (`claude-tune-p15-1d-lstm-3`, resumes the 10 missing finals); `rescore.py` now reserves memory per task (110 GB for 1-minute 4D, 45 GB for 1-minute 1D linear, 20 GB for other 1-minute 1D, 6 GB otherwise) and caps tasks in flight; rescoring resubmitted as `-p15tuned-4`, `-band-4`, `-oct-missing-3`; the 4D lookback-mean jobs still to be resubmitted once the heavy rescoring finishes. Lesson (it was already in the October runs): 4D evaluation needs one task per GPU |
| **Queue starvation on the A100 (11-08, found 23:30 UTC)** | Restarting the spectral-context ablation queued 1,179 tasks at once; Ray schedules same-shape tasks first come, first served, so every new rescoring task landed behind them: only 4 rescoring checkpoints finished in 6 hours although each takes under 6 minutes (4D ones 40-75). Fix: the ablation runner now caps cells in flight (`--max-parallel`); both ablation jobs were stopped and resubmitted (`claude-spectral-1min-3` with 8, `claude-spectral-15min-2` with 4; finished cells are skipped, so little was lost). Rescoring (`-p15tuned-4`, `-band-4`, `-oct-missing-3`, `-direct-1`, `-p15-1`) then gets the remaining slots. **Rule for the future: no job should submit more tasks than it can run soon** |
| `ray-tacc-4` environment differs from the image (numpy 1.25.2 and others from a 11-04 `pip install --user`) | tasks that unpickle numpy-2 checkpoints fail there; avoided by the rescoring job; any new job that loads checkpoints on the A100 should do the same or the container should be cleaned |
| Tuned 15-minute finals and weights | `claude-tune-p15-2` saves weights; `claude-tune-p15-1d-lstm` is restarted automatically by `claude-p15-1dlstm-watcher-1` once its trials are recorded (a `python -m` driver ships its functions by value, so a code patch does not reach a running driver). If the watcher fails, those finals have metrics only. TCN has only 10-11 successful trials of 16 recorded per band (not diagnosed) |

## 23. Handoff, 2026-11-09 04:30 UTC (historical; the current state is in 24.6)

**Watchdogs** (`training/ray/job_watchdog.py`, job lists in `training/ray/watchdog/{uc,tacc,h100}.json`; jobs `claude-watchdog-{uc,tacc,h100}-1`): every 10 minutes resubmit a tracked job whose latest submission FAILED (as `<id>-r1`, `-r2`, ...; 3-6 restarts); never resubmit SUCCEEDED or deliberately STOPPED jobs. All tracked drivers are resumable (finished work is skipped from the bucket). To stop a job for good, stop its watchdog entry's newest submission (STOPPED is respected) and tell the watchdog nothing.

| Cluster | Running (all unattended) |
|---|---|
| H100 (6 GPUs; 4 expire 11-09 13:40 UTC, 2 on 11-10 18:00) | Time-LLM tuning: `claude-direct15-timellm-3`, `claude-direct1min-timellm-2`; expected done 11-10 09:00-12:00 UTC; in-flight tasks on the expiring nodes restart from scratch (about 2-6 GPU-hours) |
| A100 (4 GPUs, lease ends 11-10 18:00 UTC) | spectral-context ablations, 3 seeds only (`claude-spectral-1min-4`: 275 cells left; `claude-spectral-15min-3`: 340 left; will not finish before the lease ends, seeds 43-44 can be added later: the runner skips finished cells); median study `claude-median-full-1` (CPU, 64 variant runs, results under `median-study/`; sweeps already done: tuned median beats the lookback mean by 3.2-4.4% at 1 minute and ResidualLinearAR 2D, see findings below) |
| UC RTX 6000 (8 GPUs) | 1-minute spatial evaluation `claude-uc-spatial-oct-2` (frozen October 4D checkpoints, 6 conditions, stride 1; 103 linear cells done, 150 neural cells left, whole-band numbers via `evaluation/analysis/spatial_whole_band.py`); 15-minute 4D tuning on the default core-four map (`claude-uc-tune4d-linear-1`, `-neural-1` ResidualConvLSTM + ConvLSTM, `-sts-1` STS-PredNet; results `tuning/powder15_4d/<band>/4d/<model>/`; 16 candidates per band then seeds 40-44); TimeRAN head-only tuning `claude-uc-timeran15-2` (`timeran_lp`, `moment_lp`, 15 minutes; full fine-tuning and random-init variants dropped as too expensive and redundant with the OFA controls) |

**Decisions made 2026-11-09:** ablations cut to 3 seeds (missing seeds later); the 15-minute spatial job with inherited hyperparameters was stopped (4 finished runs labelled "inherited" in `spatial15-oct2026/README.txt`; tuned runs will go under `spatial15-tuned-oct2026/`, runner support for loading the tuned winners is still to be written); TimeRAN restricted to head-only; STS-PredNet uses the recent-history branch only with `lc` equal to the lookback (the daily-history experiment stays separate and is not run yet).

**Findings so far (all unreviewed):** (1) the tuned 20-30 minute median beats the 60-minute lookback mean by 3.2% (600-800) and 4.4% (2400-2600) at 1 minute, and beats ResidualLinearAR 2D (0.919 vs 0.924; 0.734 vs 0.744); the level-diagnostics numbers are not directly comparable with the tables (plain 60-minute mean vs the rolled-out one-step mean, about 1% apart). (2) At 15 minutes a per-horizon median/mean choice gives 0.981 / 0.870 vs the lookback mean 1.184 / 0.982, far behind the trained models (0.809 / 0.688). (3) 1-minute 4D at the core-four receivers: whole-band error about 1.0 dB for LinearAR 4D, ResidualLinearAR 4D and the 4D lookback mean; held-out receivers are 4.7-6.0 dB when left out of the map and 1.5-1.7 when added. (4) The permuted-geometry test degrades the parameter-free lookback mean as much as the trained models, so it measures reading another receiver's value, not geometry use; interpret it relative to the baseline. (5) 15-minute 4D with inherited hyperparameters: linear 4D 0.70 dB on the full map, 0.91 at the receivers (lookback mean 1.17 and 1.22).

**Open decisions for the user:** what the new H100s (15:30 UTC) should do (Time-LLM 1-minute work, inference timing, or the 1-minute 2400-2600 re-tuning of the 2D family and linear models: up to about 80 GPU-hours); (the local commits were pushed to `origin/integrate` at 537e31b on 2026-11-09 at the user's request, so cluster syncs from `origin` are now safe; untracked files such as this document are still not in git); kriging, the receiver-count experiment, the transfer adaptation ladder and the week-2 drift evaluation are planned (`training/SPATIAL_PLAN.md`) but not queued.

## 24. Results of jobs finished 2026-11-09 (analyzed from `s3://spectrum` at 04:50 UTC)

All numbers are test MAE etc. in dB on the common range (2026-10-27 20:00 to 11-03 20:00 UTC), mean over horizons and, for the direct models, over seeds 40-44; "sd" is the seed standard deviation of the MAE. Direct models are trained per horizon (15 minute: 4/12/24/48 steps; 1 minute: 1/15/60 minutes). Hyperparameters: 16-candidate search per (resolution, band, model), validation-selected, finals on seeds 40-44. **1-minute finals here are scored on test windows thinned 15x** (disclosed; stride-1 rescoring of these exists as `rescored/direct/1min/`, 150 records, not analyzed yet).

### 24.1 Cheap and direct models on 15-minute POWDER (tuning jobs `claude-simple15-1` and the earlier direct jobs)

**600-800 MHz**

| Model | MAE dB (sd) | MSE dB2 | RMSE dB | MAPE linear % | MAPE dBm % | median final (min) |
|---|---:|---:|---:|---:|---:|---:|
| RevIN-MLP | 0.744 (0.001) | 1.97 | 1.392 | 18.50 | 0.667 | 0.1 |
| OFA (GPT-2) | 0.749 (0.003) | 2.01 | 1.406 | 18.83 | 0.672 | 13.8 |
| OFA, random-init, all weights train | 0.751 (0.005) | 2.02 | 1.408 | 19.33 | 0.674 | 26.9 |
| PatchTST | 0.755 (0.005) | 2.02 | 1.408 | 19.37 | 0.677 | 0.9 |
| OFA, random-init backbone (norms train) | 0.759 (0.006) | 2.03 | 1.412 | 18.64 | 0.681 | 4.2 |
| PatchTST, no patching (patch 1) | 0.761 (0.005) | 2.03 | 1.412 | 19.28 | 0.682 | 2.0 |
| ResidualLinearAR + MLP head | 0.764 (0.002) | 2.06 | 1.423 | 21.18 | 0.683 | 0.1 |
| NLinear | 0.780 (0.003) | 2.10 | 1.436 | 20.11 | 0.697 | 0.1 |
| DLinear | 0.784 (0.007) | 2.09 | 1.431 | 20.05 | 0.700 | 0.1 |
| ResidualLinearAR 2D (direct) | 0.788 (0.002) | 2.16 | 1.455 | 21.05 | 0.704 | 0.1 |
| LinearAR 2D (direct) | 0.812 (0.002) | 2.18 | 1.465 | 21.02 | 0.724 | 0.1 |
| PatchTST, no instance norm | 0.820 (0.027) | 2.19 | 1.464 | 21.85 | 0.732 | 1.0 |
| PatchTST, joint channels | 0.835 (0.010) | 2.31 | 1.507 | 21.37 | 0.744 | 0.1 |

**2400-2600 MHz**

| Model | MAE dB (sd) | MSE dB2 | RMSE dB | MAPE linear % | MAPE dBm % | median final (min) |
|---|---:|---:|---:|---:|---:|---:|
| RevIN-MLP | 0.617 (0.001) | 1.98 | 1.403 | 15.95 | 0.512 | 0.1 |
| OFA (GPT-2) | 0.622 (0.003) | 2.03 | 1.419 | 16.29 | 0.517 | 6.0 |
| OFA, random-init backbone (norms train) | 0.625 (0.008) | 2.00 | 1.409 | 16.12 | 0.519 | 6.0 |
| OFA, random-init, all weights train | 0.628 (0.010) | 2.09 | 1.440 | 17.13 | 0.521 | 9.6 |
| PatchTST | 0.629 (0.013) | 2.07 | 1.433 | 16.89 | 0.522 | 1.1 |
| PatchTST, no patching (patch 1) | 0.637 (0.005) | 2.17 | 1.467 | 16.22 | 0.530 | 3.1 |
| NLinear | 0.646 (0.003) | 2.13 | 1.458 | 17.62 | 0.535 | 0.1 |
| PatchTST, no instance norm | 0.647 (0.014) | 1.99 | 1.407 | 16.46 | 0.536 | 0.3 |
| ResidualLinearAR 2D (direct) | 0.664 (0.003) | 2.16 | 1.468 | 17.50 | 0.551 | 0.1 |
| PatchTST, joint channels | 0.670 (0.003) | 2.26 | 1.501 | 17.49 | 0.556 | 0.2 |
| DLinear | 0.671 (0.020) | 2.17 | 1.469 | 17.22 | 0.555 | 0.1 |
| ResidualLinearAR + MLP head | 0.672 (0.012) | 2.22 | 1.484 | 17.61 | 0.557 | 0.1 |
| LinearAR 2D (direct) | 0.702 (0.009) | 2.25 | 1.499 | 18.06 | 0.580 | 0.1 |

**Findings.**
1. **The simplest model that matches OFA is RevIN-MLP** (a 2-layer MLP on the per-window-normalized lookback; the final trains in about 0.1 minutes against 14 for OFA): 0.744 vs OFA 0.749 (600-800) and 0.617 vs 0.622 (2400-2600). It is better by 0.005 dB in both bands, which is about 1.7 times OFA's seed sd; call it "matches or slightly beats".
2. **Weight sharing across bins is most of the gap between the linear models; per-window normalization matters in the PatchTST ablation.** RevIN-MLP, NLinear, DLinear, the MLP-head model, PatchTST, OFA and Time-LLM are channel-independent with weights **shared by every bin**; LinearAR 2D and ResidualLinearAR 2D have separate weights per bin. NLinear (shared, last-value normalization) is 4% (600-800) and 8% (2400-2600) better than direct LinearAR 2D (0.780 vs 0.812; 0.646 vs 0.702), but that comparison mixes sharing and normalization: DLinear (shared, no normalization) is already 3.4% better than LinearAR 2D at 600-800 (0.784), and NLinear adds only 0.5% (600-800) and 3.7% (2400-2600) over DLinear. The clean normalization evidence is the PatchTST ablation (only normalization changes): removing instance normalization costs 8.6% (0.755 to 0.820) at 600-800 and 2.9% at 2400-2600. A shared-weight linear model with and without normalization on the same footing (a direct 1D LinearAR) was not run.
3. **Channel independence matters, patching does not.** Joint-channel PatchTST is 10.6% / 6.5% worse; removing patching (patch length 1) changes MAE by 0.8% / 1.3% (about the seed sd).
4. **The pretrained language model contributes at most 0.01 dB (1.3%).** OFA with a random-initialized backbone: 0.759 / 0.625; fully trained from scratch: 0.751 / 0.628; pretrained OFA: 0.749 / 0.622.
5. **Direct training of ResidualLinearAR beats the rolled-out one** (0.788 vs 0.809 at 600-800, 0.664 vs 0.688 at 2400-2600), so part of the apparent gap between the direct neural models and "ResidualLinearAR" in the earlier tables is rolled-out versus direct training, not model class. The learned nonlinear models still beat the direct ResidualLinearAR by 4-6% (OFA 0.749 vs 0.788; 0.622 vs 0.664).
6. Time-LLM is not in the table: its finals have not started (section 24.6).

### 24.2 Direct models on 1-minute POWDER (thinned test windows, see above)

**600-800 MHz**

| Model | MAE dB (sd) | MSE dB2 | RMSE dB | MAPE linear % | MAPE dBm % | median final (min) |
|---|---:|---:|---:|---:|---:|---:|
| PatchTST | 0.917 (0.001) | 3.31 | 1.817 | 23.36 | 0.826 | 2.8 |
| OFA (GPT-2) | 0.918 (0.003) | 3.32 | 1.820 | 23.43 | 0.826 | 13.6 |
| ResidualLinearAR 2D (direct) | 0.929 (0.000) | 3.32 | 1.821 | 24.25 | 0.835 | 0.2 |
| ResidualLinearAR + MLP head | 0.931 (0.002) | 3.31 | 1.817 | 24.06 | 0.837 | 0.2 |
| DLinear | 0.941 (0.004) | 3.34 | 1.824 | 24.49 | 0.845 | 0.5 |

**2400-2600 MHz**

| Model | MAE dB (sd) | MSE dB2 | RMSE dB | MAPE linear % | MAPE dBm % | median final (min) |
|---|---:|---:|---:|---:|---:|---:|
| OFA (GPT-2) | 0.749 (0.002) | 3.26 | 1.797 | 18.30 | 0.630 | 19.1 |
| PatchTST | 0.755 (0.001) | 3.27 | 1.801 | 18.86 | 0.634 | 1.9 |
| ResidualLinearAR 2D (direct) | 0.760 (0.000) | 3.26 | 1.798 | 19.50 | 0.637 | 0.2 |
| ResidualLinearAR + MLP head | 0.761 (0.004) | 3.24 | 1.794 | 19.17 | 0.638 | 0.2 |
| DLinear | 0.775 (0.008) | 3.30 | 1.809 | 18.88 | 0.650 | 0.4 |

PatchTST and OFA are 1.3% and 1.2% better than the direct ResidualLinearAR at 600-800 and 0.7% and 1.4% at 2400-2600, with seed sds of 0.000-0.004; RMSE is within 0.6% across all five models (MSE within about 1%). For comparison the parameter-free validation-tuned median (section 24.3) scores 0.919 and 0.734: at stride 1 (section 24.12) PatchTST scores 0.919 and 0.743, so the median ties it at 600-800 and is 1.2% better at 2400-2600 (the thinned numbers above overstate 2400-2600 by about 1.5%).

### 24.3 Median baseline and median-anchored ResidualLinearAR (`claude-median-full-1`, `claude-median-sweep-2`; code `training/ray/median_study.py`)

**Baseline sweep (done; selected on validation, scored on the test week; rolled-out one-step statistic through the repository pipeline, pooled over the four core receivers).** MAE in dB, mean over horizons:

| Resolution, band | lookback mean (whole window) | median, one window length K | median, per-horizon K |
|---|---:|---:|---:|
| 1 minute, 600-800 | 0.949 | **0.919** (K=30; +3.2%) | 0.917 (+3.3%) |
| 1 minute, 2400-2600 | 0.768 | **0.734** (K=20; +4.4%) | 0.734 (+4.4%) |
| 15 minute, 600-800 | 1.184 | 1.224 (K=2; worse) | **0.981** (K per horizon 3/2/96/96; +17.2%) |
| 15 minute, 2400-2600 | 0.982 | 0.923 (K=96; +6.0%) | **0.870** (K per horizon 3/6/96/96; +11.4%) |

All five metrics are stored for every window length (1-minute 600-800, K=30: MAE 0.919, MSE 3.787, RMSE 1.944, MAPE linear 23.53%, MAPE dBm 0.831%), with per-bin test tables under `median-study/sweep/<resolution>/<band>/per-bin/`. The whole-window mean here (0.949) reproduces the 0.950 of the model tables, whereas the level-diagnostics 60-minute mean (0.958) is a plain direct mean about 1% worse. At 1 minute the parameter-free tuned median is 0.5% and 1.3% better than ResidualLinearAR 2D (0.924, 0.744), matches the best direct learned models at 600-800 and is 2% better than them at 2400-2600 (section 24.2, whose windows are thinned). At 15 minutes a single window length fails and the per-horizon pick mixes persistence-like short windows with a whole-window statistic; it still trails the trained models (0.809 / 0.688 rolled out, 0.788 / 0.664 direct).

**Retrained ResidualLinearAR variants (done, 64 of 64 runs, no failures; `ref` = mean anchor + MSE loss, `mae` = MAE loss, `median` = median anchor with the validation-selected window K, `median_mae` = both; hyperparameters inherited from each winner; 2D: 3 seeds, 1D: 1 seed; 1-minute 1D test windows thinned 10x).** Test MAE in dB, mean over horizons, and the change against `ref`:

| Resolution, band, rep | ref | mae | median | median_mae |
|---|---:|---:|---:|---:|
| 1 minute, 600-800, 2D | 0.9240 | 0.9780 (-5.8%) | 0.9320 (-0.9%) | **0.9141 (+1.1%)** |
| 1 minute, 600-800, 1D | 0.9262 | 0.9338 (-0.8%) | 0.9294 (-0.4%) | **0.9072 (+2.0%)** |
| 1 minute, 2400-2600, 2D | 0.7435 | 0.7706 (-3.6%) | 0.7563 (-1.7%) | **0.7258 (+2.4%)** |
| 1 minute, 2400-2600, 1D | 0.7470 | 0.7546 (-1.0%) | 0.7583 (-1.5%) | **0.7334 (+1.8%)** |
| 15 minute, 600-800, 2D | 0.8071 | 0.8011 (+0.7%) | 0.8575 (-6.2%) | 0.8396 (-4.0%) |
| 15 minute, 600-800, 1D | 0.8080 | 0.7874 (+2.6%) | 0.9434 (-16.8%) | 0.7959 (+1.5%) |
| 15 minute, 2400-2600, 2D | 0.6890 | 0.6580 (+4.5%) | 0.6713 (+2.6%) | **0.6393 (+7.2%)** |
| 15 minute, 2400-2600, 1D | 0.6579 | 0.6319 (+4.0%) | 0.6511 (+1.0%) | 0.6414 (+2.5%) |

**Findings.** (1) At 1 minute neither change helps alone (MAE loss alone costs 0.8-5.8%, the median anchor alone 0.4-1.7%) but together they help by 1.1-2.4%; median_mae is the best simple linear model at 1 minute (2D: 0.914 and 0.726) and slightly beats the parameter-free tuned median (0.919, 0.734) and the best direct learned model at stride 1 (PatchTST 0.919 and 0.743; section 24.12), by 0.5% at 600-800 and 2.3% at 2400-2600. (2) At 15 minutes the MAE loss helps in every cell (+0.7 to +4.5%), while the median anchor with one window length hurts where the validation pick is short (K=2 at 600-800: -6.2% in 2D, -16.8% in 1D) and helps where it is the whole window (K=96 at 2400-2600). A per-horizon anchor would need direct models and was not run. (3) Gains of 1-2% at 1 minute are small; the 2D cells have seed sds of 0.000-0.004 and the 1D cells one seed. (4) The records hold all five metrics and per-bin tables (`median-study/variants/...`); whether the median-anchored variants also give up MSE, as the median baseline does (section 24.9), has not been analyzed.

### 24.4 Spatial evaluation of the October 1-minute 4D checkpoints (`claude-uc-spatial-oct-2`, **finished 2026-11-09 11:02 UTC: 260 of 260 cells, 0 errors**; 600-800 MHz; 10,021 windows per cell, permuted cells at stride 10; whole-band errors recomputed from the per-region files by `evaluation/analysis/spatial_whole_band.py`, results `evaluation/results/spatial_oct2026_whole_band*.csv`)

Whole-band MAE in dB, mean over horizons 1/15/60 minutes, receivers and the 5 seeds (the 4D lookback mean has 1 seed; permuted rows: 5 permutations per seed):

| Condition | ConvLSTM | DSwinLSTM (round 2) | ResidualConvLSTM | LinearAR 4D | ResidualLinearAR 4D | 4D lookback mean |
|---|---:|---:|---:|---:|---:|---:|
| core four in the map, core four scored | 1.249 | 1.244 | 1.006 | 0.995 | **0.986** | 1.015 |
| guest house left out of the map | 4.746 | 4.739 | 4.673 | 4.686 | 4.671 | 4.667 |
| guest house added (no retraining) | 4.495 | 4.103 | 1.551 | 1.702 | **1.532** | 1.553 |
| law 73 left out of the map | 6.086 | 6.065 | 5.948 | 5.965 | 5.945 | 5.939 |
| law 73 added (no retraining) | 5.923 | 5.858 | 5.617 | 5.651 | 5.612 | 5.608 |
| permuted coordinates | **1.339** | 2.621 | 2.855 | 2.742 | 2.850 | 2.859 |

**Findings.** (1) In distribution the linear and residual 4D models are within 3% of the parameter-free 4D mean (0.986-1.006 vs 1.015); ConvLSTM and DSwinLSTM are 23% worse. (2) A receiver left out of the map is forecast 4-5 times worse than one in it, and no model beats the baseline there: the error is a level mismatch with the interpolated map, not forecasting skill. (3) Adding guest house (inside the core-four grid) recovers most of the error for the residual and linear models (4.7 to 1.5-1.7) but **not for ConvLSTM or DSwinLSTM (4.5 and 4.1)**: the neural models cannot use a receiver added to the map without retraining. Adding law 73 (outside the grid) recovers almost nothing for any model (5.9 to 5.6). (4) The permuted-geometry test degrades the parameter-free baseline as much as the linear models (2.86 vs 2.74-2.85), so it measures reading another receiver's value, not geometry use. ConvLSTM barely changes (1.25 to 1.34) and DSwinLSTM changes by 2.1x; consistent with ConvLSTM relying on a learned per-location pattern rather than the input level (a hypothesis; a train-mean-map baseline would test it). (5) **ResidualConvLSTM beats ConvLSTM in every accuracy comparison and every seed:** 1-minute map metric 0.774 vs 0.893-0.904 (600-800; all five ResidualConvLSTM seeds are identical to three decimals), 0.647 vs 0.676 (2400-2600, seed 42), and at the physical receivers in all conditions except the permuted diagnostic (1.01 vs 1.25 in distribution; 1.55 vs 4.50 with guest house added). ConvLSTM is also worse than the parameter-free 4D mean, so it was dropped from the 15-minute 4D tuning (its 1-minute results stay).

### 24.5 15-minute 4D with inherited hyperparameters (`claude-uc-spatial15-s2s3-1`, stopped; four finished runs under `spatial15-oct2026/`, labelled inherited)

Default core-four map (IDW power 2, 10x10, grid fixed to the six-receiver area). Test MAE in dB, mean over horizons 4/12/24/48 steps; the map column is the mean over all 100 cells, the others use the evaluator's physical-receiver scoring.

| Model | full map | core four, physical | guest house out / added | law 73 out / added |
|---|---:|---:|---:|---:|
| 4D lookback mean (1 seed) | 1.170 | 1.215 | 4.70 / 2.00 | 6.26 / 3.38 |
| LinearAR 4D (3 seeds, 40-42) | 0.690-0.701 | 0.906-0.911 | 4.53-4.55 / 2.06-2.13 | 6.06 / 3.58 (seed 41) |

A trained linear 4D run takes about 54 minutes on CPU at 15 minutes. These are superseded by the tuned runs below; the full map (whose 100 cells are almost all interpolated) flatters the models relative to the receivers (0.70 vs 0.91).

### 24.6 Job state at 13:50 UTC on 2026-11-09 (superseded by 25.3) (watchdogs `claude-watchdog-{uc,tacc,h100}-1` running, no resubmissions needed so far)

| Cluster | Finished | Running |
|---|---|---|
| UC (8 RTX 6000) | 1-minute spatial evaluation (260 cells); linear 4D tuning at 15 minutes (done 12:07) | ResidualConvLSTM 4D tuning (`-neural-2`, restarted 13:42 without ConvLSTM: 6 of 32 trials done, 26 left); STS-PredNet tuning (`-sts-1`: trials 16 of 16 in both bands, finals 0 of 5); TimeRAN head-only (`claude-uc-timeran15-2`: `timeran_lp` 600-800 trials 32 of 32 and 2400-2600 6 of 32, `moment_lp` 0 of 64, finals 0 of 80; about 4 trials per hour, 20+ hours left) |
| A100 (4 A100, lease ends 11-10 18:00 UTC) | median study (64 of 64, done 07:42) | spectral ablations, 3 seeds (1 minute 236 and 15 minute 94 records, including older seeds 43-44; estimate by the lease end about 75% and 30% of the 390 cells each); 15-minute direct rescoring `claude-rescore-direct15-3` (relaunched 13:42; 325 records so far) |
| H100 (2 H100 since 13:40; the other 4 expired) | Time-LLM at 15 minutes (40 of 40 finals) | Time-LLM at 1 minute: trials 62 of 64 (31 and 31), finals 0 of 30 (about 45 GPU-hours; expected done 11-10 09:00-12:00 UTC on the 2 remaining GPUs); new H100s expected about 15:30 UTC |

### 24.7 Time-LLM at 15 minutes (finals finished; 5 seeds x 4 horizons; test MAE in dB, mean over horizons)

| Band | Time-LLM (MAE, sd) | OFA | Time-LLM MSE / RMSE / MAPE lin / MAPE dBm | final time, Time-LLM vs OFA |
|---|---:|---:|---|---|
| 600-800 | 0.752 (0.006) | 0.749 (0.003) | 2.01 / 1.405 / 19.03% / 0.674% | 49 vs 14 min |
| 2400-2600 | 0.626 (0.003) | 0.622 (0.003) | 2.04 / 1.425 / 16.74% / 0.519% | 50 vs 6 min |

Time-LLM matches OFA (0.4-0.5% worse, within about one seed sd) at 3.5-8x the training time. Winners: 600-800 learning rate 6.8e-5, 6 LLM layers, patch 16/16, d_model 64, d_ff 128 (validation objective 0.644 vs 0.654 for the default); 2400-2600 learning rate 1.3e-4, 6 layers, patch 16/8, d_model 32, d_ff 32 (0.610 vs 0.615). OFA's winners: 600-800 learning rate 4.5e-4, 6 layers, patch 32/16; 2400-2600 learning rate 7.5e-4, 3 layers, patch 32/32. Both language models are at the level of the RevIN-MLP (0.744, 0.617) and PatchTST (0.755, 0.629) of section 24.1.

### 24.8 15-minute 4D, tuned (`tuning/powder15_4d/<band>/4d/<model>/`; 16 candidates per band selected on the map-level validation MAE, then 5 seeds; default core-four map, IDW power 2, 10x10, grid fixed to the six-receiver area; the metrics are over all 100 map cells, which are almost all interpolated, so they are **not comparable with the 2D/1D tables**)

| Band | Model | MAE dB (sd) | MSE | RMSE | MAPE lin % | MAPE dBm % | winner (ridge alpha, learning rate, batch) | final time |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 600-800 | LinearAR 4D | 0.547 (0.002) | 0.94 | 0.965 | 13.26 | 0.483 | 1e-3, 5.9e-4, 8 | 64 min |
| 600-800 | ResidualLinearAR 4D | 0.547 (0.004) | 0.95 | 0.973 | 13.72 | 0.483 | 1e-2, 1e-3, 16 | 95 min |
| 2400-2600 | LinearAR 4D | **0.484** (0.008) | 0.82 | 0.906 | 12.08 | 0.391 | 1e-2, 2.4e-3, 8 | 63 min |
| 2400-2600 | ResidualLinearAR 4D | 0.498 (0.001) | 0.87 | 0.931 | 12.70 | 0.402 | 1e-4, 1e-3, 16 | 85 min |

**Tuning matters for 4D:** the inherited-hyperparameter runs scored 0.69-0.70 on the same map and range (section 24.5), 21-22% worse than the tuned 0.547; the 4D lookback mean scores 1.17. The two linear models are equal at 600-800 and LinearAR is better at 2400-2600. ResidualConvLSTM: 6 trials in 600-800 finished, median 3.3 hours per trial (no cheaper than at 1 minute), best validation objective 1.26 against 0.75 and 0.73 for the linear models (same units, mean over horizons). STS-PredNet (recent-history branch only, `lc` equal to the 96-step lookback; flat config, 30-32 minutes per trial): 16 of 16 trials done in both bands; its validation objective (0.171 and 0.252) is in normalized loss units and is **not comparable** with the MAE objectives above; the finals will give comparable test numbers.

### 24.9 Mean versus median per metric (from the baseline sweep; window selected on validation for each metric, test values)

| 1 minute | median, MAE | mean, MAE | median, MSE | mean, MSE |
|---|---:|---:|---:|---:|
| 600-800 | **0.919** | 0.936 | 3.76 | **3.39** |
| 2400-2600 | **0.734** | 0.751 | 3.31 | **3.14** |

Neither statistic wins both: the median wins MAE by 1.8-2.2%, the mean wins MSE by 10% and 5% (RMSE 5.1% and 2.7%), because spikes revert and the median ignores them while MSE punishes them. "The tuned median beats the lookback mean" is therefore a statement about MAE, our headline metric, and must be reported with the other metrics next to it. At 15 minutes the choices differ by metric and the gaps are mostly under 4% (MAE at 2400-2600: median K=96 beats the mean by 17%).

### 24.10 Decisions and infrastructure changes, 2026-11-09

- **Decisions:** ablations cut to 3 seeds (missing seeds 43-44 can be added later, finished cells are skipped); TimeRAN restricted to the head-only variants (`timeran_lp`, `moment_lp`; full fine-tuning and random-init dropped as costing at least 3x and redundant with the OFA controls), with a pre-existing paper sentence that must be changed to "head-only"; the 15-minute spatial runs with inherited hyperparameters were stopped (4 finished runs kept, labelled inherited in `spatial15-oct2026/README.txt`) because tuning matters (section 24.8); ConvLSTM dropped from the 15-minute 4D tuning (section 24.4).
- **Still to write:** runner support for the spatial experiments (map sensitivity, leave-one-receiver-out, receiver count) with the tuned 4D winners, under `spatial15-tuned-oct2026/`; kriging, the transfer adaptation ladder, week-2 drift and the daily-history STS-PredNet experiment are planned (`training/SPATIAL_PLAN.md`) and not queued; the 1-minute STS-PredNet search was not started.
- **Infrastructure:** job watchdogs (`training/ray/job_watchdog.py`, lists in `training/ray/watchdog/`) resubmit FAILED drivers every 10 minutes; `rescore.py` now puts the isolated library path first (the A100 rescoring had failed on every OFA checkpoint with "No module named transformers"), and the A100 workers got the language-model setup (`llm_powder --setup`); `tune_fixed` gained the 15-minute 4D source, bounded submission and reserved memory; `median_study.py`, `spatial_oct.py`, `spatial15.py` and the whole-band script are new. All code is pushed (`origin/integrate` at d31f0f8); this document and other untracked files are not in git.
- **Open for the user:** what the new H100s (about 15:30 UTC) should run (Time-LLM 1-minute finals, the second inference timing run, or the 1-minute 2400-2600 re-tuning of the 2D family and linear models, up to about 80 GPU-hours); whether to check MSE/RMSE of the median-anchored variants.

### 24.11 ElectroSense cheap tests (`claude-simple-es-1`; datasets spec4 and spec25; 16 candidates each, validation-selected at horizons 12 and 192, finals 5 horizons x 5 seeds; the paper's metric is raw MSE, mean over horizons 12/24/48/96/192; standardized MSE and MAE are in the stored records)

| Model | spec4 raw MSE (sd) | spec4 std MSE / std MAE | spec25 raw MSE (sd) | spec25 std MSE / std MAE | final time (min) |
|---|---:|---|---:|---|---:|
| **OFA, tuned (section 22.3)** | **0.279** (0.007) | - | **0.817** (0.009) | - | - |
| **Time-LLM, tuned (22.3)** | **0.275** (0.005) | - | 0.838 (0.009) | - | - |
| RevIN-MLP | 0.309 (0.003) | 0.408 / 0.471 | 1.085 (0.003) | 0.669 / 0.592 | 0.06-0.11 |
| ResidualLinearAR + MLP head | 0.309 (0.001) | 0.445 / 0.506 | 1.187 (0.006) | 0.719 / 0.619 | 0.05 |
| ResidualLinearAR 2D (direct) | 0.324 (0.001) | 0.466 / 0.523 | 1.197 (0.004) | 0.731 / 0.626 | 0.06-0.07 |
| PatchTST (own implementation) | 0.333 (0.007) | 0.429 / 0.489 | 1.083 (0.010) | 0.677 / 0.598 | 0.10-0.34 |
| NLinear | 0.344 (0.016) | 0.432 / 0.493 | 1.082 (0.006) | 0.681 / 0.603 | 0.05-0.11 |
| PatchTST, no patching | 0.347 (0.022) | 0.435 / 0.497 | 1.107 (0.020) | 0.691 / 0.607 | 0.47-4.35 |
| DLinear | 0.356 (0.007) | 0.492 / 0.505 | 1.302 (0.019) | 0.918 / 0.700 | 0.04-0.09 |
| LinearAR 2D (direct) | 0.384 (0.005) | 0.565 / 0.552 | 1.434 (0.004) | 0.936 / 0.721 | 0.04 |
| PatchTST, no instance norm | 0.414 (0.025) | 0.635 / 0.569 | 1.411 (0.026) | 1.006 / 0.725 | 0.25-0.29 |
| PatchTST, joint channels | 0.568 (0.050) | 1.011 / 0.724 | 1.477 (0.028) | 0.912 / 0.701 | 0.07-0.27 |

References from section 22.3: rolled-out ResidualLinearAR 2D 0.275-0.277 (spec4) and 1.013-1.022 (spec25); tuned TSL PatchTST 0.311 and 0.849; lookback mean 0.804 and 1.493 (spec25); paper OFA 0.299 and 0.846.

**Findings.** (1) **[RETRACTED, cause found 2026-11-10: the gap is a metric-convention artifact, see 17.1. The OFA and Time-LLM numbers it was measured against are window-averaged over steps 1..H, ours are endpoint (step H) values; in a common convention OFA equals the cheap models on spec25 (0.78 / 1.43 against RevIN-MLP 0.787 / 1.480).]** Original reading: on ElectroSense the cheap models do not match OFA, unlike on POWDER: the best cheap model (RevIN-MLP, NLinear) is 11% (spec4) and 32% (spec25) behind OFA in raw MSE (0.309 vs 0.279; 1.082 vs 0.817). With 25 channels the gap between the language models and every cheap model is large. The language-model controls (random-init backbone, trained from scratch) have **not been run on ElectroSense**, so whether the pretrained backbone matters here is open; on POWDER it contributed at most 1.3% (section 24.1). (2) The ablations repeat the POWDER pattern: removing instance normalization costs 24% and 30% in raw MSE, joint channels 71% and 36%, while removing patching costs 4% and 2% (about the seed sd). (3) **The direct ResidualLinearAR is 18% (spec4) and 17% (spec25) worse than the rolled-out one** (0.324 vs 0.275; 1.197 vs 1.022), the opposite of POWDER at 15 minutes, where direct training helps by 2-3%; the datasets are small (2,872 rows) and a direct model trains on one horizon's targets only. (4) **Our PatchTST implementation is far behind the TSL PatchTST at 25 channels** (1.083 vs 0.849, 28%) and 7% behind at 4 channels (0.333 vs 0.311); this implementation gap is not explained, so the PatchTST ablations above characterize our implementation, and the POWDER conclusion that nonlinear patch models beat linear ones should not be extended to ElectroSense without checking it.

### 24.12 Stride-1 rescoring of the 1-minute direct finals (`rescored/direct/1min/`; every test window, 40,084 per run, instead of every 15th)

| Band | Model | MAE dB (sd), stride 1 | MSE | RMSE | MAPE lin % | MAPE dBm % | thinned MAE | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 600-800 | PatchTST | **0.919** (0.001) | 3.33 | 1.822 | 23.42 | 0.827 | 0.917 | +0.002 |
| 600-800 | ResidualLinearAR 2D (direct) | 0.932 (0.000) | 3.35 | 1.828 | 24.36 | 0.838 | 0.929 | +0.003 |
| 600-800 | ResidualLinearAR + MLP head | 0.934 (0.002) | 3.33 | 1.823 | 24.15 | 0.839 | 0.931 | +0.003 |
| 600-800 | DLinear | 0.944 (0.004) | 3.36 | 1.830 | 24.57 | 0.847 | 0.941 | +0.003 |
| 2400-2600 | PatchTST | **0.743** (0.001) | 3.10 | 1.754 | 18.83 | 0.622 | 0.755 | -0.012 |
| 2400-2600 | ResidualLinearAR 2D (direct) | 0.750 (0.000) | 3.11 | 1.757 | 19.59 | 0.627 | 0.760 | -0.010 |
| 2400-2600 | ResidualLinearAR + MLP head | 0.750 (0.004) | 3.09 | 1.751 | 19.22 | 0.627 | 0.761 | -0.011 |
| 2400-2600 | DLinear | 0.763 (0.007) | 3.13 | 1.763 | 18.87 | 0.638 | 0.775 | -0.012 |

Thinning changes the MAE by at most 0.012 dB (1.6%), with the same sign for every model within a band (+0.003 at 600-800, -0.011 at 2400-2600: a different sample of the test week, not noise), and **the ranking does not change**: PatchTST is best, 0.9-1.4% ahead of the direct ResidualLinearAR; use the stride-1 numbers in the paper. **OFA is missing:** all 30 OFA records (2 bands x 3 horizons x 5 seeds) failed with `No module named transformers` (the A100 workers lacked the library); after the fix they were resubmitted as `claude-rescore-direct1min-3` (OFA only, stride 1) and Time-LLM's 1-minute finals have not finished. At stride 1 the validation-tuned parameter-free median (0.919 and 0.734, section 24.3) ties PatchTST at 600-800 and is 1.2% better at 2400-2600, and the median-anchored MAE-loss ResidualLinearAR (0.914 and 0.726) is 0.5% and 2.3% better than PatchTST.

### 24.13 Baselines on every horizon, on the model windows (`claude-baselines-all-1`, A100 CPU; `evaluation/results/baselines_all_horizons.csv`, built by `evaluation/analysis/baseline_table.py`)

Each baseline is the repository pipeline's parameter-free model, scored on the same test windows, normalization and pooling as the learned models (4 receivers pooled), with the parameter chosen **on validation separately for each horizon** (window K for the mean and median, gain for the EWMA grid 0.5-0.01; the Kalman filter has one steady-state gain per bin fitted on the training run, averaged over the four receivers). The CSV has all five metrics (MAE, MSE, RMSE, MAPE linear, MAPE dBm) for every row, plus the one-parameter selection (one value for all horizons) and the per-bin test tables are under `baselines-all-horizons/sweep/`. The 15-minute rows now include the 1-step horizon (t+15 minutes); the best row per horizon is in bold. **Selection rule:** every horizon, the 1-step horizon included, gets its own parameter chosen on its own validation MAE; the window selection of the earlier sweep (24.3) used the original horizons 4/12/24/48 and is unchanged, so the earlier choices are not affected by adding t+15 minutes. Selected values at 15 minutes (horizons 1 / 4 / 12 / 24 / 48 steps): window median K = 3 / 3 / 2 / 96 / 96 at 600-800 and 3 / 3 / 6 / 96 / 96 at 2400-2600; window mean K = 3 / 4 / 3 / 96 / 96 and 3 / 6 / 8 / 96 / 96; EWMA gain = 0.5 / 0.5 / 0.5 / 0.01 / 0.01 at 600-800 and 0.3 / 0.3 / 0.3 / 0.3 / 0.01 at 2400-2600. So the one-step horizon uses the same short window (3 steps = 45 minutes) as the 1-hour horizon, not the whole-window choice of the 6-12 hour horizons.

**1 minute, 600-800 MHz** (horizons in minutes; test MAE in dB)

| Baseline | 1 | 15 | 60 |
|---|---:|---:|---:|
| training-set mean (pooled over the 4 receivers) | 2.719 | 2.720 | 2.720 |
| persistence (last value) | 1.064 | 1.165 | 1.249 |
| window mean | 0.869 | 0.917 | 1.015 |
| window median | **0.851** | **0.900** | **1.002** |
| EWMA over the window | 0.857 | 0.914 | 1.014 |
| per-bin Kalman filter (steady-state gain) | 0.854 | 0.914 | 1.015 |

**1 minute, 2400-2600 MHz** (horizons in minutes; test MAE in dB)

| Baseline | 1 | 15 | 60 |
|---|---:|---:|---:|
| training-set mean (pooled over the 4 receivers) | 2.593 | 2.594 | 2.595 |
| persistence (last value) | 0.844 | 0.929 | 1.022 |
| window mean | 0.681 | 0.730 | 0.835 |
| window median | **0.663** | **0.716** | **0.824** |
| EWMA over the window | 0.672 | 0.728 | 0.835 |
| per-bin Kalman filter (steady-state gain) | 0.671 | 0.728 | 0.835 |

**15 minute, 600-800 MHz** (horizons in steps of 15 minutes; test MAE in dB)

| Baseline | 1 | 4 | 12 | 24 | 48 |
|---|---:|---:|---:|---:|---:|
| training-set mean (pooled over the 4 receivers) | 2.644 | 2.644 | 2.642 | 2.644 | 2.655 |
| persistence (last value) | 0.505 | 0.659 | 0.999 | 1.456 | 1.864 |
| window mean | 0.482 | 0.624 | 0.983 | 1.258 | 1.197 |
| window median | 0.493 | 0.641 | **0.974** | **1.152** | **1.155** |
| EWMA over the window | 0.466 | 0.625 | 0.987 | 1.305 | 1.237 |
| per-bin Kalman filter (steady-state gain) | **0.462** | **0.621** | 0.983 | 1.446 | 1.832 |

**15 minute, 2400-2600 MHz** (horizons in steps of 15 minutes; test MAE in dB)

| Baseline | 1 | 4 | 12 | 24 | 48 |
|---|---:|---:|---:|---:|---:|
| training-set mean (pooled over the 4 receivers) | 2.861 | 2.863 | 2.863 | 2.850 | 2.847 |
| persistence (last value) | 0.460 | 0.618 | 0.946 | 1.338 | 1.563 |
| window mean | 0.440 | 0.603 | 0.963 | 1.049 | 0.968 |
| window median | 0.451 | 0.604 | 0.976 | **0.945** | **0.956** |
| EWMA over the window | 0.457 | 0.616 | 0.957 | 1.327 | 1.005 |
| per-bin Kalman filter (steady-state gain) | **0.424** | **0.586** | **0.926** | 1.318 | 1.530 |

**Notes.** (1) Persistence at 1 minute (1.064 / 1.165 / 1.249 at 600-800) reproduces the level-diagnostics numbers computed separately, which checks the two code paths against each other (persistence is the window mean with K = 1). The per-bin Kalman filter is close to, but not the same as, the diagnostics' filter (0.854 vs 0.855 at horizon 1): here one gain per bin is averaged over the four receivers because the learned models are pooled, there each cell was fitted separately on prediction error, so small differences from the earlier numbers are expected, not a bug. (2) The training-set mean here is pooled over the four receivers because the learned models are pooled (2.72 and 2.59 dB at 1 minute); the level diagnostics' per-receiver training mean is much stronger (1.53 dB at 600-800) because the receivers differ in level, so a per-receiver level is a different, stronger reference. (3) The EWMA is truncated to the model's window (60 or 96 steps), so for small gains it approaches the window mean. (4) At 1 minute the window median is best at every horizon in both bands; at 15 minutes the best baseline changes with the horizon (the Kalman filter at the short horizons, the window median at the long ones: from 12 steps at 600-800 and from 24 steps at 2400-2600), the pattern of section 24.3; persistence is never the best.


## 25. State at 2026-11-10 02:10 UTC: what finished since section 24 and everything still running

### 25.1 Finished since section 24 (about 2026-11-09 20:00 to 2026-11-10 02:00 UTC)

| What | Where / when | Result |
|---|---|---|
| **One-step (t+15 minute) 15-minute finals for every direct model** (14 models x 2 bands x 5 seeds, existing winners, no search) | final H100 cluster; OFA family and the 10 cheap models done 16:06 UTC, Time-LLM 18:30 | table 25.2 |
| **Rolled-out 15-minute models rescored at t+15** (108 checkpoints, horizon 1 added to the windowing, no retraining) | A100, `rescored-h1/`, done 20:42 | table 25.2 |
| **Baselines on every horizon** | A100, done 15:26 | section 24.13 |
| **Second inference-timing run** | final H100 cluster, 20:58-21:08, `inference-benchmark/final-h100/inference_benchmark_20261109T2108Z.csv` | 323 rows, all on idle H100s, but **47 failed: every OFA, OFA-variant and Time-LLM row** (`No module named transformers`: the benchmark did not use the isolated library path). Fixed in code (`c54ab1f`); the rerun needs an idle cluster. Cheap models at batch 32 (ms per window, horizon 12, 2400-2600): NLinear 0.006, LinearAR 2D 0.006, ResidualLinearAR 2D 0.008, DLinear 0.010, RevIN-MLP 0.011, MLP head 0.015, PatchTST 0.085 |
| **1-minute 1D LSTMs on every test window** (stride 1) | final H100 cluster, done 23:11, `rescored-stride1/` (31 records, no failures) | 600-800: VanillaLSTM 1D 0.932, ResidualVanillaLSTM 1D 0.920 (5 seeds each); 2400-2600: ResidualVanillaLSTM 1D 0.745 (5 seeds), VanillaLSTM 1D 0.760 (**1 seed**: only seed 42 of that model exists for 2400-2600, not a failure) |
| **STS-PredNet 15-minute 4D tuning** | UC, done 19:17 | 5 of 5 finals per band: map-level MAE **0.952 (600-800) and 1.028 (2400-2600)** against 0.547 / 0.484 for LinearAR 4D (section 24.8); its records have no MAPE keys (its evaluator does not write them) |
| **ElectroSense: cause of the OFA/cheap-model "gap" found** | A100, `es-bisect*`, `direct-tuning-es-check*`, final cluster `es-window-rescore/` | a metric-convention difference (window-averaged over steps 1..H in the article and TSL against the endpoint step t+H in our evaluation); in one convention RevIN-MLP matches OFA and Time-LLM (0.284 / 0.826 against 0.279 / 0.817 and 0.275 / 0.838); details in 17.1 |
| **Tuned TSLib ElectroSense baselines are single runs** | analysis, 02:10 | the five "seeds" 40-44 are bit-identical (the library seeds itself with a fixed value); note in 13.1; fix in progress (25.3) |
| **LaTeX appendix (SpectrumLLM reproduction table)** | `latex/sigconf.tex` | caption now says the MSE is averaged over the forecast window; commented-out ResidualLinearAR2D row and sentence removed; seed sentence now a comment (the "five seeds" claim for the TSLib rows is wrong, see above); PDF not recompiled |
| **Disk on the two surviving old H100 nodes** | 14:50-15:00 | 42 GB and 44.6 GB of local checkpoint copies verified identical in S3 and deleted; the 21 GB of SpectrumLLM reproduction checkpoints (`repro/spectrumllm`, 370 files) were not in S3 and were uploaded first (all matched by path and size) |
| **Watchdogs on all four clusters** | `claude-watchdog-{uc,tacc,h100,final}-1`, lists in `training/ray/watchdog/` | resubmit FAILED drivers every 10 minutes; never triggered so far |

### 25.2 The t+15 minute cells (one-step horizon at 15-minute resolution; test MAE in dB on the common range; direct models trained for one step with the existing winners, rolled-out models are the existing tuned checkpoints rescored at horizon 1; all five metrics in `evaluation/results/t15_cells.csv`, built by `evaluation/analysis/t15_table.py`)

| Kind | Model | 600-800 MAE (sd) | 2400-2600 MAE (sd) | seeds |
|---|---|---:|---:|---:|
| direct (trained for 1 step) | RevIN-MLP | 0.467 (0.002) | 0.448 (0.005) | 5 |
| direct (trained for 1 step) | Time-LLM (GPT-2) | 0.471 (0.006) | 0.450 (0.005) | 5 |
| direct (trained for 1 step) | OFA, trained from scratch | 0.473 (0.004) | 0.434 (0.007) | 5 |
| direct (trained for 1 step) | OFA (GPT-2) | 0.473 (0.010) | 0.436 (0.006) | 5 |
| direct (trained for 1 step) | PatchTST, no patching | 0.474 (0.002) | 0.438 (0.013) | 5 |
| direct (trained for 1 step) | PatchTST | 0.476 (0.004) | 0.437 (0.007) | 5 |
| direct (trained for 1 step) | NLinear | 0.479 (0.006) | 0.468 (0.011) | 5 |
| direct (trained for 1 step) | OFA, random-init backbone | 0.479 (0.009) | 0.482 (0.017) | 5 |
| direct (trained for 1 step) | ResidualLinearAR + MLP head | 0.480 (0.004) | 0.492 (0.017) | 5 |
| direct (trained for 1 step) | DLinear | 0.480 (0.003) | 0.469 (0.022) | 5 |
| direct (trained for 1 step) | ResidualLinearAR 2D | 0.490 (0.001) | 0.483 (0.015) | 5 |
| direct (trained for 1 step) | PatchTST, no instance norm | 0.553 (0.028) | 0.486 (0.040) | 5 |
| direct (trained for 1 step) | LinearAR 2D | 0.558 (0.005) | 0.544 (0.010) | 5 |
| direct (trained for 1 step) | PatchTST, joint channels | 0.629 (0.007) | 0.545 (0.008) | 5 |
| rolled-out 1D | VanillaLSTM 1D | 0.464 (0.003) | 0.448 (0.013) | 5 |
| rolled-out 1D | ResidualVanillaLSTM | 0.467 (0.004) | 0.429 (0.006) | 5 |
| rolled-out 1D | LinearAR 1D | 0.479 (0.002) | 0.439 (0.002) | 5 |
| rolled-out 1D | ResidualLinearAR 1D | 0.481 (0.002) | 0.456 (0.003) | 5 |
| rolled-out 2D | ResidualLinearAR 2D | 0.498 (0.003) | 0.541 (0.009) | 5 |
| rolled-out 2D | LinearAR 2D | 0.553 (0.007) | 0.550 (0.005) | 5 |
| rolled-out 2D | TCN | 0.573 (0.021) | 0.466 (0.016) | 4 |
| rolled-out 2D | ResidualVanillaLSTM | 0.641 (0.002) | 0.470 (0.004) | 5 |
| rolled-out 2D | VanillaLSTM | 0.706 (0.002) | 0.609 (0.016) | 5 |
| rolled-out 2D | LSTMAttn | 0.743 (0.016) | 0.541 (0.044) | 5 |
| rolled-out 2D | AutoformerCSA | 0.999 (0.021) | 0.961 (0.022) | 5 |

| Baseline at t+15 min (parameter chosen on validation) | 600-800 | 2400-2600 |
|---|---:|---:|
| per-bin Kalman filter | 0.462 | 0.424 |
| window mean | 0.482 | 0.440 |
| EWMA | 0.466 | 0.457 |
| window median | 0.493 | 0.451 |
| persistence | 0.505 | 0.460 |
| training-set mean (pooled) | 2.644 | 2.861 |

**Findings.** (1) **At t+15 minutes the parameter-free per-bin Kalman filter is the best model in both bands** (0.462 and 0.424): the best learned models are 0.464 (VanillaLSTM 1D) and 0.429 (ResidualVanillaLSTM 1D); the best direct models 0.467 (RevIN-MLP) and 0.434 (OFA trained from scratch), so the learned models match the baseline at best (differences of 0.4-2.4%). (2) Among the direct models the same ordering as at longer horizons holds: the nonlinear models with per-window normalization and shared weights (RevIN-MLP, PatchTST, OFA family, Time-LLM, NLinear, DLinear) sit within 0.45-0.48 at 600-800 and 0.43-0.47 at 2400-2600, the per-bin LinearAR 2D and the joint-channel and no-normalization PatchTST ablations are 12-35% worse than the best direct model of each band. (3) The rolled-out 2D recurrent and attention models are much worse at one step at 600-800 (VanillaLSTM 2D 0.706, LSTMAttn 0.743, AutoformerCSA 0.999), consistent with their earlier failures to beat the lookback mean.

### 25.3 Still running (2026-11-10 02:03-02:10 UTC) and open

| Cluster (lease end) | Job | State |
|---|---|---|
| **Final H100** (11-14 01:49) | `claude-final-spectral-1min-r`, `-15min-r` (reverse-order second drivers of the spectral-context ablations, 3 seeds) | 1-minute 306 and 15-minute 385 records of the 390-cell targets (the A100 drivers work from the other end), nearly done |
| | `claude-final-es-repro-1` | untuned multi-seed Time-LLM (GPT-2 and BERT) and OFA (GPT-2) on ElectroSense, article regime, 2 datasets x horizons 12/48/192 x seeds 0-4 = 90 runs, `es-repro-seeds/`; none finished yet (the 25-channel Time-LLM runs, about 47 minutes each, go first) |
| **A100** (11-10 18:00) | `claude-spectral-1min-4`, `claude-spectral-15min-3` | the other ends of the same ablations |
| | `claude-rescore-direct15-3`, `claude-rescore-direct1min-3` | 15-minute direct rescoring including OFA (relaunched after the library fix); OFA 1-minute stride-1 rescoring: 28 of 30 done |
| **UC** (unknown) | `claude-uc-timeran15-2` | TimeRAN head-only: all 128 trials done; finals `timeran_lp` 600-800 15 of 20, the other three groups 0 of 20 |
| | `claude-uc-tune4d-neural-2` | ResidualConvLSTM 15-minute 4D tuning: 12 of 16 trials in 600-800, 1 of 16 in 2400-2600 (3.3 hours per trial) |
| **Old H100 pair** (11-10 18:00) | `claude-direct1min-timellm-2` (left running at the user's decision) | Time-LLM 1-minute finals 5 + 6 of 30; about 16 more fit before the lease ends, so about 3 remain to resume on the final cluster from S3 |

**Open, not started or in progress:** (a) **TSLib seed fix** (in progress, no code yet): patch the library's `run.py` (`fix_seed = 2021`, lines 10-13) to read the seed from an environment variable, set `TSL_SEED` in `tsl_runs.tsl_job`, set TSL up on a cluster (`tsl_runs setup`), reuse the existing winners (a `--winners-from` option in `tune_tsl`), then rerun the finals with real seeds: tuned baselines 5 models x spec4/spec25 x horizons 12/48/192 x seeds 40-44 = 150 runs, and the library-default baselines (6 models) x the same grid = 180 runs; (b) timing rerun for OFA, its variants and Time-LLM (needs idle GPUs); (c) window-averaged metric in our evaluation and endpoint reruns of the tuned OFA/Time-LLM ElectroSense finals (17.1 item d); (d) tuned TSS-LCD (about 95 GPU-hours, waiting for the user's go); (e) 4D at the physical receivers for 2400-2600 (runner wiring) and for the 15-minute tuned winners (runner support), and the spatial experiments S2/S3/S5 with the tuned winners; (f) audit of stale numbers in the paper (REVISION_NOTES section 4: the main-text evaluation still describes the July data); (g) **17 local commits are not pushed** (last push `d31f0f8`); the doc and other untracked files are not in git.

## 26. Update 2026-11-10 13:20 UTC: what ran since section 25, corrections, and the current state

### 26.1 Corrections to things I reported earlier (all mine)

| Earlier statement | What is true |
|---|---|
| ES multi-seed reproduction "will take about 7-8 hours" (02:50) | **7 of 90 runs finished in about 10.5 hours** (median 2,573 s per run; the last ones took 4,396 s). The final cluster's GPUs were oversubscribed: 6 + 6 ablation cells at 0.25 GPU, 4 TSS-LCD trials at 0.5 GPU and 8 ES tasks all wanted the same 4 GPUs, and the ES tasks waited. The ablation 15-minute driver is nearly done and TSS-LCD round 1 has 2 trials left, so it should speed up; nothing was changed to fix it. Cause is my reading of the reservation pattern, not confirmed task by task. |
| STS-PredNet smoke cell "is training" (03:30-03:46) | **It had failed in 0 seconds** (`KeyError: 'No tuning config for stsprednet (4d)'`, `spatial15.py` looked up a campaign config at the start of every cell) and I misread the driver job's RUNNING status. The 15 GB GPU process I saw was something else. Two bugs fixed in `spatial15.py` (config lookup; `config[model]["train"]["seed"]` on STS-PredNet's flat section); verification cells launched on UC at 13:10 (first two attempts failed in about 1 s, the third, `claude-uc-s15sts-600-2` and `-2400-2`, was past that at 13:15). |
| 15-minute ablations "387 of 390 done" (02:55) | S3 held 387 JSON files including non-cell files; the drivers' own count was **357 of 390** at 02:57. The restart of the A100 driver was still correct (its old queue would have redone finished cells). |
| "about 16 more Time-LLM finals fit before the old H100 lease ends" (11-10 early) | One final (`2400_2600 timellm c13 h15 seed40`) took **30,676 s (8.5 hours)** under contention; 14 finals are left. Most will not finish before 18:00 UTC; the rest resume from S3 on the final cluster. |
| A100 "will have enough work until the lease ends" | True for the GPUs (rescore-direct15 alone has 216 runs left), but the **tuned 15-minute 4D spatial runs will not finish** (26.5). |

### 26.2 What was built or launched since 03:15 UTC (all committed locally, none pushed)

| item | state |
|---|---|
| ResidualConvLSTM 15-minute 4D tuning | driver replaced by `claude-uc-tune4d-neural-3` at 10 parallel (was 6) at 02:19; finished trials kept, in-flight ones restarted. 13:07: 18 of 26 trials done (the last log line shows `[8 left]`, trial times 5.8 h for 2400-2600 latest). UC is at 4 of 8 GPUs reserved, so 4 GPUs are idle |
| `spatial15 --tuned` | trains each variant (S2 map settings, S3 leave-one-out, S5 receiver counts) with the **tuned 15-minute 4D winner** (best completed trial of `tuning/powder15_4d/<band>/4d/<model>`), seeds 40-42, scores at the physical receivers with `evaluate_condition`. New options `--gpu-fraction`. Smoke cell (LinearAR 4D, core four, seed 40, 600-800): map MAE 0.5476 (equals tuning's 0.547), physical receivers 0.91 dB, guest house 4.54 not in map / 2.09 in map, law 73 6.09 / 3.56; 1,924 s per cell |
| A100 runs of it | `claude-tacc-s15tuned-600-1` and `-2400-1` (LinearAR 4D, ResidualLinearAR 4D; 19 variants x 3 seeds x 2 models = 114 cells per band; CPU only, 24 GB each, 6 at a time per band). At 13:07 **60 of 114 cells per band done** (30 per model; one ResidualLinearAR 4D cell in 600-800 missing), about 6 cells per hour per band. 54 cells per band remain, about 9 hours, with about 4.9 hours of lease left: the S5 variants will not finish. They resume (prefix `spatial15-tuned-oct2026`, `--skip-done`) on UC or the final cluster |
| STS-PredNet spatial cells | the third verification attempt (`claude-uc-s15sts-600-2`, `-2400-2`) got past the config and seed bugs at 13:15 and was still running at 13:36 (no result yet). Because waiting an hour for a test is not acceptable, the real set (`claude-uc-s15sts-full-600-1`, `-2400-1`: seed 40, S2/S3/S5, 19 variants per band, 2 at a time) was launched at 13:26 on the idle UC GPUs without waiting for it, plus a 1-epoch smoke (`--smoke-epochs 1`, `claude-uc-s15sts-smoke-1`). **The smoke option did not make the test minutes-long**: at 13:36 it was still running after 10 minutes, so the five physical-receiver evaluations, not the training, dominate a spatial cell (the LinearAR cell also took 32 minutes, mostly evaluation). If the evaluation step has a bug the full jobs fail late; plan: look at the smoke result first. Planned reduction for the remaining STS variants: seed 40 only, and scoring the base variant from the saved final weights (needs an eval-only path, not written) |
| 15-minute ablations on the A100s | restarted with a fresh skip list (`claude-spectral-15min-4`, 33 cells to run) plus seeds 43 and 44 (`claude-spectral-15min-seeds4344-1`, 230 cells). The 1-minute ablation drivers (A100: 70 of 275 done; final: 170 of 218) run from opposite ends and may meet in the middle; neither re-reads the skip list while running |
| `tune_fixed --trials-only` | trials without winner or finals; a later larger `--budget` resumes (candidate lists are prefixes) |
| Tuned transfer evaluation | section 27 |

### 26.3 TSS-LCD tuning (authors' released model, not our rewrite)

| item | detail |
|---|---|
| Code | `training/ray/tune_tsslcd.py` (driver; `--budget`, `--trials-only`, `--from-index`), `training/ray/tsslcd_authors.py` (now takes `hparams`, a scoring partition and `save_weights`), `tests/test_tune_tsslcd.py` (3 tests). The authors' model classes are untouched |
| Data and protocol | **1-minute 600-800 MHz** (decided 11-10 to match the original paper; a 15-minute version is not planned), four core receivers, savgol off, search seed 42, full authors' schedule (300/300/2000 epochs) per trial, objective = mean over t+1, t+15, t+60 of MAE against the raw measurement on the validation range (2026-10-25 13:55 to 10-27 13:55), no eval-time smoothing. Finals: seeds 40-44 on the test range with the three eval-smoothing settings of the untuned runs |
| Search space | latent dim {16, 32, 48, 64}; learning rate log-uniform 3e-5 to 3e-4 (all three stages); batch size {64, 128, 256}; training-time context masking {0, 0.1, 0.2}. Candidate 0 is the authors' defaults (32, 1e-4, 256, 0.1); the 15 others are seeded draws |
| Where it ran | candidates 0-7 on the final cluster (`claude-final-tsslcd-tune-r1`, 4 at a time), candidates 8-15 on the old H100 pair (`claude-h100-tsslcd-tune-r2b`, 2 at a time, finished) |
| **Result at 13:07 (14 of 16 trials)** | best validation objective **1.1229 dB** (candidate 10: latent 64, learning rate 4.9e-5, batch 128, masking 0.1), then 1.1262 (candidate 3), 1.1282 (candidate 15), **1.1300 for the authors' defaults (candidate 0)**. Tuning gains about 0.6% on validation, which is small; the test finals will say whether it holds. Trial times 4,935 s to 9,313 s. Records: `tuning/tsslcd_authors/600_800/trial-*.json` |
| Next | when the last 2 trials finish, `python -m training.ray.tune_tsslcd --budget 16` (no `--trials-only`) picks the winner and trains the 5 finals; **not launched** because a launch now would rerun the 2 trials still in flight. VRAM per trial was never measured (two H100s of 95.8 GB each hold about 52 GB of Time-LLM, leaving 32-42 GB), so the old-pair driver was capped at 2 parallel |

### 26.4 Cluster state at 13:07 UTC (A100 and old H100 leases end 18:00 UTC; final H100 11-14 01:49 UTC; UC unknown)

| Cluster | GPUs reserved | Running |
|---|---|---|
| Final H100 | 4 of 4 | TSS-LCD round 1 (2 left), spectral ablations 15-minute (290 of 321) and 1-minute (170 of 218), ES repro (7 of 90 done, 83 left) |
| A100 | 4 of 4 | tuned 15-minute 4D linear spatial runs (60 of 114 per band), rescore-direct15 (216 left, about 1.7 per hour), 15-minute ablation seeds 43/44, 1-minute ablations (70 of 275), rescore-direct1min |
| UC | 4 of 8 | ResidualConvLSTM tuning (8 trials left), STS-PredNet verification cells; **4 GPUs idle**. The TimeRAN driver is not listed as running any more; its result has not been checked since 02:10 |
| Old H100 pair | 1.2 of 2 | Time-LLM 1-minute finals (14 left, 8.5 h each under contention); everything else finished (transfer stages A and B, TSS-LCD candidates 8-15); 0.8 GPU idle |

### 26.5 Open items after this update

(a) TSS-LCD round 2 (finals) once round 1 ends. (b) STS-PredNet spatial set on UC after verification; ResidualConvLSTM cells after its tuning. (c) Resume the unfinished tuned linear spatial cells (S5 and part of S3) after the A100 lease. (d) ES multi-seed reproduction: speed depends on the final cluster draining. (e) TSLib seed patch and reruns: not started. (f) Timing rerun for OFA / Time-LLM: needs idle GPUs. (g) Transfer work: section 27.7. (h) Audit of stale paper numbers, LaTeX recompile, unpushed commits (33 at 13:25 UTC): only on request.

## 27. Frozen zero-shot transfer of the completed 15-minute 1D/2D models (stages A and B)

Request (user, 2026-11-10): evaluate the completed October-trained 1D and 2D models, frozen, at 15-minute resolution, on (1) the held-out POWDER receivers guest house and law 73 during the same October test week, and (3) COSMOS, AERPAW and ARA using the longer traces; same-band transfer only (600-800 -> 600-800, 2400-2600 -> 2400-2600), source and target bands recorded, no combined site-and-band transfer. Stage B (per-window centering) follows A, stage C (causal normalizer refit from an earlier target calibration interval) after B. Stage A is independent of the unfinished 4D tuning.

### 27.1 Implementation (`training/ray/transfer15.py`; committed, not pushed)

| point | what the code does |
|---|---|
| Checkpoints | the 108 tuned 15-minute winning finals (`tuning/powder15/<band>/<rep>/<model>/weights/seed-N`, source `p15tuned`): 1D (LinearAR, ResidualLinearAR, VanillaLSTM, ResidualVanillaLSTM) and 2D (Autoformer-CSA, LinearAR, LSTMAttn, ResidualLinearAR, ResidualVanillaLSTM, TemporalConvNet, VanillaLSTM), seeds 40-44. Horizon-specific direct checkpoints (OFA, Time-LLM, PatchTST, DLinear, RevIN-MLP, residual heads) are **not** in this stage |
| Frozen | weights and the source normalization come from the checkpoint (per-frequency mean and standard deviation of the core receivers' training partition); nothing is estimated from target data; no post-hoc centering in stage A |
| Windows and horizons | lookback 96 x 15 minutes; endpoint error at steps 1, 4, 12, 24, 48 through the shared autoregressive `forecast` (horizon 1 is a rollout of length 1); **one set of origins per (band, receiver)**, each with the full 96-step history and 48-step future in one gap-free segment, shared by every model and baseline; 1D models flattened to one scalar series per bin with segment boundaries kept (existing `transfer_1d` machinery) |
| Target data | `data/powder15/<guesthouse|law73>-nuc1/20261025T1355Z/<band>/power_15min.csv`; target interval 2026-10-27T20:00Z to 2026-11-03T20:00Z (end exclusive; history from 24 hours earlier), 15-minute preprocessing as the core experiments (UTC-aligned blocks, mean in linear power converted back to dB, 12 of 15 valid minutes, gaps up to 16 blocks interpolated in dB); **625 origins per (band, receiver)**; missing share before interpolation recorded per record (0.0 for guest house 600-800) |
| Baselines | persistence and the 96-step lookback mean, on the same origins. **Not done:** source-validation-selected mean / median / EWMA settings (need frozen selections from the core receivers' validation) |
| Outputs (`s3://spectrum/transfer15-oct2026/<band>/<receiver>/<rep>/<model>/seed-N/`, baselines under `.../baseline/<name>/`) | `summary.json` (MAE, MSE, RMSE, MAPE linear, MAPE dBm, signed bias per horizon, plus provenance: source and target band, checkpoint key and sha256, training code revision, source config candidate and parameters, source training files, target file with sha256, raw window, target interval, preprocessing, missingness, normalization policy, evaluation code revision), `per_bin.csv` (horizon x bin: count, MAE, MSE, bias), `per_origin.csv` (horizon x origin time: MAE, MSE, bias, for paired blocked intervals and per-day skill), `forecasts/` (the exported arrays) |
| Stage B (`--stage B`, prefix `transfer15-oct2026-B`) | per input window and frequency bin the lookback mean is subtracted in the source normalization (the window is shown to the model at the source's mean level) and added back to the forecast; weights and normalization frozen; observed history only. Each B job recomputes A on the same windows (checks the origins match and stores the A metrics next to B). Models with full intrinsic window normalization (RevIN: RevIN-MLP, PatchTST, OFA, Time-LLM) are not centered again and record `intrinsic_window_normalization: full`; the residual-head models are recorded as `partial` (they scale the correction by the window's scale; level untouched) and are centered like the rest. None of the 1D/2D models in this stage has full RevIN |
| Drivers | `claude-h100-t15-stageA-1` and `-stageB-1` on the old H100 pair (`--skip-done`, watchdog entries in `training/ray/watchdog/h100.json`); old-H100 containers were updated by copying the files, because `sync-cluster-code.py` has no old-H100 option |

### 27.2 Validation

Smoke runs on guest house 600-800 (`transfer15-smoke*`, since superseded): 625 origins at every horizon, identical origin times for all models and baselines; stage A of the 1D VanillaLSTM and the 2D ResidualLinearAR **recomputed with the stage B evaluator (centering off) matches the stage A run to four decimals at every horizon**; centering leaves the 2D ResidualLinearAR unchanged to four decimals and changes the 1D VanillaLSTM strongly (t+12 MAE 1.398 -> 0.805 dB, bias -0.57 -> +0.32).

### 27.3 Results (preliminary: MAE in dB at t+12, mean over the two held-out receivers and five seeds; table regenerated by `evaluation/analysis/transfer15_table.py`, long CSV `evaluation/results/transfer15_stage_ab.csv`, 2,200 rows, all 220 stage A and 216 stage B jobs ok)

| model | 600-800: A frozen | B centered | 2400-2600: A | B |
|---|---|---|---|---|
| persistence (baseline) | 1.437 | | 1.483 | |
| lookback mean (baseline) | 1.539 | | 1.514 | |
| 1D LinearAR | 1.003 | 0.999 | 0.864 | 0.870 |
| 1D ResidualLinearAR | 1.000 | 1.000 | 0.875 | 0.875 |
| 1D VanillaLSTM | 2.004 | 1.105 | 1.414 | 1.296 |
| 1D ResidualVanillaLSTM | 1.038 | 1.038 | 0.972 | 0.972 |
| 2D LinearAR | 1.011 | 0.972 | 0.919 | 0.910 |
| 2D ResidualLinearAR | 0.960 | 0.960 | 0.883 | 0.883 |
| 2D TemporalConvNet | 1.106 | 1.021 | 0.966 | 0.883 |
| 2D ResidualVanillaLSTM | 1.339 | 1.339 | 1.489 | 1.489 |
| 2D Autoformer-CSA | 2.276 | 2.276 | 1.703 | 1.704 |
| 2D LSTMAttn | 4.566 | 2.014 | 3.035 | 2.428 |
| 2D VanillaLSTM | 4.448 | 1.760 | 3.251 | 2.115 |

What it shows, and what it does not: the linear and residual-head models transfer to the held-out receivers without adaptation and beat persistence (0.86-1.01 dB against 1.44-1.48); the plain LSTMs fail frozen (2D VanillaLSTM and LSTMAttn 4.4-4.6 dB against 1.44 persistence at 600-800) and centering recovers a large part of the failure (4.45 -> 1.76, 4.57 -> 2.01) but not all of it (still worse than persistence), while the 1D VanillaLSTM goes from worse than persistence to better (2.00 -> 1.11). Models whose B equals A to three decimals (ResidualLinearAR, ResidualVanillaLSTM in both representations, Autoformer-CSA, and the 1D ResidualLinearAR) are consistent with being level-equivariant already (an anchored residual or an internal decomposition); **I checked that the numbers are equal, not the mechanism in the model code**. Not yet done: other horizons in the write-up, per-receiver tables, paired blocked intervals, per-day skill, the difference between the two receivers, comparison with the core receivers' in-domain error.

### 27.4 COSMOS, AERPAW and ARA: frozen targets, runner and run (13:30 UTC; `training/ray/transfer_external.py`, `evaluation/analysis/transfer_external_intervals.csv`)

**Data found** (`data/{cosmos,aerpaw,ara}` on the old H100 nodes; per-minute traces): COSMOS `sdr2-md1` and `sdr2-s1-lg1`, run `20260619T181xZ`, 113.5 h each, one contiguous segment, bands 600_800, 2400_2600 and 3500_3700 (200 bins each); AERPAW CC1, CC2, LW1, run `20220208`, one wideband `87_6020` file per site (5,933 columns at x.5 MHz; longest contiguous stretches 168.0 h, 168.0 h and 114.0 h; CC1 and LW1 have 1 and 2 gaps); ARA sites ames-ue-000, curtissfarm-ue-000/-001/-002, feedmill-ue-001, horticulture-ue-004, agronomyfarm-ue-010 (many 3.4-5 h runs, a few 45-167 h runs; several files have exactly 10,000 rows, cause not checked).

**Freezing (done before any scoring, from data availability only).** For every (platform, site, run, band) with a band we have checkpoints for (600_800, 2400_2600; 3500_3700 is skipped, AERPAW wideband files count for both bands) the code takes the longest segment of raw minutes with no gap above 10 minutes, where **a row with every bin NaN counts as a missing minute**, converts it with the core 15-minute preprocessing (12 of 15 minutes valid per block, linear-power mean, interpolation over at most 16 blocks; wideband files sliced to the band's 200 bins) and keeps the target only if it has at least **96 origins** (60 hours of contiguous data: 24 h history + 12 h future + one day of origins). The scored interval starts 24 hours after the first block and ends at the last block. All 93 candidates are listed with their reason in the CSV.

| result of the freeze | |
|---|---|
| Usable targets | **19 of 93**: AERPAW CC1 (528 origins), CC2 (529), LW1 (312) in both bands; ARA ames-ue-000 `20260702T1920Z` (219; both bands), curtissfarm-ue-001 `20260702T1410Z` (523; both), curtissfarm-ue-002 `20261013T1900Z` (523; 600-800 only), horticulture-ue-004 `20260618T1255Z` (523; both; **12.5% of blocks interpolated**) and `20260702T1920Z` (412; both); COSMOS sdr2-md1 and sdr2-s1-lg1 (311 each; both bands). curtissfarm-ue-001 has 4.8% interpolated blocks; every other usable target 0% |
| Excluded for length | 58 targets: every 3.4-5 h run, the 44.7-44.9 h `20260715T2200Z` runs (36 origins) and the 20-hour feedmill run |
| Excluded for data quality (conversion fails) | **16**: all of feedmill-ue-001 (12; for the 139-hour run 42% of the rows are entirely NaN, and the 15-minute rule then leaves gaps over 16 blocks) and agronomyfarm-ue-010 (4; every value NaN) |

**Runner.** `python -m training.ray.transfer_external run` scores every 15-minute tuned checkpoint (the same 108 as section 27.1) of the matching band on every frozen target, **stage A and stage B in one job on identical windows** (RevIN-style models would not be centered twice), plus persistence and the lookback mean per target. Per-site results only; same outputs as section 27.1 under `s3://spectrum/transfer15-external-oct2026/<stage>/<platform>/<site>/<run>/<band>/<rep>/<model>/seed-N/` (baselines under `baseline/...`), with provenance (raw file, converted file sha256, frozen interval, preprocessing and imputation share, checkpoint sha256, source config, code revision). A job fails if its origin count differs from the frozen count.

**Run.** `claude-h100-ext-2` on the old H100 pair (watchdog entry added): 19 targets x 108 checkpoints, 1,045 jobs (1,022 left after the restart). `-ext-1` had a bug in my origin check for the flattened 1D models (rows are origins x 200 bins, so the check compared 105,600 with 528): all 1D jobs failed in 10-19 s and nothing was written for them; fixed and restarted at 13:33 (first 2 minutes: 18 ok, 0 failed). Expected about 1.5-2 hours at the current rate; the pair also runs Time-LLM finals.

### 27.5 Open items for section 27

(a) Check the `ext` run's results and analyze: per-site tables, comparison with persistence, the A to B change by platform, per-day skill (the traces are 114-168 hours, so per-day drift is available from `per_origin.csv`). (b) **Stage C** (causal normalizer refit from an earlier target calibration interval): not started; needs a calibration interval frozen before the scored interval for each target. (c) Horizon-specific direct checkpoints, scored from their matching checkpoints (missing ones reported as missing), RevIN models not centered twice. (d) Source-validation-selected mean / median / EWMA baselines with frozen settings. (e) Paired blocked confidence intervals from `per_origin.csv`. (f) Per-receiver and per-horizon tables for the paper. (g) The 10,000-row cap and the horticulture 12.5% interpolation should be looked at before the paper cites those targets. (h) Combined site-and-band transfer stays out of scope by instruction.


## 28. State at 2026-11-10 16:20 UTC: what ran since section 26, corrections, results, and what is still open

### 28.1 Cluster state (A100 and old H100 leases end 18:00 UTC; final H100 11-14 01:49 UTC; UC unknown)

| Cluster | GPUs reserved | Running |
|---|---|---|
| Final H100 | 4 of 4 | ES repro (81 of 90 left; **only 9 records written since it started at about 01:30 UTC**, 5,540 s per run under contention: the ablations and TSS-LCD trials share the GPUs), spectral ablations 15-minute (290 of 321) and 1-minute (180 of 218), TSS-LCD trial 7 (the last of the 16) |
| A100 | 4 of 4 | tuned 15-minute 4D linear spatial cells (now in the S5 receiver-count variants), 1-minute ablations (80 of 275), 15-minute ablations (10 of 33 plus seeds 43/44). The direct-15-minute rescoring driver was stopped at 15:57 (202 runs left, about 5 per hour because it was starved) and moved to UC |
| UC | 7 of 8 after 16:20 | ResidualConvLSTM tuning (4 trials left; one 600-800 trial took 13.4 h), neighbour ablation (22 jobs), direct-15-minute rescoring (387 of 426 left at 16:18), **TSS-LCD finals (5 seeds, one GPU each, about 6 GB VRAM each)**. Before 16:20 UC had 6 of 8 GPUs idle |
| Old H100 pair | 1.2 of 2 | Time-LLM 1-minute finals (12 left; one final took 30,676 s). The external transfer run finished at about 15:50; 0.8 GPU idle |

### 28.2 Things I said that were wrong, found since section 26

| Statement | What is true |
|---|---|
| "The STS-PredNet physical-receiver path should work after two fixes" | It did not. After the config and seed fixes the first evaluation failed with `KeyError: 'branches'`, then `Unsupported STS-PredNet output activation: 'none'`. Cause: the shared model factory builds a **different STS-PredNet class** (`models/STSPredNet.py`) from the one the integrated trainer and our tuning used (`training/STS-PredNet/stsprednet.py`: separate input interface, other activation handling, different state dict). Fixed by `load_stsprednet_for_forecast` in `training/common/spatial_checkpoint_evaluation.py` (rebuilds the training class from the checkpoint's `model_config`, closeness-only, wrapped for the shared rollout). The checkpoints do carry `frequencies` and `normalization` (the common-pipeline save path), which I first feared they lacked |
| "`--smoke-epochs` will make STS-PredNet tests take minutes" | No: with 1 epoch a cell still took 1,604 s before failing in the evaluation. Training, the map-level scoring and **five physical-receiver evaluations** each cost time. A physical-receiver condition for STS-PredNet takes about 20 minutes (a 48-step autoregressive rollout over a 96-frame history for every origin), so about **100 minutes per cell for the five conditions on top of about 80 minutes of training**. The fast test that worked: `--eval-only --eval-from-tuning` on the tuning finals (fails in seconds if the path is broken) |
| The full STS-PredNet set (19 variants x 2 bands, seed 40) is feasible | About 3 hours per cell, roughly 28 hours per band. Stopped at 14:25 (the user paused STS-PredNet to look at the other spatial tests first). The base variant `core4-p2` is scored from the tuning finals, not retrained; the in-flight `claude-uc-s15sts-eval-600-2` (3 seeds, about 100 minutes) is the only STS-PredNet job left running |
| "the A100 will keep busy and the tuned spatial runs finish" | They will not all finish by 18:00; the unfinished S5 cells resume on UC or the final cluster (`--skip-done`) |

### 28.3 Spatial results so far (15-minute, tuned winners; 142 finished cells at 14:20: LinearAR 4D and ResidualLinearAR 4D, seeds 40-42; S5 cells still arriving; rows: `/evaluation/results` has the scripts' outputs only where noted)

Physical-receiver MAE (dB, mean over horizons 4/12/24/48 steps and the two linear models) with the receivers in the input map:

| Variant | 600-800 | 2400-2600 |
|---|---:|---:|
| IDW power 1 | 2.034 | 2.407 |
| power 2 (default, `core4-p2`) | 0.905 | 0.934 |
| power 4 | 0.790 | 0.637 |
| power 16 (about nearest neighbour) | 0.790 | 0.639 |
| power 2 in the linear domain | 1.044 | 1.292 |
| grid 5x5 | 1.372 | 1.628 |
| grid 20x20 | 0.864 | 0.864 |

The map-level MAE (all 100 cells) moves the other way for smoothing: power 1 has the lowest map MAE (0.488 and 0.449) and the worst receiver error; power 16 has a higher map MAE (0.673, 0.578) and a better receiver error. The map-level score is not a faithful proxy for the receivers.

Leave-one-out (S3), receivers left out of the map vs added back without retraining (MAE dB; mean over seeds and both linear models):

| | 600-800 | 2400-2600 |
|---|---:|---:|
| scored at the receivers in the map | about 0.9 | about 0.93 |
| left out (humanities / madsen / moran / sage point / guest house / law 73) | 3.72 / 2.91 / 5.88 / 2.75 / 4.29 / (base model: 6.08) | 2.85 / 4.56 / 5.37 / 5.27 / 3.62 / 2.12 (2 cells) |
| left out then added without retraining | 1.1-1.8 | 0.7-2.0 |

Residual head vs plain: at most 0.003 dB apart on `core4-p2`.

**Level decomposition** (`evaluation/analysis/loo_level_decomposition.py`, `evaluation/results/loo_level_decomposition.csv`, committed locally): a zero-lag IDW estimate of each receiver from the others, no model. Its MAE equals the models' left-out MAE to within a few tenths (moran 600-800: 5.87 vs 5.88; humanities 3.65 vs 3.72; law 73 from the core four 6.15 vs 6.08), and **removing each frequency bin's static offset (mean over the week) removes 64-84% of it, leaving 0.7-1.6 dB**; removing a single scalar offset removes only 4-30%. So the left-out error is a stable **per-bin level difference** (spectral shape between receivers), not a gain. The finding was already visible at 1 minute (every model equals the lookback mean at a left-out receiver: 4.67 vs 4.67 dB), so it shows that the left-out test measures level differences, not spatial learning.

**Lookback-mean baseline** (partial at 14:45: 12 of 38 variants; `claude-uc-s15-lbm-{600,2400}-1`): the 4D lookback mean at the receivers in the map is 1.215 dB (600-800 and 2400-2600 `core4-p2`: 1.215 and 1.216) against 0.905 and 0.934 for the linear models, so the linear models are **23-26% better on the default map and 31-33% better on power 4 and 16 maps, but only 4-7% better on the smoothest map (power 1)**. **Permuted geometry** (S4; partial, 22 of about 110 cells): linear 4D 2.74 (600-800) and 3.21 (2400-2600) against 0.9 in distribution, and the lookback mean 2.90 and 3.33: the permutation degrades the parameter-free baseline almost as much, so it does not isolate learned geometry. Both jobs are `--tuned`/`--permuted` runs of `spatial15` under `spatial15-tuned-oct2026/` (permuted cells under `<band>/permuted/`).

### 28.4 New experiments for "do the neighbours help?" (question 4 of `OVERVIEW.md`)

| What | Where | State |
|---|---|---|
| In-distribution 1D/2D models at the four core receivers on the same origins (the 4D vs per-receiver comparison) | `training/ray/transfer15.py` now accepts the four core receivers (`RECEIVERS` has six; `HELD_OUT` is the default); `claude-h100-t15-core-1`: LinearAR and ResidualLinearAR 1D/2D, seeds 40-44, both bands, 4 receivers, plus baselines | **finished: 168 ok, 0 failed** (`transfer15-oct2026/<band>/<receiver>/...`) |
| Neighbour ablation | `training/ray/neighbor_ablation.py` (new hook `series_transform` in `evaluate_condition`): for each core receiver the other three receivers' series are replaced by a 24-hour time shift, a 52.5-hour shift, or their per-bin mean level; the receiver itself is untouched and scored; models LinearAR 4D, ResidualLinearAR 4D (tuning finals, seeds 40-44) and the 4D lookback mean as control; paired on origins valid in every mode. Output `neighbor-ablation-oct2026/<band>/<model>/seed-N/{per_origin.npz, summary.json}` | running on UC as `claude-uc-neighbor-abl-2/-3/-4` (22 jobs; each about 50-55 minutes; **15 of 22 finished by 16:25** (all ten LinearAR 4D jobs and both lookback-mean controls; 3 of the 10 ResidualLinearAR 4D jobs); an earlier driver `-1` failed on pickled timestamps and was fixed) |
| Analysis | `evaluation/analysis/neighbor_value.py`: A (4D vs per-receiver models and persistence on identical origins, per receiver and pooled, moving-block bootstrap with one-day blocks) and B (ablation deltas) | written and committed, **not yet run** (needs the ablation jobs) |

### 28.5 Transfer evaluation (section 27)

| item | state |
|---|---|
| COSMOS, AERPAW, ARA runner and run | `claude-h100-ext-2` **finished: 1,022 jobs ok, 0 failed** (19 targets x 108 checkpoints + 19 baseline jobs, stages A and B); results under `s3://spectrum/transfer15-external-oct2026/`; **not yet analyzed** (needs a per-site table like `evaluation/analysis/transfer15_table.py` produces for the POWDER receivers) |
| Frozen-target decisions | 19 of 93 candidate targets usable; the rules and exclusions are in 27.4 and in `evaluation/analysis/transfer_external_intervals.csv` |
| Stage C, direct models, source-selected baselines, confidence intervals | not built |

### 28.6 Part 1 of `OVERVIEW.md` (can models make fine-grained predictions?) and the other three open results

| item | state at 16:18 |
|---|---|
| Heuristics, 1D/2D models, PatchTST, DLinear | results exist; the direct models' common-range rescoring (426 checkpoints, `claude-uc-rescore-direct15-1`, `rescored/`) is running on UC (387 left at 16:18; OFA runs take 17-60 s each) |
| OFA | 1-minute rescoring finished (`claude-rescore-direct1min-3`, 0 left); 15-minute rescoring as above |
| Time-LLM 1-minute | **18 of 30 finals done**, and at least 3 seeds in every cell: 600-800 h1 {40,41,44}, h15 {40,41,42}, h60 {40,41,42}; 2400-2600 h1 {40,41,44}, h15 {40,41,42}, h60 {41,42,43}. 12 missing (2 per cell, about 30 GPU-hours). It needs about 52 GB of VRAM per run, so only the H100s can take them (UC's 24 GB cards cannot). **Decision pending:** report n=3 per cell (disclosed) and treat the other two seeds as optional |
| TSS-LCD tuned finals | trials **15 of 16 finished** (best validation 1.1229 dB, candidate 10: latent 64, learning rate 4.9e-5, batch 128, masking 0.1; the authors' defaults 1.130); the last trial (candidate 7, latent 64, batch 64) is still running on the final cluster. Finals for the best of the 15 were launched on the final cluster at 15:54 but waited in the queue (5 tasks pending, all four GPUs reserved by other drivers), so at 16:20 they were cancelled and **relaunched on UC** as `claude-uc-tsslcd-finals-1` (`--skip-trials`, one GPU each, about 6 GB VRAM each, running at 100% utilization). If candidate 7 beats 1.1229 the winner changes and `tune_tsslcd --budget 16` redoes the finals. Expected finish: not measured on RTX 6000s (H100 trials took 4,900-9,300 s) |

### 28.7 Durability (asked 16:00)

Results are written to S3 as each job finishes; weights are saved by the jobs that produce them. A scan of the A100 and old H100 nodes found no recent large result files that exist only locally (the large `/tmp` directories are re-downloadable pretrained weights). The driver logs, which live only in each head container, were archived at 14:35 UTC: `s3://spectrum/job-logs/{tacc,h100,final,uc}/joblogs-20261110T1435Z.tgz` (124, 39, 16 and 29 jobs). **Not in S3:** work in flight when a lease ends (a Time-LLM final, a spatial cell mid-training), and our code: the 48 local commits were **pushed to `origin/integrate` at 16:3x UTC (45dc184..1be00e2)**; this document, `OVERVIEW.md` and other untracked files are not in git and exist only on this machine.

### 28.8 Open items

(a) Finish the neighbour ablation and run `evaluation/analysis/neighbor_value.py`. (b) TSS-LCD finals (UC), then summarize tuned versus authors' defaults on the test range. (c) Time-LLM 1-minute: decision on n=3 versus n=5, and where the remaining 12 would run. (d) Analyze the external transfer results. (e) STS-PredNet: base variant from the tuning finals (running), then which spatial variants to retrain (user paused this). (f) ResidualConvLSTM tuning (4 trials left), then its spatial cells. (g) The final cluster's ES repro is starved (81 of 90 left): the ablations should drain first. (h) Lookback-mean baseline and S4 permuted runs still finishing; S5 cells resume after the A100 lease. (i) Stage C and the direct models in the transfer work. (j) Audit stale paper numbers, recompile LaTeX, commits were pushed on request (see 28.7).


## 29. State at 2026-11-10 17:30 UTC: neighbour-value results, TSS-LCD trials complete, rescoring fix, part 2 readiness

### 29.1 Cluster state (A100 and old H100 leases end 18:00 UTC; final H100 11-14 01:49 UTC)

| Cluster | GPUs reserved | Running |
|---|---|---|
| Final H100 | 4 of 4 | ES repro (80 of 90 left: one record in the last hour), spectral ablations 15-minute (300 of 321) and 1-minute (180 of 218). The TSS-LCD trial driver finished (all 16 trials) |
| A100 | 4 of 4 | tuned 15-minute 4D linear spatial cells (now `k2-*` S5 variants), 1-minute ablations (80 of 275), 15-minute ablations (10 of 33 + seeds 43/44) |
| UC | 7 of 8 | **TSS-LCD finals** (5 seeds, one GPU each; started 16:20, no finish yet, so no timing on RTX 6000s), ResidualConvLSTM tuning (4 trials left), direct-15-minute rescoring (383 of 426 left), neighbour ablation driver (finished all 22 jobs), **a `claude-uc-deepspred-acf-probe-1` job I did not launch** (`training/DeepSPred/probe/acf_probe.py`, CPU, someone else's) |
| Old H100 pair | 1.2 of 2 | Time-LLM 1-minute finals (**11 left**; the lease ends in about 30 minutes and in-flight finals will be lost; resume from S3 on the final cluster) |

### 29.2 What I found wrong since section 28

| Statement | What is true |
|---|---|
| The A100 direct-15-minute rescoring was "starved" | The cause is in `training/ray/rescore.py`: GPU tasks are **hard-pinned round-robin to specific nodes** (`NodeAffinitySchedulingStrategy(..., soft=False)`), so each pinned task waits for its own node's GPU even while other nodes have free capacity. That also stalled the UC rerun for 40 minutes after the TSS-LCD finals filled UC (log line unchanged from 16:18 to 16:57). Fixed with `soft=True` (commit "rescore: soft node affinity"), UC driver restarted as `claude-uc-rescore-direct15-2` at 16:59. It now progresses but slowly: only about 2 GPU slots (two nodes with 0.5 GPU free) and the remaining jobs are the heavy ones (MOMENT 230 s, OFA 17-60 s); 383 of 426 were left at 17:26, so the finish is probably 2-5 hours, not measured |
| TSS-LCD candidate 7 might change the winner | It finished: **16 of 16 trials ok, best still 1.1229 dB (candidate 10)**, so the finals launched on UC use the final winner and need no redo |

### 29.3 Do the neighbours help? Results (question 4 of `OVERVIEW.md`; `evaluation/analysis/neighbor_value.py`, outputs `evaluation/results/neighbor_value_compare.csv` and `neighbor_value_ablation.csv`; committed locally)

Setting: the four core receivers, 15-minute, October test week, frozen tuned base models (LinearAR 4D and ResidualLinearAR 4D, tuning finals seeds 40-44, default IDW power 2 map), MAE in dB averaged over horizons 4/12/24/48 steps, pooled over the receivers; paired on identical origins (about 2,480 per comparison); 95% intervals by a moving-block bootstrap with one-day blocks.

**A. 4D map models against per-receiver models that see only the receiver's own history** (seed-averaged; delta = 4D minus other):

| Band | 4D (LinearAR / Residual) | 1D linear | 2D linear | 2D ResidualLinearAR | 4D lookback mean | persistence |
|---|---:|---:|---:|---:|---:|---:|
| 600-800 | 0.908 / 0.904 | 0.781 | 0.804 | 0.792 | 1.215 | 1.265 |
| 2400-2600 | 0.937 / 0.939 | 0.653 | 0.683 | 0.676 | 1.216 | 1.131 |

The 4D models beat the parameter-free baselines clearly (0.31 and 0.36 dB better at 600-800; 0.28 and 0.19 at 2400-2600; all intervals exclude 0) but are **worse than the per-receiver models**: by 0.10-0.13 dB (13-16%) at 600-800 and 0.25-0.29 dB (27-44%) at 2400-2600, intervals excluding 0 in every pair (for example 600-800, LinearAR 4D vs 2D LinearAR: +0.105, [0.092, 0.117]).

**B. Neighbour ablation** (the scored receiver untouched; the other three replaced; ablated minus real MAE; all four receivers pooled):

| Neighbours | 600-800 LinearAR 4D | 600-800 Residual | 2400-2600 LinearAR 4D | 2400-2600 Residual | lookback-mean control |
|---|---:|---:|---:|---:|---:|
| shifted 24 h (short-term co-movement broken) | +0.006 [0.003, 0.011] | +0.006 | +0.006 [-0.002, 0.012] | +0.006 | 0.001 / 0.004 |
| shifted 52.5 h (daily phase broken too) | +0.024 [0.018, 0.030] | +0.023 | +0.021 [0.006, 0.034] | +0.021 | 0.001 / 0.005 |
| per-bin mean level only (no dynamics) | +0.012 [0.005, 0.014] | +0.009 | +0.012 [-0.002, 0.021] | +0.011 | -0.003 / 0.000 |

**Reading.** Breaking the neighbours' dynamics costs the 4D models only **0.6-2.6%**; the parameter-free control does not change. So the models take almost all of their skill from the receiver's own series (the map cell around a receiver is dominated by that receiver's data). Two cautions: (1) the ablation measures sensitivity, not benefit (a time-shifted neighbour also injects inconsistent signal into the blended map cell, so some of the small increase may be noise injected into the blend, not lost information); (2) the default map is power 2 on a 10x10 grid: the power-4 variant reaches 0.790 (600-800) and 0.637 (2400-2600), level with the per-receiver models (0.78-0.80; 0.65-0.68) in a roughly comparable though not paired comparison, which suggests most of the 4D deficit in table A is map fidelity (a receiver's value read from an interpolated grid cell), not missing neighbour information. **At 15 minutes the linear 4D models show no measurable benefit from spatial context**; the neural 4D models (ResidualConvLSTM, STS-PredNet) are still untested for this. Not done: the same ablation for sharper maps (power 4) and for the neural models; the time-shifted-neighbour effect at 1 minute.

### 29.4 STS-PredNet base variant (600-800, tuning finals seeds 40-42, scored at the physical receivers by `--eval-only --eval-from-tuning`)

In the map 1.256 / 1.191 / 1.162 dB (linear 4D: 0.906; 4D lookback mean 1.215); guest house not in the map 4.65 / 4.46 / 4.97, added to the map 4.13 / 4.12 / 4.64 (linear 4D: 4.54 / 1.84); law 73 not in the map 5.87-6.64, added 6.03-6.66 (linear: 6.08 / 3.26). So STS-PredNet is about 30% worse than the linear 4D in distribution and cannot use a receiver added to the map without retraining (the same as ConvLSTM and DSwinLSTM at 1 minute). 2400-2600 has **no valid result**: those cells hold the stale errors from before the fixes and were never rerun (about 100 minutes).

### 29.5 Part 2 of `OVERVIEW.md` (why do some models do worse than heuristics; level pathways): readiness for writing

| item | state | evidence |
|---|---|---|
| Plain vs residual LSTM, LinearAR vs ResidualLinearAR, LSTMAttn, TCN, Autoformer-CSA | ✅ | 15-minute tuned finals and the transfer stages A and B (sections 27.3, 28.5) |
| Frozen vs window-centered on guest house and law 73 | ✅ | MAE at t+12, A to B: 1D VanillaLSTM 2.00 to 1.11, 2D VanillaLSTM 4.45 to 1.76, LSTMAttn 4.57 to 2.01 (600-800; persistence 1.44); the residual-head models, Autoformer-CSA and 1D linear change by 0.00-0.04 (Autoformer-CSA stays at 2.28) |
| RevIN-MLP accuracy and design | ✅ | 0.744 vs OFA 0.749 (600-800) and 0.617 vs 0.622 (2400-2600), trains in about 0.1 minutes against 14 |
| PatchTST with and without instance normalization | ✅ | removing it costs 8.6% (600-800) and 2.9% (2400-2600) |
| **Constant-offset response tests** | ○ not started; design in the TCN TODO (add c dB to a window, record the forecast shift, expect slope 1 for level-equivariant models) | needed to turn "B equals A" (verified only as equal numbers) into a mechanism claim |
| STS-PredNet level diagnostics | ○ | base variant frozen scores only (29.4); no centering or offset test |
| A matched shared-weight linear model with and without normalization | ○ never run | needed for the RevIN-MLP vs ResidualLinearAR confound the takeaway mentions |

The core argument can be drafted now (centering recovers much of the plain-LSTM and attention failure, not all; level handling alone is not enough). The mechanism claims wait for the offset test; the final RevIN-MLP and PatchTST numbers wait for the common-range rescoring.

### 29.6 Open items

(a) Offset-response tests for the 1D/2D checkpoints (minutes of compute), then RevIN-MLP, PatchTST and STS-PredNet. (b) TSS-LCD finals (UC), then tuned vs authors' defaults on the test range. (c) Rescoring (383 left). (d) Time-LLM 1-minute: 11 finals left, n=3 per cell already (decision pending). (e) Analyze the external transfer results (1,022 jobs ok, unanalyzed). (f) STS-PredNet 2400-2600 base scoring and the choice of retrained variants (paused by the user). (g) ResidualConvLSTM tuning (4 trials left) and its spatial cells; the neighbour ablation for sharper maps and for neural models. (h) Unfinished S5 spatial cells after the A100 lease; lookback-mean and S4 permuted runs (partial numbers in 28.3). (i) 48 commits were pushed at 16:3x; the 6 commits since then are local.


## 30. State at 2026-11-10 18:25 UTC: the A100 and old H100 leases ended; what reached S3; the rescoring fix; the decision pending

### 30.1 Leases

Both the A100 cluster and the old H100 pair ended at 18:00 UTC (at 18:01 the A100 head closed the connection and the old H100 had no route). Results are written to S3 as each job finishes, so finished work is safe; lost is only work in flight (Time-LLM finals and spatial cells mid-run) and the **driver logs after 14:35 UTC**: the archives made at 14:35 (`s3://spectrum/job-logs/{tacc,h100,final,uc}/joblogs-20261110T1435Z.tgz`) are the last copies of the A100 and old H100 logs. Two more archives were attempted at 18:01 and could not be fetched. Only the **final H100 (until 2026-11-14 01:49 UTC) and UC (lease end unknown)** remain.

### 30.2 What reached S3 from the expired clusters (counted at 18:05-18:20)

| item | count |
|---|---|
| Time-LLM 1-minute finals (30 = 2 bands x horizons 1/15/60 x 5 seeds) | **20 finals**, at least 3 seeds in every cell: 600-800 h1 {40,41,44}, h15 {40,41,42,44}, h60 {40,41,42}; 2400-2600 h1 {40,41,44}, h15 {40,41,42}, h60 {40,41,42,43}. The 10 missing finals need about 52 GB of VRAM each (only H100s; UC's 24 GB cards cannot) |
| Tuned 15-minute 4D linear spatial cells (57 per band and model: 19 variants x 3 seeds) | LinearAR 4D: 50 per band; ResidualLinearAR 4D: 47 per band. 17 of 19 variants have results in both bands; **`k3-1` and `k3-2` (the 3-receiver subsets) are missing**, and some seeds of the S5 variants are missing; they resume with `spatial15 --tuned --skip-done` (prefix `spatial15-tuned-oct2026`) |
| Direct-model rescoring records (`rescored/`) | 1,315 |
| ES repro (`es-repro-seeds/`) | **11 of 90** records (all on the final cluster) |

### 30.3 Correction to 29.2: node pinning was not fixed by `soft=True`

I wrote that making the affinity soft fixed the stalled rescoring. It did not: at 17:58 the rescore driver (`claude-uc-rescore-direct15-2`) had made no progress for 32 minutes, with 8 `rescore_entry` tasks in `PENDING_NODE_ASSIGNMENT` (`node_id: null`, 0.25 GPU, 4 CPU, 6 GiB) while two nodes had 0.5 GPU free. Ray's soft node affinity still waits on a busy pinned node. The real fix is the new option **`rescore --no-pin`** (commit "rescore: --no-pin"), which drops the affinity (it only existed to dodge a numpy mismatch on one A100 host, not an issue on UC). Restarted at 18:00 as `claude-uc-rescore-direct15-3`: 18 completions in the first 2.5 minutes, 0 failed. Progress at 18:22: **329 of 426 left**, about 1.7 per minute; the remaining jobs include the heavy ones (MOMENT and TimeRAN, about 230 s each), so my estimate is 2-3 more hours (not measured). The same hard pinning is probably why the A100 rescoring crawled earlier (5 per hour), which I had put down to starvation.

### 30.4 Cluster state (18:22 UTC)

| Cluster | GPUs | State |
|---|---|---|
| UC | 8 of 8 reserved, all at 100% | TSS-LCD finals (5 seeds, one GPU each, started 16:20, no finish yet; RTX 6000 timing unknown, H100 trials took 4,900-9,300 s), ResidualConvLSTM tuning (4 trials left; 600-800 trials take about 13 h), rescoring (above). The CPU-only `claude-uc-deepspred-acf-probe-1` (not mine) is no longer listed |
| Final H100 | 4 of 4 reserved, all at 100%; 6 tasks pending | ES repro (79 left per its log; last line unchanged for about 25 minutes), spectral ablations 15-minute (300 of 321) and 1-minute (180 of 218) |

### 30.5 Decision pending: the ES repro is too slow to finish

11 records in about 17 hours (01:30 to 18:20), 4,400-5,500 s per run, because the two ablation drivers and the earlier TSS-LCD trials share the four final-cluster GPUs. At that rate 79 more runs would not finish before the lease ends on 11-14. It is the stderr evidence for the ElectroSense replication in part 5. Proposed: stop the 1-minute spectral ablation driver on the final cluster (the A100 half of that grid is gone, it is the slowest, finished cells stay in S3 and a restart skips them), which would give the ES repro about 6 more slots. Not done; waiting for a go.

### 30.6 Open items (changes since 29.6)

(a) ES repro scheduling (30.5). (b) Time-LLM 1-minute: 20 of 30 finals; the other 10 need an H100 (final cluster). (c) Resume the S5 spatial cells (`k3-1`, `k3-2` and missing seeds). (d) The rescoring finish (2-3 h) gates the final PatchTST, RevIN-MLP, OFA and Time-LLM numbers for parts 1, 2 and 5. (e) The constant-offset response test for part 2 (not started). (f) TSS-LCD finals, ResidualConvLSTM tuning, external-transfer analysis, STS-PredNet 2400-2600 (as 29.6). (g) 8 local commits are unpushed.


## 31. State at 2026-11-10 20:10 UTC: new GPU nodes, the offset-response results, TSS-LCD finals, and what the paper's model list is missing

### 31.1 New capacity (verified read-only by me; set up by another agent)

| item | state |
|---|---|
| Final cluster | now **6 H100s**. Hosts `ray-final-5` and `ray-final-6` run **`final_ray-worker-extra`** (Swarm label `ray.final-extra=true`; hosts 1-4 keep `final_ray-worker` with `ray.gpu=true`; the selectors are disjoint, so no duplicate workers). The extra service has the same image tag (`spectrum-ray:2.54.0-final-64bd487e533c`), environment and secrets as the standard worker plus two mounts (`/workspace/pylibs`, `/workspace/hf-cache`). Both hosts: code revision current, numpy 2.1.3, torch 2.5.1+cu124 with CUDA, transformers 4.44.2, POWDER data and ElectroSense caches present, S3 readable. Nothing needed fixing; the labels were deliberately left alone (switching them to `ray.gpu=true` would have duplicated workers and could interrupt tasks) |
| UC cluster | now **12 GPUs**: host `ray-v100` with **4x Tesla V100-PCIE-32GB** (driver 560, 80 CPUs, 125 GB RAM) runs `uc_ray-worker-extra` (label `ray.uc-extra=true`, one container with `RAY_NUM_GPUS=4`, as recommended: four containers would each advertise every CPU). Checked: code revision current, numpy 2.1.3, torch 2.5.1+cu124, compute capability 7.0, CUDA matmul works, transformers 4.44.2, S3 readable; `torch.cuda.is_bf16_supported()` returns True (emulated; the repository uses no bf16). Ray resources: 464 CPUs, 1.45 TiB memory. The Swarm ports had been blocked by a security-group rule until the other agent fixed it (my first diagnosis, "a different network segment", was wrong: `sharednet1` is `10.140.80.0/22`, so both hosts are on one subnet) |
| V100 caveats | about 3-4x slower than an H100 in fp32 (no TF32 or bf16 fast path), so they are not a drop-in for H100 work; at 32 GB they fit Time-LLM rescoring (12-15 GB per task) but not Time-LLM training (about 52 GB). Ray's `memory` resource (125 GB host) limits tasks that reserve 24 GB each: only 37 GiB were left with three STS-PredNet tasks. Tasks can be steered with Ray's automatic resource `accelerator_type:V100` (new option `rescore --require-resource`). **Decision (user, 19:30): keep existing work on the H100s and put new work on the V100s**; a Time-LLM rescoring job I had started there for seeds 43/44 was stopped |
| A100 and old H100 clusters | expired 18:00 UTC (section 30) |

### 31.2 Constant-offset response test: results (`training/ray/offset_response.py`, `evaluation/analysis/offset_response_table.py`, `evaluation/results/offset_response.csv`, `offset_gain_vs_centering.csv`; committed locally)

Frozen tuned checkpoints (1D and 2D, seeds 40-42), the October test week of two core receivers (humanities, moran), every input window shifted by c = -20, -10, -5, +5, +10, +20 dB before the checkpoint's own source normalization, forecast compared with the unshifted forecast on the same origins (thinned: stride 4 in 2D and 8 in 1D); **gain = mean forecast shift / c** (1 = level carried exactly). 128 runs on CPU, 0 failures. Gain at t+12:

| model | 600-800 | 2400-2600 | pattern |
|---|---:|---:|---|
| ResidualLinearAR 1D/2D, ResidualVanillaLSTM 1D/2D, Autoformer-CSA | 1.000 | 1.000 | exact at every offset |
| LinearAR 1D/2D | 0.98-0.99 | 0.96-0.98 | near 1, constant |
| TCN 2D | 0.64 | 0.59 | positive shifts followed (0.96-1.04), negative ones barely (0.07-0.44) |
| VanillaLSTM 1D | 0.47 | 0.41 | partial, strongest for small shifts |
| VanillaLSTM 2D | 0.13 | 0.38 | nearly ignores level |
| LSTMAttn 2D | 0.12 | 0.07 | ignores level |

Joined with the frozen-versus-centered transfer results (27.3; MAE at t+12 on guest house and law 73, A frozen to B centered, 600-800): LSTMAttn 2D 4.57 to 2.01, VanillaLSTM 2D 4.45 to 1.76, VanillaLSTM 1D 2.00 to 1.11, TCN 2D 1.11 to 1.02; the gain-1 models do not change. **Correlation between a model's gain and how much centering helps it: -0.95 (600-800) and -0.74 (2400-2600), 11 models per band.** Findings: (1) the earlier statement that the residual models and Autoformer-CSA are "level-equivariant" (section 27.3, equal numbers only) is now backed by behaviour (gain 1.000); (2) **Autoformer-CSA carries level perfectly and is still the worst of the group (2.28 dB at 600-800)**, so level handling does not ensure good forecasts; (3) centering does not make the plain 2D recurrent and attention models good: after centering they are still worse than persistence (1.44 at 600-800); (4) TCN's asymmetry is what ReLUs on frozen training-range inputs would produce (campaign TODO of 11-09), consistent with the data, not verified in the model code; it transferred well to guest house and law 73, which sit above their neighbours in level (bias +4.0 and +5.5 dB), so a target below the training range could still fail (not tested). **Not done:** the direct models (RevIN-MLP, PatchTST with and without instance normalization, DLinear, NLinear; needs the per-horizon loader), STS-PredNet (reuse `series_transform` shifting all receivers), other horizons in the paper table, the receivers' real per-bin offsets instead of a constant.

### 31.3 TSS-LCD tuned finals complete (`claude-uc-tsslcd-finals-1` on UC RTX 6000s, `tsslcd-authors-tuned/savgol-off/`)

All 16 trials and 5 finals finished. Test-range MAE against raw measurements, mean of t+1/15/60, savgol off: **1.1515, 1.1702, 1.1621, 1.1655, 1.1591, mean 1.162 (sd 0.007)**; untuned authors' defaults 1.164 (sd 0.006); lookback mean 0.950. Tuning changes the test score by 0.2% (within the seed spread); the model stays about 22% above the lookback mean. By horizon (seeds 40-43): t+1 1.14-1.15, t+15 1.16-1.18, t+60 1.16-1.18; eval-time averaging 6:3 centred scores 1.136-1.155 against raw. Each final took about 12,600-12,900 s (3.5 h) on an RTX 6000, against 4,900-9,300 s per trial on H100s. **Reporting decision to make (suggested: savgol off as the headline):** the authors' released pipeline filters by default and does not say so; with the centered filter on, the untuned model scores 1.166 on raw measurements (no difference), and scores against the filtered target (0.74-0.92) are a change of target, not accuracy; the causal filter is implemented but not run.

### 31.4 Rescoring and the Time-LLM schedule (corrections)

| item | state / correction |
|---|---|
| Direct-model 15-minute rescoring (780 checkpoints) | **110 left at 20:03**: Time-LLM 29, TimeRAN 23, MOMENT 9, OFA random-backbone 24, OFA from-scratch 25. OFA, DLinear, PatchTST (all variants), RevIN-MLP, NLinear, the residual heads and the linear 2D models are done. UC drivers: `claude-uc-rescore-part1-2` and `-part2-1` (finished), `claude-uc-rescore-rest-1` (OFA controls, MOMENT, TimeRAN, running) |
| Time-LLM 15-minute rescoring on the final H100s (`claude-final-rescore-timellm15-1`) | 21 of 50 done. **My estimate of 3-6 hours was wrong**: 4 finished in the first 75 minutes (each 25-80 minutes under contention; the 6 H100s are shared with the ES repro and two ablation drivers; 7 tasks pending), which would put the finish near 05:00 UTC on 11-11. Time-LLM rescoring does not fit UC's 24 GB cards (12-15 GB per task; out-of-memory when two share a GPU), only the H100s (and, in principle, the 32 GB V100s) |
| Time-LLM 1-minute finals | 20 of 30 (every cell has at least 3 seeds). **n=5 chosen by the user**: the 10 missing finals (600-800 h1 {42,43}, h15 {43}, h60 {43,44}; 2400-2600 h1 {42,43}, h15 {43,44}, h60 {44}) are queued as `claude-final-timellm1min-finals-1` (a CPU-only wrapper job that waits until the Time-LLM rescoring job ends, then runs `tune_direct --models timellm --max-parallel 3`; watchdog entry, 3 restarts), **lower priority than the ES repro** (Ray has no job priorities, so this is done by delayed start and a small parallel cap). About 35 H100-hours (finals took 1.7-8.5 h each); with the rescoring running to about 05:00, the finals would land late on 11-11 |
| Proposal waiting for the user | stop the two spectral-ablation drivers on the final cluster (1-minute 190 of 218, 15-minute 300 of 321): frees about three GPUs-worth for the Time-LLM rescoring and the ES repro; finished cells stay in S3 and a restart skips them |
| `rescore.py` | `--no-pin` (30.3) and `--require-resource NAME[=AMOUNT]` added; the default node pinning stays |

### 31.5 Work running on UC at 20:10 (12 GPUs)

STS-PredNet 2400-2600 base-variant scoring (`claude-uc-s15sts-eval-2400-3`, 3 seeds, one V100 each; about 1.5-3 h estimated, not yet measured), resumed S5 spatial cells (`claude-uc-s15tuned-resume-{600,2400}-1`, 17 cells per band, CPU), the remaining rescoring, ResidualConvLSTM tuning (4 trials left). The 600-800 STS-PredNet base results stand (29.4). 

### 31.6 What the LaTeX model list is missing (`latex/sigconf.tex`, `latex/model-configurations-table.tex`; asked 20:00, not edited)

Baselines section: only the lookback mean and LinearAR (plus the residual form). **Missing, and central to Parts 1 and 2:** persistence, the validation-tuned lookback **median** (the strongest 1-minute heuristic), the **EWMA**, and the per-bin **Kalman filter** (best at t+15 minutes); the median-anchored, MAE-loss ResidualLinearAR is the best simple linear model at 1 minute. Models section: **RevIN-MLP** (the centerpiece of Part 2; "introduce it here"), **PatchTST** with its three ablations (no instance normalization, joint channels, no patching), **DLinear and NLinear** (shared-weight linear references) are absent from the main text and table (DLinear and PatchTST appear only in the appendix reproduction table for ElectroSense); the table lists "Residual LinearAR + MLP" and MOMENT but the text has no paragraph for either; the OFA controls (random-backbone, trained-from-scratch) that Part 5 relies on are not described. Costs for most of these are already in `evaluation/results/appendix_model_costs.csv`.

### 31.7 Open items

(a) Direct-model and STS-PredNet offset response (V100s). (b) Time-LLM rescoring (29 left) and the 10 extra Time-LLM 1-minute finals. (c) The decision on stopping the ablation drivers. (d) Per-horizon 15-minute tables and paired intervals for Part 1. (e) Analyze the external-transfer results (1,022 jobs ok, unanalyzed). (f) ES repro (about 74 left at 7 per hour). (g) ResidualConvLSTM tuning (4 trials) and its spatial cells. (h) 15 local commits are unpushed; `OVERVIEW.md` and this document are untracked. (i) My earlier statement that a per-bin offset removes 64-84% of the interpolation error and a scalar 4-30% (28.3, 29.x) should read **59-84% and about 0-30%** across both bands (law 73 at 2400-2600 is the low end for the per-bin offset; sage point at 600-800 is slightly negative for the scalar).


### 31.8 Entropy versus model advantage, recomputed on the October benchmark (asked 20:30; the paper's figure and rho = 0.77 are from the June/July data, generated 2026-07-30, and the campaign had listed the entropy diagnostic as open)

**Code and outputs (committed locally):** `evaluation/analysis/october_entropy.py` (run on a UC worker: power-difference entropy = Lempel-Ziv phrase-count entropy rate of the 8-state quantized first differences, normalized, using the repository's `compute_region_entropy.metrics`; on the October **training** run 20261015T1434Z for the four core receivers, averaged over receivers; per 1 MHz bin for both bands and both resolutions, per annotated region for 600-800 from `plan_regions_600_800.csv`, 17 regions of which 13 are not noise floor) to `evaluation/results/october_entropy.csv` (834 rows); `evaluation/analysis/entropy_advantage_october.py` (joins it with per-bin test errors from S3: 1-minute 600-800 `oct2026-results/final-eval-v3/main/per-bin` (round 2 where it exists), 1-minute 2400-2600 `rescored/per-bin/band/2400_2600`, 15-minute `rescored/per-bin/p15tuned/<band>`, baseline = whole-window lookback mean from `median-study/sweep/<res>/<band>/per-bin/mean_k60|mean_k96.csv`; errors are means over seeds and the horizons both tables hold: 1/15/60 minutes and 4/12/24/48 steps) to `evaluation/results/entropy_advantage_october.csv`.

**Findings.** (1) Entropy ranges widely across bins (600-800, 1 minute: 0.37-0.88, median 0.52; 2400-2600: 0.37-0.59; 15 minutes: 0.42-0.69 and 0.42-0.53) and **error rises with entropy almost perfectly**: Spearman rho of entropy with the lookback mean's error 0.99 / 0.95 per bin (1 minute, 600-800 / 2400-2600), 0.92 / 0.91 (15 minutes), 0.91 per region at 1 minute (13 regions) and 0.67 at 15 minutes (the paper's July figure: 0.77). (2) **At 1 minute the models' advantage over the lookback mean is tiny at every entropy level**: for the six models that stay near the baseline (1D and 2D LinearAR and ResidualLinearAR, 1D and 2D ResidualVanillaLSTM) the gap in dB by entropy tercile (low / mid / high) is -0.004 / -0.023 / -0.030 (600-800, per bin), -0.019 / -0.034 / -0.022 (per region, flat) and -0.000 / -0.018 / -0.038 (2400-2600); the other models (plain LSTM and attention, TCN, Autoformer-CSA) are worse than the baseline in every tercile (+0.06 to +0.10 dB per bin at 600-800). So "nothing beats the baseline at 1 minute, and not because the bands are easy" is supported; the paper's wording "does not increase with entropy" is slightly too strong at bin level (a monotone but 0.03 dB effect) and right at region level. (3) **At 15 minutes the advantage grows with entropy**: -0.024 / -0.316 / -0.602 dB (600-800 per bin; relative -6% / -22% / -24%), -0.106 / -0.411 / -0.664 per region, +0.003 / -0.235 / -0.451 at 2400-2600; the other models -0.001 / -0.129 / -0.317 (600-800) and +0.021 / -0.057 / -0.138; rho(entropy, gap) between -0.45 and -0.73. **The paper's claim was made on 1-minute data and should not be extended to 15 minutes**: where there is headroom, the models use it most in the more irregular bands.

**Caveats.** The tercile grouping of "models that stay near the baseline" was chosen after seeing the per-model table (all models are in the CSV); relative gaps are inflated in quiet bins where the baseline error is tiny, so the dB rows are the ones to cite; 15-minute entropy uses the same dB thresholds on larger differences, so its values are not comparable with 1-minute entropy; high entropy is not the same as headroom (the most irregular bands may be unpredictable in principle; the Fano bound for the most irregular July channel was 0.485); the per-region analysis exists only for 600-800 (the annotated plan); 1-minute 2400-2600 1D VanillaLSTM has 1 seed; no confidence intervals. **Not done:** replace the paper's figure (`figures/entropy_model_advantage_rectangles.pdf`, July data) with an October version; the 15-minute version of the figure; the Fano predictability bound per bin.


## 32. State at 2026-11-10 ~23:10 UTC: cluster status, readiness for Parts 1 and 2 of OVERVIEW.md

### 32.1 Cluster status (read-only check at 22:58 UTC, plus a job-status sweep)

**UC (8 RTX 6000 + 4 V100):** 1.0 of 12 GPUs reserved; only `claude-uc-tune4d-neural-3` (ResidualConvLSTM, 2 trials left) is running. The 4 V100s are at 0% and most RTX 6000s at 0-18%. Finished cleanly: TSS-LCD finals (`tsslcd-finals-1`), `rescore-rest-1`, STS-PredNet 2400-2600 base scoring (`s15sts-eval-2400-3`, 3 of 3 ok; results not yet read into a table), S5 resumes at both bands (17 of 17 each), offset test (128 ok), neighbour ablations (22 + 11 ok), permuted (55 of 55) and lbm (19 of 19) runs, TimeRAN 15-minute tuning (80 finals launched; 40 TimeRAN and 40 MOMENT linear-probe final records; 0 fine-tuned records).
**Final H100 (6 GPUs):** 4.0 of 6.0 reserved; final-6 idle. Running: spectral-ablation 15-minute (310 of 321 cells) and 1-minute (190 of 218), ES repro (1 left), Time-LLM 15-minute rescoring (**2 left**, not 29 as written in 31.7), and the wrapper `claude-final-timellm1min-finals-1` (the 10 extra Time-LLM 1-minute finals, n=5), which starts when the rescoring ends.
**Conclusion:** the clusters are not busy for the next few hours; UC is nearly idle. Queue candidates are in 32.3.

### 32.2 Do we have what Parts 1 and 2 need?

**Part 1 (can models make fine-grained predictions?): the argument and the numbers for claims 1-5 exist; the write-up can start, with these gaps.**
- Per-horizon 15-minute table for the direct models: not built (data exist).
- Time-LLM: 15-minute rescoring nearly done (2 left); the 1-minute rows rest on n=3-4 until the 10 queued finals run (they wait on the rescoring).
- Paired blocked confidence intervals for the "ties" claims: not computed (the neighbour-value script has the bootstrap to reuse).
- Entropy figure: October version with a 15-minute panel not yet drawn (numbers exist in `october_entropy.csv`).
- Second test week and July data: not synced/converted, so no results there (collection ended 20:00 UTC; waiting for the user to confirm the files are final).

**Part 2 (level pathways): the offset-gain table and the frozen-versus-centered numbers exist for the models in the offset test; not complete.**
- Offset response of RevIN-MLP, PatchTST (with/without instance norm), DLinear, NLinear: loader not built.
- STS-PredNet offset response (`series_transform` can be reused) and STS 2400-2600 base numbers (scored, not tabulated).
- Matched shared-weight linear control: never run.
- Frozen-versus-centered at all horizons with intervals; stage C of the transfer protocol.
- External-site transfer (COSMOS, ARA, AERPAW): targets frozen and runs finished earlier, results not analyzed; July and new-frequency targets not run.

### 32.3 Queue candidates (awaiting the user's choice)

(1) Direct-model and STS offset tests on the V100s (build loader). (2) Sync and convert the second test week, then frozen transfer on it and on July. (3) Analyze the external-transfer results (CPU). (4) Part 1 per-horizon tables and paired intervals (CPU). (5) STS-PredNet retrained variants: paused by the user, not queued. (6) Time-LLM 1-minute finals will start by themselves.

Local commits remain unpushed; `OVERVIEW.md` and this file are untracked.


## 33. State at 2026-11-10 ~23:45 UTC: TSS-LCD at 15 minutes and with the causal filter, STS-PredNet variants relaunched, external-transfer and neighbour analyses

### 33.1 TSS-LCD: 15-minute run and causal Savitzky-Golay (requested 23:00)

**Decision (user): use the authors' defaults, not the tuned winner.** The tuned winner (latent 64, learning rate 4.88e-5, batch 128, masking 0.1) improved validation MAE by 0.6% (1.1229 vs 1.1300) and test MAE by about 0.2% (1.162 vs 1.164) at 1 minute, so the new runs use the authors' defaults (latent 32, 1e-4, batch 256, masking 0.1), 300/300/2000 epochs, savgol off headline, no tuning at 15 minutes.
**Code (committed locally, synced to UC; `training/ray/tsslcd_authors.py`):** new `PROFILES` (`1min` unchanged, `15min`), flags `--resolution` and `--hparams`. 15-minute profile: reads `data/powder15` (converted on the worker by `ensure_powder15`), **lookback 100, future 48**, horizons 1/4/12/24/48. Lookback 100 rather than the benchmark's 96 because the authors' frequency encoder uses the lookback as an attention dimension with 5 heads (96 is not divisible by 5; the first launch failed with that assertion). The test partition is loaded `lookback - 1` blocks early, so the 625 forecast origins per window set are exactly the benchmark's (first origin = first block of the test week) whatever the lookback. Partition ranges are now ceil/floor-aligned to the block grid (first launch found no gap-free run because 14:34 is not on the 15-minute grid). Eval smoothing at 15 minutes: off only.
**Launch history (all on UC):** `-1` (grid bug), `-2` (heads assertion), `-3` (GPU shared 0.5 each: CUDA OOM, two processes of about 11.7 GB on a 24 GB card; stopped), **`claude-uc-tsslcd15-default-4`** (1.0 GPU each, 10 runs = savgol off and causal x seeds 40-44, prefix `tsslcd-authors-15min-default`; running at 23:30, no failures logged). **`claude-uc-tsslcd1-causal-1`**: 1-minute causal savgol, authors' defaults, 5 seeds, eval settings 1:1:centered and 6:3:causal, prefix `tsslcd-authors-full` (beside the existing off/centered runs). ETA not measured: the 1-minute finals took about 13,300 s on an RTX 6000; the 15-minute training set has about 19x fewer windows, so the 15-minute runs should take far less, but memory per run is higher than at 1 minute.
**Not run:** tuned winner at 15 minutes (decision above); causal at 1 minute with the tuned winner.
**Caveat to state:** the 15-minute lookback is 25 hours (100 steps) for TSS-LCD against 24 hours for the other models; the savgol window (51) at 15 minutes spans 12.75 hours, unchanged from the 1-minute setting.

### 33.2 STS-PredNet retrained variants relaunched on the H100s (asked 23:25)

`claude-final-s15sts-full-600-1` on the final cluster: `spatial15 --tuned --skip-done --models stsprednet --band 600_800 --experiments s2 s3 s5 --seeds 40 --max-parallel 3 --gpu-fraction 1.0`, prefix `spatial15-tuned-oct2026`. 18 runs over 19 variants (the base variant `core4-p2` already has results from the tuning finals). Cost per cell on an H100 not measured (UC RTX 6000: about 3 h per cell). 2400-2600 variants are not launched yet; they follow once 600-800 finishes or GPUs free up (the two spectral-ablation drivers are close to done).

### 33.3 External-site transfer (COSMOS, AERPAW, ARA), first analysis (`evaluation/analysis/transfer_external_table.py`, `evaluation/results/transfer_external_stage_ab.csv`)

2,090 summaries (1,026 stage A, 1,026 stage B, 38 baselines = 19 targets x 2). MAE in dB at t+12, mean over targets with seeds averaged first; same-band transfer; A = frozen, B = per-window centering. Lookback-mean baseline in parentheses.
- **Residual and linear models carry over to new sites and beat the lookback mean almost everywhere:** ResidualLinearAR 1D/2D at AERPAW 600-800 0.91/0.99 (lbm 1.25), 2400-2600 0.77/0.81 (0.95); ARA 600-800 1.31/1.37 (1.54), 2400-2600 1.30/1.31 (1.45); LinearAR 1D similar. Centering changes residual models by 0.000 (they are level-invariant by construction).
- **Plain LSTM/attention fail when frozen and centering rescues only part:** VanillaLSTM 2D 20.3 (AERPAW 600-800), 13.6 (ARA 600-800), 23.5 (COSMOS 600-800); LSTMAttn 20.2/13.8/23.9. After centering 2.6/2.6/0.8 and 2.3/2.3/1.2, still above the lookback mean at AERPAW and ARA. VanillaLSTM 1D frozen 13.3 to 1.05 centered (AERPAW 600-800), close to the lookback mean 1.25; ResidualVanillaLSTM is level-invariant and unchanged.
- **Autoformer-CSA is poor everywhere and not helped by centering** (2.7 to 6.0 dB frozen, B equal to A): its failure is not a level offset.
- **COSMOS is nearly static** (lookback mean 0.087 dB at 2400-2600, 0.43 at 600-800): no learned model beats the lookback mean at 2400-2600 (residual models 0.11-0.14); that site says nothing about forecasting skill.
- Caveats: 2 targets (COSMOS), 3 (AERPAW, one wideband file sliced to the bands), 4-5 (ARA) per band; AERPAW is 2022 data; AERPAW results must be cut to the shared window when reported; intervals are not computed; ARA row wording in the paper table mismatches the frozen targets; the July collection and the second test week are not scored.

### 33.4 Neighbour value, rerun with the complete data (`evaluation/results/neighbor_value_compare.csv`, `neighbor_value_ablation.csv`)

Base 15-minute 4D models (map of the four core receivers; scored at the physical receivers on the same origins), pooled over receivers, MAE mean over horizons 4/12/24/48, paired block bootstrap:
- **The 4D models beat persistence and the 4D lookback mean** (600-800: 0.90-0.91 vs 1.27 and 1.22; 2400-2600: 0.94 vs 1.13 and 1.22) **but are worse than the per-receiver 1D/2D linear models** that see only the receiver's own history: +0.10 to +0.13 dB at 600-800 (95% intervals 0.09-0.14) and +0.25 to +0.29 dB at 2400-2600.
- **Breaking the neighbours barely changes the 4D error:** replacing neighbours with their level costs +0.009-0.012 dB (600-800) and +0.011-0.012 (2400-2600, interval includes 0); shifting them by 52.5 hours costs +0.021-0.024 (about 2.5%); the lookback mean (control) does not move. So the 4D linear models use neighbour dynamics only marginally and are not better than a receiver's own history.
- Limits: linear and residual-linear 4D models only (ConvLSTM, ResidualConvLSTM, STS-PredNet not in this ablation); ResidualConvLSTM tuning still has 2 trials left.

### 33.5 Open items

(a) Results of the TSS-LCD 15-minute and causal runs; STS-PredNet 600-800 variants (then 2400-2600). (b) Direct-model offset loader (V100s). (c) Second-week and July data: waiting for the user to confirm the files are final. (d) Part 1 per-horizon tables and paired intervals; STS-PredNet 2400-2600 base table; Part 3 ablation analysis once the two spectral drivers finish. (e) ES repro (1 left), Time-LLM rescoring (2 left) and the 10 Time-LLM 1-minute finals.

### 33.6 Correction and first result: the spectral-context ablation is complete for seeds 40-42 (23:55 UTC)

**Correction to 32.1 and the 23:45 summary:** the spectral-ablation records are NOT partial. `oct2026-spectral-context/` (1 minute) and `oct2026-spectral-context-15min/` hold **390 of 390 ok records each for seeds 40-42** (5 models x 13 non-noise regions x 2 conditions x 3 seeds; 0 failures). The "N of M cells" counters of the two `--reverse` drivers (310 of 321 and 190 of 218) were not a measure of what is missing; why the drivers still run is not explained (they may be redoing cells the forward drivers already wrote). Records for seeds 43-44 are older and incomplete (about 35-39 per resolution) and are not used.
**Analysis (`evaluation/analysis/spectral_ablation_final.py`, `evaluation/results/spectral_ablation_final.csv` and `_by_region.csv`):** effect of removing the out-of-region bins (`region_only - full_context`, MAE in dB, mean over horizons; each region counted once, seed-averaged, bootstrap over the 13 regions). > 0 means context helps.
| model | 1 min delta (95% interval; regions hurt of 13) | 15 min delta (95% interval; regions hurt of 13) |
|---|---|---|
| ResidualVanillaLSTM | +0.007 (0.004 to 0.010; 12) | +0.029 (0.021 to 0.037; 13) |
| Autoformer-CSA | +0.027 (0.018 to 0.037; 13) | +0.018 (-0.011 to 0.047; 7) |
| TCN | +0.003 (-0.001 to 0.009; 8) | -0.002 (-0.025 to 0.019; 8) |
| LSTMAttn | **-0.152** (-0.246 to -0.064; 3) | -0.036 (-0.221 to 0.184; 4) |
| VanillaLSTM | **-0.154** (-0.302 to -0.006; 3) | **-0.827** (-1.204 to -0.477; 2) |
**Reading:** for the models that work (ResidualVanillaLSTM, Autoformer-CSA, TCN) removing all out-of-region bins changes the error by 0.0-0.03 dB (0.3-2.2%), small and mostly in the direction of context helping; for the plain LSTM and LSTMAttn it *improves* the forecast (VanillaLSTM -28% at 15 minutes), i.e. their full-band inputs hurt. So the models take at most a small gain from spectral context, which agrees with the strong channel-independent baselines. Caveats: inherited 1-minute hyperparameters (15-minute runs not tuned), five 2D models only (the LinearAR no-cross-bin control and the PatchTST joint-versus-independent comparison are in other tables), `region_only` replaces other bins with noise-floor values and trains on the region's loss only, intervals are over 13 regions (not independent draws), the matched strong cross-frequency model was never run.

### 33.7 Spectral drivers stopped; direct-model offset test and new TSS-LCD runs tabulated (00:40 UTC)

**Spectral drivers:** `claude-final-spectral-1min-r` and `-15min-r` stopped at 00:26 UTC with the user's approval (all 390 cells per resolution had ok records since 23:37; the drivers held stale to-do lists and nothing had been written for 50 minutes) and removed from `training/ray/watchdog/final.json`.
**Direct-model offset response** (`evaluation/analysis/offset_response_table.py --prefix offset-response-direct-oct2026/`, `evaluation/results/offset_response_direct.csv`; 120 runs per horizon = 10 models x 2 bands x 2 receivers (humanities, moran) x 3 seeds; checkpoints are the 15-minute tuned direct finals; the gain is the forecast shift divided by the input offset at t+12 and t+48; same definition as 31.x):
- **Gain 1.000 at every offset and both horizons:** RevIN-MLP, PatchTST, PatchTST joint-channel, PatchTST without patching, NLinear, ResidualLinearAR 2D, ResidualNonlinear 2D. Each has a per-window level pathway (instance normalization, last-value subtraction or a residual anchor).
- **PatchTST without instance normalization (`patchtst_noinorm`): gain 0.13-0.67, falling with |c| and asymmetric** (600-800, t+12: 0.40 at -20, 0.67 at -5, 0.65 at +5, 0.38 at +20; t+48 at +20: 0.135 at 600-800). The only direct model that saturates: removing the one normalization step turns a level-carrying model into one that largely ignores a level change.
- **Linear models without an anchor:** LinearAR 2D 0.95-0.99, DLinear 0.94-1.01 (constant across c, as linear maps must be).
- Caveats: two receivers, three seeds, thinned origins (stride 4); 15 minutes only; OFA, Time-LLM, MOMENT, TimeRAN and STS-PredNet not tested (STS can reuse `series_transform`).
**TSS-LCD, new runs** (authors' defaults, 5 seeds, raw-target MAE, no eval smoothing; `evaluation/results/tsslcd_new_runs.csv`):
- **15 minutes, 600-800** (625 test windows, lookback 100): savgol off mean-over-horizons **1.157 (sd 0.059)**; per horizon t+1/4/12/24/48 = 1.118 / 1.130 / 1.163 / 1.207 / 1.167. **Causal savgol: 1.108 (sd 0.063)**; 1.048 / 1.064 / 1.130 / 1.203 / 1.096. For comparison (tuned baselines, same range) the lookback mean is 0.909 over the five horizons (0.482 / 0.624 / 0.983 / 1.258 / 1.197), the median 0.883, the Kalman filter 1.069. TSS-LCD is about 2x worse than the heuristics at t+1 and t+4, about equal at t+48 (1.167 vs 1.197 lookback mean), and 22-27% worse than the lookback mean averaged over horizons. The causal filter helps by about 4% (1.157 to 1.108) but does not close the gap.
- **1 minute, 600-800** (9,888 windows): off 1.164 (existing), centered 1.166; **causal at 1 minute is still running** (`claude-uc-tsslcd1-causal-1`, no records yet).
- Caveat: the 15-minute training set has about 500 windows, which is a very small training set for this model; its 15-minute lookback is 100 steps, not 96; defaults, not tuned.


## 34. State at 2026-11-11 ~01:30 UTC: Part 1 per-horizon 15-minute table, tuned vs inherited at 15 minutes, latest numbers in one place

### 34.1 Part 1 per-horizon 15-minute table (`evaluation/analysis/part1_per_horizon.py`, `evaluation/results/part1_per_horizon_15min.{csv,txt}`)

Test MAE in dB on the common range (2026-10-27 20:00 to 11-03 20:00 UTC, 625 origins), pooled over the four core receivers, mean over seeds (5 unless stated; heuristics deterministic). Horizons in 15-minute steps; **mean_4h = mean over t+4/12/24/48** (the paper convention; t+1 is shown separately; TimeRAN/MOMENT linear-probe have no t+1). Direct = one model trained per horizon; rolled-out = the tuned one-step 1D/2D checkpoints rescored on this range.
**600-800 MHz:**
| model | t+1 | t+4 | t+12 | t+24 | t+48 | mean_4h |
|---|---|---|---|---|---|---|
| RevIN-MLP (direct) | 0.467 | 0.577 | 0.718 | 0.810 | 0.868 | **0.743** |
| OFA | 0.473 | 0.585 | 0.723 | 0.814 | 0.874 | 0.749 |
| OFA from scratch | 0.473 | 0.582 | 0.719 | 0.824 | 0.880 | 0.751 |
| Time-LLM | 0.471 | 0.583 | 0.728 | 0.819 | 0.878 | 0.752 |
| PatchTST | 0.476 | 0.591 | 0.727 | 0.824 | 0.878 | 0.755 |
| OFA random backbone | 0.479 | 0.587 | 0.737 | 0.830 | 0.883 | 0.759 |
| ResidualNonlinear 2D (direct) | 0.480 | 0.590 | 0.730 | 0.829 | 0.908 | 0.764 |
| MOMENT linear probe | n/a | 0.600 | 0.740 | 0.838 | 0.884 | 0.766 |
| NLinear / DLinear | 0.479 / 0.480 | 0.604 / 0.609 | 0.753 / 0.761 | 0.854 / 0.861 | 0.909 / 0.906 | 0.780 / 0.784 |
| ResidualLinearAR 2D (direct) | 0.490 | 0.607 | 0.759 | 0.857 | 0.930 | 0.788 |
| TimeRAN linear probe | n/a | 0.624 | 0.765 | 0.856 | 0.914 | 0.790 |
| ResidualLinearAR 1D / LinearAR 1D (rolled out) | 0.481 / 0.479 | 0.600 / 0.598 | 0.774 / 0.773 | 0.892 / 0.892 | 0.936 / 0.938 | 0.800 |
| ResidualLinearAR 2D (rolled out) | 0.498 | 0.605 | 0.782 | 0.903 | 0.946 | 0.809 |
| PatchTST without instance normalization | 0.553 | 0.640 | 0.797 | 0.901 | 0.944 | 0.820 |
| TCN 2D (4 seeds) | 0.573 | 0.680 | 0.816 | 0.911 | 0.956 | 0.841 |
| VanillaLSTM 2D | 0.706 | 0.774 | 0.908 | 1.014 | 1.213 | 0.977 |
| **median (heuristic)** | 0.493 | 0.641 | 0.974 | 1.152 | 1.155 | **0.980** |
| lookback mean | 0.482 | 0.624 | 0.983 | 1.258 | 1.197 | 1.016 |
| LSTMAttn 2D | 0.743 | 0.846 | 0.979 | 1.112 | 1.272 | 1.052 |
| ResidualVanillaLSTM 2D | 0.641 | 0.722 | 0.959 | 1.262 | 1.542 | 1.121 |
| TSS-LCD, causal savgol | 1.048 | 1.064 | 1.130 | 1.203 | 1.096 | 1.123 |
| TSS-LCD, savgol off | 1.118 | 1.130 | 1.163 | 1.207 | 1.167 | 1.167 |
| Kalman | 0.462 | 0.621 | 0.983 | 1.446 | 1.832 | 1.220 |
| persistence | 0.505 | 0.659 | 0.999 | 1.456 | 1.864 | 1.244 |
| Autoformer-CSA 2D | 0.999 | 1.000 | 1.200 | 1.361 | 1.525 | 1.272 |
(1D VanillaLSTM and ResidualVanillaLSTM at 600-800 have only t+1 so far: their t+4..48 rescoring is in `claude-uc-rescore-fullrange-1`.)
**2400-2600 MHz (mean_4h):** RevIN-MLP **0.616**, OFA 0.622, OFA random backbone 0.625, Time-LLM 0.626, OFA scratch 0.628, PatchTST 0.629, TimeRAN linear probe 0.633, NLinear 0.646, PatchTST without instance norm 0.647, MOMENT linear probe 0.650, ResidualLinearAR 1D 0.662, ResidualLinearAR 2D direct 0.664, LinearAR 1D 0.666, ResidualVanillaLSTM 1D 0.670, DLinear 0.671, ResidualLinearAR 2D rolled out 0.688, TCN 0.688, VanillaLSTM 1D 0.762; **median 0.870**, lookback mean 0.896, VanillaLSTM 2D 0.995, LSTMAttn 1.022, Kalman 1.090, persistence 1.116, ResidualVanillaLSTM 2D 1.149, Autoformer-CSA 1.155.
**Reading:** at 15 minutes the learned direct models beat the best heuristic (median) by about 24% at 600-800 (0.743 vs 0.980) and 29% at 2400-2600 (0.616 vs 0.870); the advantage appears at t+12 and grows to t+48 (0.868 vs 1.155 at 600-800), while at t+1 and t+4 the heuristics (Kalman 0.462 / 0.621 at 600-800) tie or beat every learned model (RevIN-MLP 0.467 / 0.577). The direct models with a per-window level pathway (RevIN-MLP, OFA family, PatchTST, NLinear, residual heads) sit within 2% of each other (0.743-0.764); OFA pretrained, random-backbone and from-scratch controls are within 1% (no visible benefit of language pretraining); Time-LLM is 1.2% behind RevIN-MLP. TSS-LCD is 51-59% worse than the median at 600-800 (1.123 / 1.167 vs 0.980) and equal to the best heuristic only at t+48. TimeRAN and MOMENT are linear-probe records only (no fine-tuned records exist).

### 34.2 Tuned vs inherited hyperparameters at 15 minutes (same range, rescored: `rescored/p15` inherited 1-minute settings vs `rescored/p15tuned` tuned at 15 minutes; MAE mean over t+4/12/24/48; answer to "are the 15-minute hyperparameters very different?" by result, not by parameter values, which were not compared)

Models that work change little and slightly for the worse: LinearAR 1D +1.8% / +2.9%, ResidualLinearAR 1D +2.6% / +2.7%, ResidualLinearAR 2D +3.9% / +2.4% (2400-2600 / 600-800), LinearAR 2D +0.3% / -4.7%. Tuning rescues weak models: **TCN 2D -39% / -32%** (1.134 to 0.688; 1.240 to 0.841), VanillaLSTM 2D -6% / -17%, VanillaLSTM 1D -7% (2400-2600). Some get worse: ResidualVanillaLSTM 2D +27% / +3%, LSTMAttn 2D +14% / +6%, Autoformer-CSA +9% / +5%. Not checked: whether the tuned-worse cases are validation overfitting or noise. The spectral ablation and the TSS-LCD 15-minute runs use inherited / authors' settings (no 15-minute tuning); TSS-LCD tuning at 15 minutes (about 16 minutes per run) is possible but not queued (awaiting the user).

### 34.3 Latest numbers from this session in one place

- **Spectral context** (33.6): effect of removing out-of-region bins (MAE dB, 13 regions, seeds 40-42): ResidualVanillaLSTM +0.007 (1 min) / +0.029 (15 min), Autoformer-CSA +0.027 / +0.018, TCN +0.003 / -0.002, LSTMAttn -0.152 / -0.036, VanillaLSTM -0.154 / -0.827.
- **Offset response, direct models** (33.7): gain 1.000 for RevIN-MLP, PatchTST (and joint-channel, no-patching variants), NLinear, ResidualLinearAR 2D, ResidualNonlinear 2D; PatchTST without instance normalization 0.13-0.67 (saturating); LinearAR 2D 0.95-0.99; DLinear 0.94-1.01.
- **External-site transfer** (33.3): ResidualLinearAR beats the lookback mean at AERPAW and ARA (e.g. 0.91 vs 1.25 dB at AERPAW 600-800); plain LSTM/attention frozen 13-24 dB, centered 1-2.6 dB; Autoformer-CSA not helped by centering; COSMOS nearly static.
- **Neighbour value** (33.4): 4D linear models +0.10 to +0.13 dB (600-800) and +0.25 to +0.29 dB (2400-2600) worse than per-receiver models; breaking neighbours costs at most 0.024 dB.
- **TSS-LCD** (33.7): 15 minutes 1.157 (off) / 1.108 (causal) over t+1..48 (34.1 gives the mean over t+4..48: 1.167 / 1.123); 1 minute 1.164 (off), 1.166 (centered), causal pending.
- **Pooled vs per-receiver 2D at 15 minutes** (22.4, inherited hyperparameters, identical windows): pooling helps or ties for 2D models (LinearAR 2D +8.8% / +14.0% for per-receiver, LSTMAttn +13.2% / +6.5%, ResidualLinearAR 2D +1.0% / -2.2%); the 1D residual LSTM is better per-receiver (-20.3% / -5.6%). No matched 1-minute comparison exists (1-minute per-receiver and pooled-6 runs used a preliminary cutoff).
- **Cluster work in flight at 01:30 UTC:** STS-PredNet variants (final cluster, 36 cells), TSS-LCD 1-minute causal, full-range rescoring (70 checkpoints), Time-LLM 1-minute finals (queued), ResidualConvLSTM tuning (2 trials). Forks running: paired confidence intervals and the October entropy figure (results to be added as 34.4 and 34.5).

### 34.5 October entropy figure (fork, commits `be175d1`, `bda96d4`; 600-800 MHz; files `evaluation/results/figures/entropy_model_advantage_october.{png,pdf}`, `evaluation/analysis/entropy_advantage_regions_october.py`, `evaluation/plotting/plot_entropy_advantage_october.py`, data `evaluation/results/entropy_advantage_october_regions.csv`)

2x2 figure (1 minute left, 15 minutes right): top, lookback-mean MAE against region entropy; bottom, each model's MAE minus the lookback mean's per region. Viewed and checked. **Numbers:** Spearman rho of entropy with the lookback-mean error: **region level (13 non-noise regions) 0.91 at 1 minute and 0.67 at 15 minutes; bin level (200 bins) 0.988 and 0.924** (2400-2600 bins: 0.946 and 0.906). Gap to the lookback mean by entropy third (low / mid / high, bin level): at 1 minute ResidualLinearAR 2D -0.007 / -0.032 / -0.038, TCN -0.001 / -0.025 / -0.033, ResidualVanillaLSTM 0.000 / -0.005 / -0.011, while VanillaLSTM and LSTMAttn are worse than the baseline in every third (+0.11 to +0.23); at 15 minutes ResidualLinearAR 2D -0.028 / -0.368 / -0.731, ResidualLinearAR 1D -0.033 / -0.376 / -0.743, TCN -0.026 / -0.353 / -0.651, VanillaLSTM -0.43 in the top third, Autoformer-CSA +0.03 / +0.13 / +0.10. **This updates 31.8:** the top-third 15-minute advantage is -0.65 to -0.74 dB (31.8 said "up to -0.6"), and the 15-minute region-level rho (0.67) is much lower than the bin-level one (0.92), because 13 regions is few.
**Caveats:** tercile grouping chosen after seeing the per-model table (the figure shows all nine models); relative gaps inflated in quiet bins (dB plotted); no confidence intervals; 1D LSTMs missing at 15 minutes until the rescoring finishes; TCN has 4 seeds; two region points (VanillaLSTM and LSTMAttn at R1, +0.89 and +1.31 dB) are off scale in the 1-minute panel; 600-800 only.
**Paper edits needed (not applied; `latex/sigconf.tex` lines ~993-1016, the whole entropy block is currently commented out and describes the July data):** (1) uncomment it, point `\includegraphics` at `figures/entropy_model_advantage_october.pdf` (copy into `latex/figures/`), full-width figure if needed; (2) replace the caption and paragraph with the fork's drafts (rho 0.91/0.99 at 1 minute, 0.67/0.92 at 15 minutes; top-third best-model gap 0.03-0.04 dB at 1 minute, -0.03 to -0.73 dB at 15 minutes); (3) delete rho = 0.77 and the bold "Model advantage does not improve with entropy", which the 15-minute panel contradicts; (4) rewrite the takeaway box (error tracks entropy at both resolutions; at 1 minute learned models gain at most 0.04 dB at any entropy, at 15 minutes their advantage grows with it); (5) the sentences describing the old rectangle style (black lines, green and red bars) must be rewritten for the scatter.

### 34.6 Lookback mean and median: tuned (window chosen on validation) and untuned (full lookback window), both resolutions (asked 01:00)

`evaluation/analysis/baseline_table.py` has a new selection `full_lookback` (mean and median over the whole lookback: **96 steps at 15 minutes, 60 minutes at 1 minute**; read from the same sweep records, test metrics at that window) next to `per_horizon` and `one_parameter` (window chosen on the validation range). The 192 existing rows of `evaluation/results/baselines_all_horizons.csv` are unchanged (checked); 32 rows added. The per-horizon 15-minute table (34.1) now lists both.
Test MAE in dB (untuned full lookback / tuned per horizon):
| | t+1 | t+4 | t+12 | t+24 | t+48 |
|---|---|---|---|---|---|
| 15 min, 600-800, median | 1.070 / 0.493 | 1.088 / 0.641 | 1.122 / 0.974 | 1.152 / 1.152 | 1.155 / 1.155 |
| 15 min, 600-800, mean | 1.072 / 0.482 | 1.103 / 0.624 | 1.180 / 0.983 | 1.258 / 1.258 | 1.197 / 1.197 |
| 15 min, 2400-2600, median | 0.873 / 0.451 | 0.882 / 0.604 | 0.909 / 0.976 | 0.945 / 0.945 | 0.956 / 0.956 |
| 15 min, 2400-2600, mean | 0.901 / 0.440 | 0.924 / 0.603 | 0.987 / 0.963 | 1.049 / 1.049 | 0.968 / 0.968 |
Mean over t+4/12/24/48, 15 minutes: 600-800 median 1.129 untuned vs 0.980 tuned, mean 1.184 vs 1.016; 2400-2600 median 0.923 vs 0.870, mean 0.982 vs 0.896. At t+24 and t+48 the validation-chosen window is the full 96 steps, so tuned and untuned coincide there; at short horizons the full window is far worse (t+1: 1.07 vs 0.49).
| 1 min | t+1 | t+15 | t+60 |
|---|---|---|---|
| 600-800, median | 0.873 / 0.851 | 0.912 / 0.900 | 1.017 / 1.002 |
| 600-800, mean | 0.898 / 0.869 | 0.927 / 0.917 | 1.022 / 1.015 |
| 2400-2600, median | 0.692 / 0.663 | 0.734 / 0.716 | 0.851 / 0.824 |
| 2400-2600, mean | 0.714 / 0.681 | 0.743 / 0.730 | 0.847 / 0.835 |
At 1 minute the untuned 60-minute window is only 0.01-0.03 dB (about 1-4%) worse than the tuned one, so the 1-minute "ties" claim does not depend on tuning the heuristic; at 15 minutes it does (the tuned median is 13% better than the untuned median at 600-800). The tuned window is chosen per horizon; the `one_parameter` selection (a single window for all horizons) is also in the CSV. TimeRAN and MOMENT linear probes at 15-minute POWDER: done (34.1; 20 finals per model and band).


## 35. State at 2026-11-11 ~01:10 UTC: cluster status, Time-LLM finals relaunched, TimeRAN facts, what is open

### 35.1 Cluster status (nvidia-smi and `ray status` at about 00:45 UTC)

**UC:** 6.0 of 12 GPUs reserved; about 7 GPUs idle (RTX 6000 uc-3, -5, -6; two of the four V100s; uc-2 and uc-4 at an idle baseline of 18-21% with 1.4 GB). Running: TSS-LCD 1-minute causal (`claude-uc-tsslcd1-causal-1`, no records yet), full-range rescoring of 70 checkpoints (`claude-uc-rescore-fullrange-1`: 60 of the 1-minute `oct` source and the 10 15-minute 1D LSTMs at 600-800), ResidualConvLSTM tuning (`claude-uc-tune4d-neural-3`, 2 trials left). Finished since 23:00: TSS-LCD 15-minute (10 runs), direct-model offset test (240 ok), `claude-uc-offset-direct-1`.
**Final H100 cluster:** 5.6 of 6.0 reserved; only `final-2` is busy (Time-LLM at 51 GB, 100%); the other five H100s sit at 47-53% utilization with about 1 GB because the STS-PredNet cells reserve a whole GPU but use about half. STS-PredNet variants: **13 of the 36 non-base cells finished by 00:43** (49 STS result files including the 36 base-variant seeds); at that rate (about 10 per hour, estimate) the rest need about 2 more hours. Jobs: `claude-final-s15sts-full-600-1` and `-2400-1` (seed 40, S2/S3/S5, `--max-parallel` 3 and 2).

### 35.2 Time-LLM 1-minute finals: my wrapper never started them

`claude-final-timellm1min-finals-1` (the delayed-start wrapper) had a quoting error (`grep \"Status for job\"` inside `bash -c '...'` made grep read `for` and `job"` as file names), so its wait loop never exited and the 10 missing finals never started, although the 15-minute rescoring job had succeeded. **Fixed at 00:37:** the wrapper was stopped and `claude-final-timellm1min-finals-2` launched directly (`tune_direct --resolution 1min --models timellm --max-parallel 1`, 5 finals per band, using the one free H100); watchdog entry replaced. Time per final not measured; it can use more GPUs only by pausing STS cells (the user was told).

### 35.3 TimeRAN and MOMENT

Head-only variants only (`timeran_lp`, `moment_lp`): fine-tuned and random-init variants dropped on 2026-11-09 (at least 3x the cost, redundant with the OFA controls); the paper's sentence must say "head-only". 15-minute POWDER: 128 trials and 20 finals per model and band (4 horizons x 5 seeds) done and rescored. Mean over t+4/12/24/48: TimeRAN 0.790 / 0.633, MOMENT 0.766 / 0.650 (600-800 / 2400-2600), against RevIN-MLP 0.743 / 0.616 and the tuned median 0.980 / 0.870; the TimeRAN-versus-MOMENT sign flips between bands (no paired test). **Resource facts for any further run:** peak GPU memory about 5.7 GB per run (both models), 24-33 minutes per final (5.7-5.8 GB recorded in the final records), any GPU (RTX 6000, V100) can run several; each node needs `momentfm` and the cached MOMENT/TimeRAN weights (`tune_direct --setup-timeran`); the UC RTX 6000 nodes have them, the V100 host has not been checked. Not done: 1 minute, ElectroSense, transfer, offset test (the offset test needs a loader check for MOMENT).

### 35.4 Open items (all unstarted unless marked)

(a) Paired confidence intervals: fork in progress (its result will be 34.4). (b) October entropy figure: done (34.5); the paper block in `latex/sigconf.tex` still has to be uncommented and rewritten (five edits listed in 34.5). (c) TSS-LCD 1-minute causal result; TSS-LCD tuning at 15 minutes (not queued; awaiting the user). (d) STS-PredNet variants (running), then their tables; ResidualConvLSTM tuning (2 trials) and its spatial cells; 15-minute ConvLSTM/DSwinLSTM variants. (e) TimeRAN/MOMENT at 1 minute and in transfer. (f) Second test week and July data (waiting on the user's confirmation that the files are final), then the transfer runs on them. (g) Transfer evaluation and offset test for OFA, Time-LLM, TimeRAN, MOMENT and the target-trained RevIN-MLP; stage C. (h) Language-model cost timing rerun; ElectroSense metric corrections. (i) ~25 local commits unpushed; `OVERVIEW.md` and this file untracked.

### 34.4 Paired confidence intervals for the Part 1 "ties" claims (fork; commits `bc2f4bc`, `df8a2ae`, `e872e35`; `training/ray/origin_errors.py`, `evaluation/analysis/paired_intervals_part1.py`, `evaluation/results/paired_intervals_part1.csv` (1,400 rows, 280 pooled); per-origin errors on S3 under `origin-errors-oct2026/`; UC jobs `claude-uc-origins-15min-1` (808 ok), `-1min-1` (448 ok), `-baselines-2`, `-baselines1-2`)

Method: per-origin, per-horizon MAE over the 200 bins, seeds 40-44 averaged per origin, origins matched on timestamps, moving-block bootstrap (block = 1 day of origins: 96 at 15 minutes, 1,440 at 1 minute; 2,000 draws), receivers resampled independently and averaged (as in `neighbor_value.py`). Baselines recomputed on the same origins: persistence, lookback mean (96 steps at 15 minutes, 60 at 1 minute), validation-tuned median and mean (window per horizon, rolled out as in the tables), per-bin Kalman. Verification: 15-minute RevIN-MLP 0.7183 against the stored 0.7180 on the evaluator's origins; baselines match the transfer15 files exactly (15 minutes) and `baselines_all_horizons.csv` within 0.001-0.003 dB (1 minute); 40,084 origins at 1 minute, no thinning.
**1 minute, 600-800 (delta = model minus baseline, dB, mean over t+1/15/60, 95% interval; negative = model better):** vs the tuned median: PatchTST direct **+0.002 [-0.000, +0.004] (+0.2%)**, ResidualLinearAR 2D +0.006 [+0.005, +0.009] (+0.7%), LinearAR 2D +0.012 [+0.010, +0.015] (+1.3%), DLinear +0.026 [+0.024, +0.029] (+2.8%); vs the lookback mean: PatchTST -0.038 [-0.040, -0.035] (-4.0%), ResidualLinearAR 2D -0.033 (-3.5%), LinearAR 2D -0.027 (-2.9%), DLinear -0.014 (-1.4%); vs Kalman: PatchTST -0.009 [-0.010, -0.008], ResidualLinearAR 2D -0.004 [-0.005, -0.003], LinearAR 2D +0.002, DLinear +0.016. At t+15 PatchTST vs the median +0.005 [+0.003, +0.007]. **2400-2600:** vs the median PatchTST +0.008 [+0.005, +0.011] (+1.1%), ResidualLinearAR 2D +0.009 [+0.006, +0.012] (+1.3%); vs the lookback mean -0.035 and -0.034 (-4.5%, -4.4%); vs Kalman PatchTST -0.003 [-0.004, -0.001], ResidualLinearAR 2D -0.001 [-0.002, -0.000].
**Reading at 1 minute:** the best learned models are within 0.2-1.3% of the tuned median, and **the median is slightly but significantly better in almost every cell** (only PatchTST at 600-800 has an interval that includes 0): "ties" should read "within about 1% of, and not better than, the tuned median". All learned models beat the 60-minute lookback mean by 3-4.5% and the Kalman filter by at most 0.9%. (This corrects the earlier wording that the best learned model was "within about 1.5%" without intervals, and agrees with 34.6: the untuned window costs the heuristic only 1-4%.)
**15 minutes, mean over t+1/3/6/12 h, 600-800:** vs the tuned median PatchTST -0.248 [-0.276, -0.179] (-25.1%), RevIN-MLP -0.260 [-0.292, -0.192] (-26.4%), ResidualLinearAR 2D rolled out -0.196 [-0.229, -0.129] (-19.9%); vs the lookback mean -0.399 / -0.412 / -0.345 (-35.0% / -36.2% / -30.3%). **2400-2600:** RevIN-MLP -0.271 [-0.307, -0.210] (-30.9%), PatchTST -0.260 [-0.303, -0.190] (-29.7%) vs the median; vs the lookback mean -0.344 and -0.333 (-36.2%, -35.1%). DLinear, NLinear, LinearAR and the 1D rolled-out models are -18.7% to -28.1% against the median; **every learned model beats every baseline at 15 minutes with intervals far from 0** (t+12: PatchTST -27.4% at 600-800, -36.7% at 2400-2600).
**Caveats:** OFA, Time-LLM, TimeRAN and MOMENT skipped (need GPU and isolated libraries); no 1-minute RevIN-MLP checkpoints; no 1D LSTMs, TCN or Autoformer in these comparisons; intervals cover origin-sampling variation only (seeds averaged first, so seed spread is not included; receivers treated as independent; no multiplicity correction); the 15-minute per-origin set follows the transfer15 convention (625 origins per receiver, first origins use pre-test history), so absolute MAE is about 1% below the stored table; rolled-out 15-minute rows use 620 origins per receiver (2,481 pooled) with the baseline matched to the same origins.


## 36. State at 2026-11-11 01:43 UTC: transfer coverage, transfer jobs on UC, TSS-LCD at 2400-2600 and 15-minute tuning

### 36.1 Transfer coverage against Table 2 of the paper (user question; "week 2 of the test is not available yet")

Regimes: **A** frozen (checkpoint normalization), **B** per-window centering, **C** causal normalizer refit from an earlier calibration interval. Before this session: A and B existed only for the 11 core 1D/2D models (LinearAR and ResidualLinearAR 1D/2D, VanillaLSTM and ResidualVanillaLSTM 1D/2D, LSTMAttn, TCN, Autoformer-CSA) on guesthouse/law73 and on the 19 COSMOS/AERPAW/ARA targets; C had never been built; no direct or large model (RevIN-MLP, PatchTST, DLinear, NLinear, OFA, Time-LLM, TimeRAN, MOMENT), TSS-LCD or STS-PredNet had any transfer result. Row status: Reference (in distribution) done; New collection points A/B done for the 11 models; COSMOS/AERPAW/ARA A/B done for the 11 models; **Later interval (11-03 to 11-10): not possible, the week-2 data are not available**; Separate collection (June-July) and New frequency (800-1000, 5725-5925 MHz, July): not run. **Paper text issues:** the New-frequency row cannot be scored by the same-band protocol (needs a normalization/applicability decision, task B is reporting options); the ARA row text (600-800 only, 07-02 to 07-04, ames/horticulture) does not match the frozen targets; the Later-interval row should read "planned" until the data exist.

### 36.2 Transfer work launched on UC (two forks; neither has final tables yet)

**Task A (all final models, regimes A/B/C)**, commits `7ade95c`, `38a4544`, `17287dd`, `d52a962`: `training/common/transfer_regimes.py` (+`tests/test_transfer_regimes.py`, 4 tests), `training/ray/transfer_all.py` (rolled-out 1D/2D checkpoints and the horizon-specific direct finals; targets guesthouse/law73 and the external targets; output `transfer-all-oct2026/<regime>/...`), `evaluation/analysis/transfer_all_table.py` (-> `evaluation/results/transfer_all_regimes.csv`); `transfer15.evaluate_frozen` gained optional `normalization_override` and `origin_horizon` (existing calls unchanged).
**Stage C definition (fixed before scoring):** calibration interval = the loaded rows before the first scored origin, i.e. the first origin's 96-block (24 h) history (guesthouse/law73: the 24 h before 2026-10-27 20:00; external targets: the first 96 blocks); per-bin mean and standard deviation refit on that interval alone and used in place of the checkpoint's statistics for the whole scored interval; weights frozen; standard deviation floored at 0.25 x the source's (declared up front, not tuned); "Cm" = mean-only refit; origins identical across regimes (direct single-horizon models scored on the shared 48-step-future origins: 625 POWDER, frozen counts for externals).
**Models with their own per-window normalization** (RevIN-MLP, PatchTST and its joint/nopatch variants, OFA and controls, Time-LLM, MOMENT, TimeRAN) run in regime A only by default; B and C are identical to A for them by construction (checked at guesthouse 600-800, seed 40, t+12: A and C agree to three decimals for OFA 0.686/0.686, MOMENT 0.683/0.683, TimeRAN 0.694/0.694, Time-LLM 0.694/0.694). B and C are run for PatchTST without instance normalization, NLinear, DLinear, LinearAR 2D, the residual heads and the 11 core models.
**Smoke checks:** A and B reproduce the existing numbers exactly for the 2D model, the 1D model within 1e-5. Example (MAE dB, t+12, A / B / C / Cm): LinearAR 2D guesthouse 600-800 0.774 / 0.736 / 0.734 / 0.736; AERPAW CC1 600-800 1.503 / 1.184 / 1.179 / 1.183. Preliminary only: COSMOS 2400-2600 LinearAR 2D 1.160 (A) / 0.122 (B) / 0.068 (C) against a lookback mean 0.087.
**Jobs (all RUNNING at the fork's last report):** `claude-uc-tall-direct-cheap-2` (10 cheap direct models, all regimes, 816 jobs left), `claude-uc-tall-rolled-2` (C and Cm for the 1D/2D models, 1,122 jobs left), `claude-uc-tall-llm-1` (OFA, OFA controls, MOMENT, TimeRAN, A only; about 567 jobs at about 160 s each), `claude-uc-tall-timellm-1` (Time-LLM seeds 40-42, 69 jobs, 3 GPUs, about 6 h at the current pace). **UC CPU saturation:** hosts at load 30-65 on 48-80 cores (84-102 GB used on four hosts); the fork capped torch at 2 threads per job and restarted the first two jobs with `--skip-done`; no completed jobs logged for the first 6 minutes after the restart. Not covered: STS-PredNet (4D), TSS-LCD, the June-July targets. To finish: wait for the four jobs, rerun with `--skip-done` if any died, then `python evaluation/analysis/transfer_all_table.py --cache evaluation/results/transfer_all_raw.csv`.
**Task B (June-July Separate collection, A and B for the 11 models; New-frequency options):** forked, no report yet.

### 36.3 TSS-LCD: 2400-2600 and 15-minute tuning (queued 2026-11-11 01:43)

Until now TSS-LCD had only been run on 600-800 (1 minute: untuned, 16-trial tuning, tuned finals, causal running; 15 minutes: authors' defaults, 10 runs). `training/ray/tsslcd_authors.py` now has `--band` (commit `c6192e1`); `training/ray/tune_tsslcd.py` has `--resolution 15min` (same search space and draws, own prefixes, finals with eval smoothing off). **Queued on UC:** `claude-uc-tsslcd15-2400-1` (authors' defaults, savgol off and causal, 5 seeds, 10 runs, prefix `tsslcd-authors-15min-default-2400`; about 16 minutes per run); `claude-uc-tsslcd1-2400-1` (1 minute: off first, then causal, as a sequential chain so the queue is not flooded; about 3.7 h per run, about 8 h total if GPUs stay free; prefix `tsslcd-authors-full-2400`); `claude-uc-tsslcd15-tune-1` (15-minute tuning at 600-800: 16 trials, 3 parallel, then 5 finals; roughly 2 h, probably more under the CPU load). The 2400-2600 data path was not tested separately before launch (jobs started, no finished run seen yet). UC at the time: 11.25 of 12 GPUs reserved.
**Not launched:** TimeRAN/MOMENT at 1 minute (`timeran_lp`, `moment_lp` have no 1-minute records; `claude-uc-timeran-setup-2` was still running and UC CPUs were saturated); peak memory of the 15-minute finals is about 5.7 GB.
**Verified absent (do not assume done):** 1-minute TimeRAN/MOMENT; 15-minute TSS-LCD tuning at the time of the check; TSS-LCD at 2400-2600.

### 36.4 Also finished since section 35

Paired confidence intervals (34.4); TSS-LCD 15-minute runs; direct-model offset test; entropy figure (34.5). Part 1's "still needs" line in OVERVIEW.md was updated for the paired intervals.

### 36.5 Task B result: June-July "Separate collection" transfer (A frozen / B centered), finished on UC (fork; commits `0adff5a`, `48af14c`, `5e1d05a` and the table-script commit; `training/ray/transfer_july.py`, `evaluation/analysis/transfer_july_intervals.csv`, `evaluation/analysis/transfer_july_table.py`, `evaluation/results/transfer_july_stage_ab.csv`; S3 prefix `transfer15-july-2026/`; UC jobs `claude-uc-july-transfer-3` finished, `-1`/`-2` stopped and relaunched, `july-smoke-1`, `july-verify-1`)

**Targets (frozen before scoring):** 26 usable of 28, 13 per band; receivers humanities, madsen, moran, sagepoint, guesthouse, cpg, ebc; each has a 06-28 run (about 5 days; 14 targets, 325-330 origins each) and a 07-03 run (about 2.6 days; 12 usable targets, 104-110 origins each, just above the 96 minimum); the two intervals stay separate (16 h gap). Excluded: guesthouse's 07-03 run in both bands (two gaps above 10 minutes, longest segment 21 h, 0 origins). law73 and web only have a 06-30 run (not in the table row). No imputation needed (share 0.0).
**Run:** 108 checkpoints (11 models x seeds 40-44; TCN 2D 4 seeds) plus persistence and lookback mean on every target; 1,274 jobs on the last launch, 0 failed; CPU only; about 80 minutes (the 1D LSTMs take about 4 minutes per job; the first launch reserved 12 GB per task and was memory-starved). Verified: three ARA cells rerun through the new driver match the stored external results within 1e-8 MAE for A and B; 78-job smoke run passed.
**MAE at t+12 in dB, A frozen / B centered** (lookback mean / persistence in the first row; core four = humanities, madsen, moran, sagepoint, 8 targets; other = guesthouse, cpg, ebc, 5 targets):
| 600-800 | core four | other |
|---|---|---|
| lookback mean / persistence | 1.074 / 1.043 | 1.234 / 1.131 |
| ResidualLinearAR 2D | 0.799 / 0.799 | 0.794 / 0.794 |
| LinearAR 1D | 0.799 / 0.794 | 0.811 / 0.805 |
| TCN 2D | 0.955 / 0.817 | 0.863 / 0.809 |
| VanillaLSTM 2D | 1.569 / 1.224 | 4.078 / 1.629 |
| LSTMAttn 2D | 1.687 / 1.729 | 4.212 / 1.937 |
| Autoformer-CSA 2D | 1.238 / 1.238 | 2.099 / 2.099 |
| 2400-2600 | core four | other |
|---|---|---|
| lookback mean / persistence | 1.049 / 1.091 | 1.744 / 1.540 |
| ResidualLinearAR 2D | 0.681 / 0.681 | 0.964 / 0.964 |
| LinearAR 1D | 0.684 / 0.675 | 1.032 / 1.036 |
| TCN 2D | 0.706 / 0.827 | 1.131 / 1.128 |
| VanillaLSTM 2D | 1.325 / 1.373 | 4.323 / 2.851 |
| LSTMAttn 2D | 1.297 / 1.989 | 3.022 / 2.381 |
| Autoformer-CSA 2D | 1.342 / 1.342 | 2.685 / 2.685 |
**Reading:** residual and linear models transfer to the June-July collection at 0.65-0.75 of the lookback mean and centering changes them by 0.000 (residual) or at most 0.03 dB; plain LSTM and attention models are bad frozen (up to 4.3 dB) and centering helps them on guesthouse/cpg/ebc (VanillaLSTM 2D 4.08 to 1.63 at 600-800), **but centering can hurt on the core four at 2400-2600** (LSTMAttn +0.69, TCN +0.12, VanillaLSTM 1D +0.29); Autoformer-CSA is poor everywhere and unchanged by centering. Not computed: confidence intervals; cpg and ebc separately from guesthouse; the direct and large models on these targets (they need task A's evaluator: `transfer_all.job` can take these targets).
**New-frequency row (800-1000 and 5725-5925 MHz) needs a decision; not run:** the data exist only in the 07-03 run at all seven receivers (200 one-MHz bins each, 61-63 h usable, guesthouse has the 21 h gap problem); mean level -128 dBm (800-1000) and -131 dBm (5725-5925), near the noise floor, against -122 dBm at 600-800; the 200-bin shape matches and `transfer_2d.validate_target_frequencies` allows positional cross-band use, but the source statistics are per bin position and the level mismatch is 6-9 dB; the external driver refuses source and target bands that differ, so a cross-band variant needs new code in a new file. **Options:** (1) positional cross-band transfer of the 600-800 and 2400-2600 checkpoints (2 sources x 2 targets) with stage B centering for the level (scores only whether temporal structure transfers; the fork's recommendation, with the caveat that 5725-5925 is almost all noise floor); (2) the same plus stage C (calibrate the normalizer on an earlier target interval then freeze it; needs task A's stage C); (3) footnote the row as not evaluated (combined band and level transfer is a different experiment from the other rows). **Awaiting the user's choice.**
**Side note:** UC's memory reservation sat at about 1.4 of 1.45 TiB for most of the run, so jobs from different drivers compete for memory.


## 37. State at 2026-11-11 03:03 UTC: overnight setup, H100 rebalancing, what is queued, what has no code yet

### 37.1 Correction: STS-PredNet progress and the H100 timeline

My "30-60 minutes until the H100s free up" (section 35) was wrong: I had counted result files, which include non-variant files. Counting finished cells in the job logs at 02:07: **600-800 5 of 18, 2400-2600 2 of 18** (about 1.6-2 h per cell per GPU slot; STS cells use about half of each H100: 46-55% utilization, about 1 GB). Time-LLM 1-minute finals: 10 left (5 per band; about 1.5 h each by the earlier 92-minute trial median, unmeasured for finals), running one at a time since 00:37 with no final completed by 02:07.

### 37.2 Overnight setup (user: "no intervention at the time", leaving the PC off)

**Final cluster (6 H100s), all reserved 6.0 of 6.0 at 02:28.** At 02:21 I stopped (user approved the redistribution) the two STS jobs and `claude-final-timellm1min-finals-2` (losing the in-flight work: five STS cells and one Time-LLM final), then relaunched: `claude-final-timellm1min-finals-3` (all 10 finals submitted at once, no parallel cap; each reserves 0.6 GPU, so 6 run, one per H100, and 4 wait); `claude-final-s15sts-full-600-2` and `-2400-2` (seed 40, `--gpu-fraction 0.4 --max-parallel 6`, `--skip-done`: one 0.4 cell fits beside each Time-LLM task, two per GPU once Time-LLM is gone; 13 and 15 cells to run). Ray's queue rebalances without intervention; priority is not guaranteed (STS cells can take a freed GPU before a waiting Time-LLM final). Follow-ups queued as waiting jobs: `claude-final-s15sts-seeds-1` (`training/ray/scripts/run_after.py` waits for the two seed-40 STS jobs and the Time-LLM job to end, then runs STS variants for seeds 41 and 42 in both bands; script `training/ray/scripts/sts_seeds_followup.sh`; about 70 cells); `claude-final-spectral-seeds-1min-1` (186 cells) and `-15min-1` (221 cells): spectral-context ablation seeds 43-44, 3 cells at a time, cost nothing while the cluster is full. The two earlier `--reverse` spectral drivers had been stopped (33.7). `claude-final-simple1min-revin-1` (1-minute RevIN-MLP and NLinear: 128 trials, 60 finals) finished. **Watchdog** `claude-watchdog-final-1` (resubmits a FAILED driver every 10 minutes, max 4 restarts; reads `training/ray/watchdog/final.json` each cycle) now tracks all of these.
**UC (8 RTX 6000 + 4 V100), 11-12 of 12 GPUs reserved, load 22-37:** queued/running: transfer jobs from the fork (`claude-uc-tall-direct-cheap-2` 305 left, `-rolled-2` 1,089 left, `-llm-1` 464 left, `-timellm-1` 63 left), TSS-LCD 15-minute tuning (13 trials left), TSS-LCD 2400-2600 at 15 minutes (10 runs; the first done: 0.923 dB, seed 40, savgol off) and at 1 minute (5 off, then 5 causal), TSS-LCD 600-800 1-minute causal, TimeRAN/MOMENT at 1 minute (`claude-uc-timeran1min-1`, 128 trials then finals, 2 at a time; node setup finished on all hosts including the V100 host), full-range rescoring (63 left), ResidualConvLSTM tuning (2 trials; no visible progress; about 13.7 h per trial). Estimated queue depth 8-12 h (rough, CPU contention); the tail may idle (the longest chain is the 1-minute TSS-LCD 2400-2600 runs); no UC filler added (UC memory reservation about 1.4 of 1.45 TiB). UC watchdog `claude-watchdog-uc-1` (`training/ray/watchdog/uc.json`) tracks these drivers. Only failed jobs on UC are three old DeepSPred probes from 11-10.
**Leases:** final cluster ends 2026-11-14 01:49 UTC; the UC end is not recorded in this file.

### 37.3 Also finished since 36

1-minute RevIN-MLP and NLinear (they had never been run at 1 minute: the 11-09 simple-baseline search covered 15 minutes, ElectroSense and, at 1 minute, only the direct ResidualLinearAR; test windows thinned 15x as for the other 1-minute cheap models, to be rescored at stride 1 before any table needing every window); June-July transfer A/B (36.5); `tune_tsslcd --resolution 15min` and `tsslcd_authors --band` code (36.3).

### 37.4 Things with no code yet (being prepared; two forks)

(1) Offset-response test for OFA, its controls, Time-LLM, TimeRAN, MOMENT and STS-PredNet. (2) Direct and large models on the June-July targets (an adaptor from `evaluation/analysis/transfer_july_intervals.csv` to `transfer_all`). (3) Cross-band positional transfer for the New-frequency row (user decision pending; code prepared behind a flag, not run). Not planned: TSS-LCD transfer (the model is a joint four-receiver model with 800 columns, so receiver transfer does not map onto it); STS-PredNet transfer (4D).

### 37.5 Decision (user, about 03:30 UTC 11-11): spatial analysis focuses on 15 minutes

1-minute models are "a waste for spatial analysis": no further 1-minute spatial work (no 1-minute 2400-2600 spatial study, no 1-minute map-construction variants, no 1-minute neighbour or kriging runs). The finished 1-minute 600-800 study (section 24.4: 260 cells; left-out and added receivers, permuted coordinates) stays as a supporting result only. The kriging, IDW-sensitivity, leave-one-out, permuted and neighbour work is all 15-minute (tasks E and F). Paper consequence: Part 4's headline spatial evidence should be the 15-minute tables; the 1-minute table is context.

### 37.6 Task D result: code ready (not run) for the direct/large/C models on the June-July targets and for cross-band transfer (fork; commits `9cad73e`, `3e53527`, `86a0530`, `b4bdf26`; smoke outputs under `transfer-all-july-smoke/` and `transfer-crossband-smoke/` on S3, deletable; `claude-uc-julyall-smoke-2` stopped by the fork)

**1. June-July adaptor (`training/ray/transfer_all.py`, additive; `evaluation/analysis/transfer_july_all_table.py`; `tests/test_transfer_all_july.py`):** `is_core_target(t)` (only the October guesthouse/law73 targets use the core conversion; every other POWDER target, including June-July runs, converts like an external site), `targets(..., intervals, core)`, CLI flags \`--intervals\` and \`--no-core-powder\`; the 26 frozen targets (13 per band) are not redefined. The table script merges \`transfer15-july-2026/\` (the 11 core models, A/B, baselines) with \`transfer-all-july-2026/\` (C/Cm, direct and large models), splits core four from guesthouse/cpg/ebc (\`--by-run\` splits the two runs) and flags groups with missing targets as incomplete.
**Launch commands (UC; add \`--max-parallel\` as shown; all with \`--groups powder --no-core-powder --intervals evaluation/analysis/transfer_july_intervals.csv --prefix transfer-all-july-2026 --skip-done\`):** \`python -m training.ray.transfer_all run --source rolled --regimes C Cm ... --max-parallel 24\` (about 1,404 jobs; the 1D LSTMs, about 260 jobs at 4-8 minutes each, dominate: roughly 2-3 h); \`... --source direct ... --max-parallel 24\` (cheap direct models, about 1,300 jobs, 15-20 minutes); \`... --source direct --models ofa ofa_rand ofa_scratch timeran_lp moment_lp ... --max-parallel 8\` (about 650 jobs at 0.25-0.5 GPU, 3-4 h); \`... --source direct --models timellm --seeds 40 41 42 ... --max-parallel 3\` (78 jobs on 3 whole GPUs, about 4 h). Estimates only (smoke cells are tiny; the rest scaled from the October runs). Smoke (madsen 07-03 600-800, seed 40, t+12, dB): rolled LinearAR 2D C 0.607, Cm 0.614; direct RevIN-MLP A 0.596; direct LinearAR 2D A 0.628, B 0.604, C 0.597, Cm 0.603. **Not verified:** the large models on the June-July targets (a one-cell CPU smoke with \`moment_lp\` ran over 10 minutes and was stopped; the code path equals the October one task A verified).
**2. Cross-band positional transfer (\`training/ray/transfer_crossband.py\`, \`tests/test_transfer_crossband.py\`, frozen \`evaluation/analysis/transfer_crossband_intervals.csv\`): built, NOT chosen by the user, not run.** 12 of 14 usable targets (6 receivers x 2 bands: humanities, madsen, moran, sagepoint, cpg, ebc at 800-1000 and 5725-5925 MHz; the 07-03 run only; 100-110 origins each); guesthouse excluded in both bands (21 h gap, 85 blocks, 0 origins). Applies each checkpoint band (600-800, 2400-2600) positionally to each new band; A and B by default, C/Cm with \`--regime-c\`; full-normalization models run in A only. Commands: \`python -m training.ray.transfer_crossband run --skip-done --max-parallel 24\` (11 core models + 10 cheap direct models, CPU, about 1,296 rolled + 1,200 direct + 12 baseline jobs, about 1.5-3 h), then the same with \`--regime-c\` (reruns the groups whose last regime is not done, all their regimes). Output \`transfer-crossband-july-2026/from-<source band>/\`. Not covered: OFA, Time-LLM, TimeRAN, MOMENT (no cross-band path). Smoke (madsen, seed 40, t+12): LinearAR 2D from 600-800 onto 800-1000 A 0.736 / B 0.555 against lookback mean 0.542 and persistence 0.587; onto 5725-5925 (from 2400-2600) A 0.125 / B 0.088 / C 0.026 / Cm 0.088 and RevIN-MLP A 0.0088 against lookback mean 0.0082 and persistence 0.0111 (the band is at the noise floor: compare ratios to the lookback mean, not absolute errors). **Caveats:** 5725-5925 is noise floor; the 6-9 dB level mismatch makes regime A poor; per-bin source statistics are meaningless for a new band; it tests only whether temporal structure transfers, not knowledge of the new band. Converted files for the new bands are on the UC workers (\`data/ext15/powder/.../800_1000\`, \`5725_5925\`).
**Status of these runs:** not launched (UC fully reserved); they are the next UC work once the TSS-LCD and transfer queues drain.

### 37.7 Task E result: 15-minute spatial tables and the neural neighbour-ablation code (fork; commit `0ddfe34`, synced to UC)

**Code:** `evaluation/analysis/spatial15_table.py` -> `evaluation/results/spatial15_{cells,receivers,coverage,s2,s3,s4,s5,references}.csv`; `training/ray/neighbor_ablation.py` now accepts `stsprednet`, `residualconvlstm`, `convlstm` (neural models use a GPU share, `--gpu-fraction` default 0.5; linear models and the lookback mean stay on CPU; new `--origin-stride` for smoke tests; old CLI unchanged); `evaluation/analysis/neighbor_value.py` `FOUR_D` includes `stsprednet` and `residualconvlstm` when their `per_origin.npz` exist (existing outputs unchanged); `tests/test_neighbor_ablation.py`, `tests/test_spatial15_table.py` (8 passed, 1 skipped: the CLI test needs `ray`, absent locally). **Smoke not finished:** `claude-uc-nbr-smoke-sts-1` (STS-PredNet, 600-800, seed 40, every 20th origin, CPU) still running after about 30 minutes on CPU-saturated UC hosts, so the neural path is untested end to end and there is no cost estimate; output will be at `neighbor-ablation-smoke/600_800/stsprednet/seed-40/`; CPU is probably impractical for the full run (treat STS-PredNet as GPU work). **Full-run command (when capacity allows):** `python -m training.ray.neighbor_ablation --models stsprednet --seeds 40 41 42 43 44 --skip-done --gpu-fraction 0.5 --max-parallel 4` (20 cells, each 16 evaluations = 4 receivers x 4 modes); add `residualconvlstm` once its tuning finals exist. **Rerun `spatial15_table.py` after the kriging cells (task F, variant `core4-krige`) and the STS-PredNet variants finish.**
**Coverage (cells of 57 expected = 19 variants x seeds 40-42; lookback mean 19):** LinearAR 4D and ResidualLinearAR 4D 57 in both bands; STS-PredNet 8 (600-800) and 6 (2400-2600) at the time of the run; ResidualConvLSTM 0.
**S2, 600-800 (physical-receiver MAE dB, mean over t+4/12/24/48; map MAE = the model's own interpolated map, not comparable with 1D/2D):**
| Model | Variant | Map MAE | Core four in input | Guest house/law 73 absent | Added |
|---|---|---|---|---|---|
| LinearAR | power 2 | 0.546 | 0.906 | 5.319 | 2.816 |
| LinearAR | power 16 | 0.673 | 0.796 | 6.302 | 1.187 |
| ResidualLinearAR | power 2 | 0.549 | 0.904 | 5.301 | 2.286 |
| ResidualLinearAR | power 4 | 0.616 | 0.785 | 5.558 | 0.983 |
| STS-PredNet | power 2 | not scored | 1.203 | 5.508 | 5.322 |
| lookback mean | power 2 | 1.170 | 1.215 | 5.480 | 2.685 |
In-sample error depends little on the map settings except through power (4 and 16 are better, as in 28.3), while power 16 and 5x5 grids matter at the unseen receivers; **STS-PredNet barely uses an added receiver (5.3 against 2.3 for ResidualLinearAR)** and, in sample, is worse than the linear models (1.203 vs 0.904 at power 2; seed 40 only except the base cell).
**S3, 600-800, guest house left out / added back (own-history 2D ResidualLinearAR 0.746):** LinearAR 4.284 / 1.949, ResidualLinearAR 4.298 / 1.545, lookback mean 4.465 / 1.940: the left-out error is essentially the interpolation error and is 3-8 times the own-history error; law 73 and moran are worse (5.6-5.9). **S5, guest house absent (ResidualLinearAR):** 4.481 with 2 receivers, 4.508 with 3, 4.544 with 4: more receivers do not help the unseen one. **S4 permuted coordinates, 600-800:** LinearAR 0.906 to 2.636, ResidualLinearAR 0.904 to 2.761, lookback mean 1.215 to 2.902 (the parameter-free control degrades as much). The 2400-2600 tables show the same patterns (CSVs).
**Caveats:** STS-PredNet rows are seed 40 only (core power 2: 3 seeds), so no spread; the base STS-PredNet cell has no map-level number (scored from the tuning finals); own-history references are frozen transfer for guest house and law 73 and in-distribution for the core receivers; ResidualConvLSTM appears only as zero coverage.

### 37.8 Task C result: offset test code for the large models and STS-PredNet (fork; commits `3c99100`, `e30a220`, synced to UC; tests `tests/test_offset_response.py`, `tests/test_offset_response_sts.py` pass)

**Code:** `training/ray/offset_response.py` now takes `--source direct --models ofa ofa_rand ofa_scratch timellm timeran_lp moment_lp` (isolated libraries and TimeRAN weights loaded; per-model GPU share: OFA family 0.5, Time-LLM 1.0, TimeRAN/MOMENT 0.25; 24 GB; new helper `resources_for`); new `training/ray/offset_response_sts.py` shifts all four receivers' series by c dB through `series_transform` and measures the response at the physical receivers (checkpoints `tuning/powder15_4d/<band>/4d/stsprednet/weights/seed-N`); its `--offsets` and `--stride` are for smoke tests; `transfer_all.py` and `transfer15.py` untouched.
**Smoke, large models** (`claude-uc-offset-llm-smoke-1`, 600-800, humanities, seed 40, t+12, all offsets; finished; output `s3://spectrum/offset-response-llm-smoke/`; `offset_response_table.py --prefix offset-response-llm-smoke/` tabulates it): **gain 1.000 at every offset for OFA (22 s), OFA random-backbone (25 s), OFA from-scratch (55 s), TimeRAN (203 s), MOMENT (277 s) and 1.0001 for Time-LLM (384 s)**: per-window normalization carries the level exactly, as expected. One receiver, one seed, one horizon, origins thinned with stride 4 as in the existing test.
**STS-PredNet smoke: still running, no gain yet** (\`claude-uc-offset-sts-smoke-1\`: full offsets, CPU, stride 4, no output after more than 25 minutes; \`-2\`: offset +10 only, stride 40; outputs will land at \`s3://spectrum/offset-response-sts-smoke/<band>/<receiver>/4d/stsprednet/seed-40/summary.json\` and \`offset-response-sts-smoke2/...\`). STS is far slower than the other models (the 3-seed scoring took about 100 minutes on a GPU), so treat it as GPU work. **The STS gain is not guaranteed to be 1:** it relies on the checkpoint's own normalization, which does not cancel the offset.
**Full-run commands** (UC or the H100s): \`python -m training.ray.offset_response --source direct --models ofa ofa_rand ofa_scratch timeran_lp moment_lp timellm --bands 600_800 2400_2600 --receivers humanities moran --seeds 40 41 42 --horizons 12 48 --skip-done --max-parallel 8\` (144 jobs; estimate 25-35 GPU-minutes for OFA, TimeRAN, MOMENT and about 6.5 GPU-minutes per Time-LLM job, a few GPU-hours in total); \`python -m training.ray.offset_response_sts --bands 600_800 2400_2600 --seeds 40 41 42 --device cuda --gpu-fraction 0.5 --skip-done --max-parallel 6\` (6 jobs, GPU recommended, time unknown until a smoke finishes); tables with \`python evaluation/analysis/offset_response_table.py --prefix offset-response-direct-oct2026/ --horizon 12\` and \`--prefix offset-response-sts-oct2026/ --horizon 12\`.

### 37.9 Task F result: kriging code and frozen variograms built; the paired map-construction comparison NOT run (fork, wrapped up at the user's request; commits `a10c5f3`, `886f2b8`, `7de36ca`, `e15090f`, `9f530c9`, local; nothing synced to the final cluster; 44 tests pass: `tests/test_kriging.py`, `tests/test_spatial15_origins.py` and the existing spatial and map-builder tests)

**Design:** ordinary kriging with a frozen exponential variogram plus nugget (gamma(0)=0), exact at the receivers (weight 1, variance 0); weights depend only on coordinates, form an (H,W,S) array like the IDW weights, interpolation stays in dB; `training/common/kriging.py` (weights, variance, empirical semivariogram, fit); `map_builder._map_from_sites` and the frozen-grid path of `evaluate_condition` dispatch on `grid["method"]=="kriging"`; IDW untouched (tests reproduce the old formula); forecast sampling back at the receivers still uses IDW power 2 as in every other variant. **Variogram fit** (`training/ray/fit_variogram.py`, frozen result `evaluation/analysis/kriging_variograms.json`; fitted on the normalized training series, per receiver and bin, 666 steps; the raw-dB fit was dominated by level offsets and degenerate (pure nugget at 2400-2600) and is kept in the JSON; no test data used): 600-800 nugget 0.46, sill 5.43, range 126 km; 2400-2600 nugget 0, sill 0.64, range 502 m (empirical gamma 0.44-0.54 and 0.43-0.66 over 729-1358 m). **Only 4 receivers (6 pair distances) for 3 parameters:** at 600-800 the range is unidentified (bootstrap 207 m to 136 km, fit at its bound; gamma is about 0.5 at every distance, so the weights are near-equal everywhere except at the receivers' own cells); at 2400-2600 beyond about 1.5 km the weights are also near-equal. Median weight change between bootstrap draws 0.014 (600-800) and 0.027 (2400-2600), 95th percentile 0.105 and 0.074. Prediction variance (normalized units; total variance 5.9 and 0.64): mean 0.62 at 600-800 (no better than predicting the mean), 0.49 at 2400-2600 (below the total only near receivers).
**Early result (one cell):** 4D lookback mean, 600-800, kriging map, seed 42: map-level MAE 1.093 but **physical-receiver MAE 2.509 dB against 1.215 for IDW power 2**: the near-average kriging map loses receiver-specific levels. The linear-model kriging cells have not finished.
**Code:** `training/ray/spatial15.py` has the new S2 variant `core4-krige` (`load_variogram` injects the band's frozen variogram; existing variant names and the grid contract unchanged; after a final-cluster sync, `--experiments s2 s3 s5` jobs also include `core4-krige`, harmlessly since finished cells are skipped; **the final cluster has deliberately NOT been synced**: workers import `variants()` fresh); `training/ray/spatial15_origins.py` (re-evaluates finished spatial cells from saved weights and dumps per-origin errors for the four core receivers to `spatial15-origin-errors-oct2026/`; spatial15 stores none); `evaluation/analysis/map_construction_vs_per_receiver.py` (written, cannot run yet: paired block bootstrap, block 96, 2,000 draws, receivers pooled, all eight map constructions against per-receiver models, persistence and the lookback mean, plus an unpaired cross-check from `result.json`). Task E's table script needs only the new variant name `core4-krige` (same cell layout under `spatial15-tuned-oct2026/<band>/core4-krige/<model>/seed-N/`).
**UC jobs already running (not stopped; resumable with `--skip-done`; start lines not seen):** `claude-uc-krige-smoke-1` (linear 4D seed 40 still training; its lookback-mean cell is done), `claude-uc-krige-600-lin-1` (linearar4d seeds 41-44), `claude-uc-krige-600-res-1` (residuallinearar4d seeds 40-44), `claude-uc-krige-2400-1` (linearar4d, residuallinearar4d, lookbackmean4d seeds 40-44).
**Commands (not yet run unless stated):** UC kriging cells, one per band: \`python -u -m training.ray.spatial15 --tuned --experiments s2 --variants core4-krige --prefix spatial15-tuned-oct2026 --memory-gb 12 --max-parallel 3 --skip-done --band <band> --models lookbackmean4d linearar4d residuallinearar4d --seeds 40 41 42 43 44\` (lookback mean seed 42 only); per-origin dumps once the kriging cells and the existing S2 cells are done: \`python -m training.ray.spatial15_origins --bands 600_800 2400_2600 --models linearar4d residuallinearar4d lookbackmean4d --skip-done --max-parallel 8\`; analysis: \`python evaluation/analysis/map_construction_vs_per_receiver.py --out evaluation/results/map_construction_vs_per_receiver.csv\`; **STS-PredNet kriging cells (written and registered, not launched):** after syncing the final cluster, \`ray job submit --no-wait --submission-id claude-final-s15sts-krige-1 -- bash training/ray/scripts/sts_krige_followup.sh\` (waits on \`claude-final-s15sts-seeds-1\`, then \`spatial15 --tuned --skip-done --models stsprednet --experiments s2 --variants core4-krige --seeds 40 41 42 --max-parallel 6 --gpu-fraction 0.4 --memory-gb 24 --prefix spatial15-tuned-oct2026 --band <band>\` for both bands); watchdog entry \`{"id": "claude-final-s15sts-krige-1", "entrypoint": "bash training/ray/scripts/sts_krige_followup.sh", "max_restarts": 4}\` (already in \`training/ray/watchdog/final.json\`).
**Caveats:** the kriging maps are close to a plain average because the data show no distance structure (not a tuned kriging); the 600-800 range is unidentified; the other S2 variants have seeds 40-42 only while the base map and kriging have seeds 40-44, so seed-averaged paired comparisons use unequal seed counts. **Open: the paired comparison (the purpose of the task: whether "4D linear models are worse than per-receiver models" survives better maps) has not been run.**


## 38. Status at 2026-11-11 03:37 UTC (all forks finished; user turning the PC off)

### 38.1 What finished since section 37

All five forks are done (offset test for the large models 37.8, 15-minute spatial tables and neural neighbour-ablation code 37.7, June-July and cross-band transfer code 37.6, kriging code 37.9, plus the earlier transfer fork). Earlier this session: paired intervals (34.4), October entropy figure (34.5), per-horizon 15-minute table (34.1), tuned/untuned baselines (34.6), June-July transfer A/B (36.5). **Decision recorded (37.5): spatial analysis is 15-minute only.**

### 38.2 Cluster state at 2026-11-11 03:37 UTC

**Final cluster (6 H100s): 5.7 of 6.0 reserved; running:** Time-LLM 1-minute finals (`claude-final-timellm1min-finals-3`: still 10 + 10 finals on S3, 15 + 15 expected, i.e. none new since the 02:21 relaunch), STS-PredNet variants seed 40 (`-full-600-2`, `-full-2400-2`), the waiting STS seeds 41-42 job (`claude-final-s15sts-seeds-1`), spectral-ablation seeds 43-44 (`claude-final-spectral-seeds-1min-1`, `-15min-1`). Not yet launched on the final cluster: STS-PredNet kriging cells (`training/ray/scripts/sts_krige_followup.sh`; the cluster was deliberately not synced with the kriging code while the STS jobs run).
**UC: 12.0 of 12.0 GPUs reserved; running:** transfer jobs (`claude-uc-tall-rolled-2`, `-llm-1`, `-timellm-1`), TSS-LCD 15-minute tuning (11 trials done of 16), TSS-LCD 2400-2600 at 15 minutes (7 of 10 runs done) and at 1 minute (`claude-uc-tsslcd1-2400-1`, chain), TSS-LCD 600-800 1-minute causal, TimeRAN/MOMENT at 1 minute (`claude-uc-timeran1min-1`: 2 trials done for `timeran_lp`, 0 for `moment_lp`; 2 at a time), full-range rescoring, ResidualConvLSTM tuning (2 trials left, no visible progress), the four kriging jobs (`claude-uc-krige-smoke-1`, `-600-lin-1`, `-600-res-1`, `-2400-1`), STS-PredNet smokes (`claude-uc-offset-sts-smoke-1`, `claude-uc-nbr-smoke-sts-1`; CPU, hours expected). No longer running (finished or ended; not verified which): `claude-uc-tall-direct-cheap-2`, `claude-uc-july-transfer-3`, `claude-uc-offset-sts-smoke-2`. Estimated queue depth 8-12 h (rough).
**Watchdogs:** `claude-watchdog-final-1` and `claude-watchdog-uc-1` running. **Gap:** the four kriging jobs, the two STS smokes and the TimeRAN-related smokes are NOT in `training/ray/watchdog/uc.json`: if one of them crashes it is not restarted (all are resumable with `--skip-done`).
**Leases:** final cluster ends 2026-11-14 01:49 UTC; UC end not recorded.

### 38.3 Open decisions (user)

(1) **New-frequency row** (800-1000 and 5725-5925 MHz): positional transfer with centering (code ready, not run), plus stage C, or a footnote that it is not evaluated. (2) Raise TimeRAN/MOMENT at 1 minute from 2 to 4 parallel (UC memory reservation is the cost). (3) TSS-LCD tuning at 15 minutes beyond the queued 600-800 search (e.g. at 2400-2600); not queued. (4) Whether to run the neighbour ablation for STS-PredNet (GPU, 20 cells) and the large-model offset test (144 jobs, a few GPU-hours) when capacity frees.

### 38.4 What is not started or not run (all code ready unless noted)

The paired map-construction comparison (kriging, IDW powers, grids against the per-receiver models; its per-origin dumps and analysis script exist but have not run; it decides whether "4D linear models are worse than per-receiver models" survives better maps: at IDW power 4/16 section 28.3 suggests the gap closes); the June-July runs for the direct, large and C/Cm models (4 commands in 37.6); the cross-band run; the STS-PredNet and large-model offset full runs (37.8); the STS-PredNet neighbour ablation (37.7); ResidualConvLSTM spatial cells (blocked on its tuning); stage C and transfer for TSS-LCD and STS-PredNet (not planned); the second test week (data not available); the language-model cost timing rerun and the ElectroSense endpoint reruns of the tuned OFA/Time-LLM finals (no code change needed, not queued); the paper's entropy block and the dataset tables' remaining text (sections 34.5 and 36.1).

### 38.5 Repository state

Everything above is committed locally on branch `integrate`; **42 commits are not pushed**; `OVERVIEW.md` and `training/CAMPAIGN_STATUS.md` are untracked. Nothing local runs while the PC is off; every job above runs on the cluster heads.

### 38.6 Re-check at 13:20 UTC on 11-11 (about 10 h after section 38) and what was launched in a hurry

**Finding: both clusters were mostly idle, and part of that was my own mistake.** Final cluster 2.1 of 6 reserved (3 H100s at 0%), UC 8.5 of 12. The STS-PredNet seed-40 jobs, the TSS-LCD jobs, the transfer jobs and the kriging cells had finished. **My waiting job \`claude-final-s15sts-seeds-1\` was still blocked because it waited on \`claude-final-timellm1min-finals-3\`** (long-running); stopped. **Launched at 13:22 (no smoke tests; user leaving):** final cluster, after syncing it to HEAD (kriging code now deployed there; the seed-40 STS jobs had ended): \`claude-final-s15sts-seeds-2-600_800\` and \`-2400_2600\` (STS-PredNet, seeds 40-42, S2/S3/S5, \`--gpu-fraction 0.4 --max-parallel 6\`; 39 runs over 20 variants each, which includes the new \`core4-krige\` variant); UC: \`claude-uc-julyall-rolled-1\` (regimes C/Cm, 1D/2D, June-July), \`claude-uc-julyall-cheap-1\` (cheap direct models), \`claude-uc-julyall-timellm-1\` (Time-LLM, seeds 40-42, 2 at a time), \`claude-uc-nbr-sts-1\` (STS-PredNet neighbour ablation, 10 jobs, 0.5 GPU, 3 at a time), \`claude-uc-origins-s15-1\` (per-origin dumps for the map-construction comparison), plus the earlier \`claude-uc-offset-llm-1\` (large-model offset test). **Failed:** \`claude-uc-julyall-llm-1\` and \`-2\` (OFA, controls, TimeRAN, MOMENT on June-July) hit CUDA out-of-memory errors (MOMENT/TimeRAN need about 6.6 GB each and several processes share the 24 GB cards; Ray's fractional GPU shares do not cap memory), so both were stopped; the large-model June-July run still needs a launch with 1 task per GPU. These jobs are NOT in the watchdog list (only the kriging jobs and the offset job are), so a crash is not restarted. Still running from before: \`claude-uc-timeran1min-1\`, \`claude-uc-tune4d-neural-3\`, \`claude-final-timellm1min-finals-3\`, the two spectral filler jobs.

### 38.7 Check at 13:53 UTC on 11-11 (after the relaunches in 38.6)

**Both clusters are busy.** UC: 11.5 of 12 GPUs reserved, all 12 GPUs at 94-100% utilization (CPU load 7-20 on 48-80 cores, reserved CPU 158 of 464); final cluster: 5.3 of 6 reserved, all six H100s at 96-100% (the Time-LLM final on \`final-1\` uses 53 GB; the STS cells share the other GPUs at 1-2 GB each). **Running on UC:** \`claude-uc-julyall-rolled-1\`, \`-cheap-1\`, \`-timellm-1\`, \`claude-uc-julyall-llm-3\`, \`claude-uc-nbr-sts-1\`, \`claude-uc-origins-s15-1\`, \`claude-uc-timeran1min-1\`, \`claude-uc-tune4d-neural-3\` (the large-model offset test \`claude-uc-offset-llm-1\` is no longer listed: finished or ended, not verified). **Running on the final cluster:** \`claude-final-s15sts-seeds-2-600_800\` and \`-2400_2600\` (STS-PredNet seeds 40-42 including \`core4-krige\`), \`claude-final-timellm1min-finals-3\`, the two spectral seed-43/44 filler jobs. No failed jobs.
**Large-model June-July transfer relaunched with one whole GPU per task (\`claude-uc-julyall-llm-3\`, \`--gpu-fraction 1.0 --max-parallel 3\`):** the CUDA out-of-memory errors are gone (MOMENT cells finish in 40-100 s each: e.g. madsen 07-03 2400-2600 seed 44 A = 0.357 dB, humanities 07-03 1.078, guesthouse 06-28 1.205); 591 jobs left at 13:53. Not yet in the watchdog list (resumable with \`--skip-done\`).
**Not yet checked:** results of the finished jobs (kriging cells, per-origin dumps, TSS-LCD 2400-2600 at 1 and 15 minutes, TSS-LCD 15-minute tuning, the large-model offset test, June-July transfer cheap and rolled sets); the paired map-construction comparison still has to be run once \`claude-uc-origins-s15-1\` finishes.


## 39. Results processed at 2026-11-11 14:09 UTC (everything finished since section 37 that was not yet analyzed or tabulated)

### 39.1 TSS-LCD, all result sets (`evaluation/analysis/tsslcd_table.py` -> `evaluation/results/tsslcd_all.csv`; raw-target MAE in dB, seeds 40-44, no eval smoothing; tuned and untuned heuristics from `baselines_all_horizons.csv`)
**Tuning matters at 15 minutes, unlike at 1 minute; my earlier "use the authors' defaults at 15 minutes" (36.3, from the 1-minute result) was a poor guess.** 15 minutes, 600-800: authors' defaults 1.157 (sd 0.059; causal savgol 1.108) against the **tuned winner 0.898 (sd 0.007; -22%)**; per horizon t+1/4/12/24/48: defaults 1.118 / 1.130 / 1.163 / 1.207 / 1.167, tuned 0.850 / 0.872 / 0.896 / 0.923 / 0.948. Against the heuristics (mean over the same five horizons): tuned median 0.883, tuned mean 0.909, untuned median 1.117, untuned mean 1.162. Over t+4/12/24/48 (the paper convention) the tuned TSS-LCD is 0.910, about 7% better than the tuned median (0.980) and 22% worse than RevIN-MLP (0.743). 15 minutes, 2400-2600 (defaults only; no tuning run): 0.989 (off), 0.975 (causal) against tuned median 0.786 (+26%). 1 minute, 600-800: defaults 1.164 (off), 1.166 (centered), **1.185 (causal: worse)**, tuned 1.162, against tuned median 0.918 (+27%). 1 minute, 2400-2600 (new): 0.852 (off, sd 0.007), 0.836 (causal, sd 0.010) against tuned median 0.734 (+14% to +16%). Reading: TSS-LCD never beats the tuned median except at 15 minutes after tuning (tuned 0.898 vs 0.883 over five horizons: a tie), causal savgol helps at 15 minutes (-4%) and at 1 minute 2400-2600 (-2%) but hurts at 1 minute 600-800. **Open: TSS-LCD tuning at 15 minutes at 2400-2600** (not run).

### 39.2 1-minute per-horizon table with RevIN-MLP and NLinear (`evaluation/analysis/part1_per_horizon_1min.py` -> `evaluation/results/part1_per_horizon_1min.csv`; direct finals at t+1/15/60, **test windows thinned 15x**, seeds up to 5)
600-800 (mean over t+1/15/60): PatchTST 0.917, Time-LLM 0.917, RevIN-MLP 0.918 (0.852 / 0.901 / 1.001), OFA 0.918, **tuned median 0.918 (0.851 / 0.900 / 1.002)**, EWMA and Kalman 0.928, ResidualLinearAR 2D 0.929, ResidualNonlinear 2D 0.931, NLinear 0.932, untuned median (60 min) 0.934, tuned mean 0.934, DLinear 0.941, untuned mean 0.949, persistence 1.159, TSS-LCD 1.162. 2400-2600: **tuned median 0.734 (best)**, EWMA and Kalman 0.745, OFA and Time-LLM 0.749 (+2%), tuned mean 0.749, RevIN-MLP 0.754, PatchTST 0.755, ResidualLinearAR 2D 0.760, NLinear 0.763, DLinear 0.776, TSS-LCD 0.852, persistence 0.932. **At 1 minute the learned models tie the tuned median at 600-800 and trail it by 2-3% at 2400-2600** (consistent with the paired intervals, 34.4); RevIN-MLP matches OFA, PatchTST and Time-LLM at 1 minute as at 15 minutes. Caveat: thinned windows; rescoring at stride 1 is still pending for RevIN-MLP and NLinear.

### 39.3 Offset test for the large models (`claude-uc-offset-llm-1` ended: 95 ok, 49 failed with CUDA out-of-memory)
Gain 1.000 at every offset and at t+12 and t+48 for OFA, OFA random-backbone, OFA from-scratch, PatchTST and RevIN-MLP (6 runs each); TimeRAN and MOMENT 1.000 where they finished (5 and 2-5 of 6 runs); **Time-LLM and part of TimeRAN/MOMENT are missing** (the 0.25 GPU shares overcommitted the 24 GB cards). Relaunched with one whole GPU per task: `claude-uc-offset-llm-2` (`--models timeran_lp moment_lp timellm --gpu-fraction 1.0 --max-parallel 2 --skip-done`). Table: `evaluation/results/offset_response_direct.csv`.

### 39.4 Transfer in regimes A, B, C, Cm (`evaluation/analysis/transfer_all_table.py` -> `evaluation/results/transfer_all_regimes.csv`, raw cache `transfer_all_raw.csv`; MAE at t+12, dB, mean over targets, seeds averaged; groups AERPAW / ARA / COSMOS; baselines lookback mean / persistence)
**External sites are complete for the 11 core models, the cheap direct models in all four regimes, and the large models in regime A; the October guesthouse/law73 group ("powder") was still incomplete in the table** (jobs `claude-uc-tall-direct-cheap-2`, `-rolled-2`, `-llm-1`, `-timellm-1` had ended; relaunched with `--skip-done` as `-cheap-3`, `-rolled-3`, `-llm-2`, `-timellm-2`). **600-800 (AERPAW / ARA / COSMOS; baselines 1.246 / 1.541 / 0.432):** regime **C (normalizer refit) is the best fix for the models that fail frozen**: Autoformer-CSA A 5.97 / 5.07 / 0.96, B 5.98 / 5.08 / 0.96 (centering does nothing), **C 1.51 / 1.87 / 0.52**; LSTMAttn A 20.2 / 13.8 / 23.9, B 2.31 / 2.33 / 1.17, C 1.49 / 1.81 / 0.64; VanillaLSTM 2D A 20.3 / 13.6 / 23.5, B 2.61 / 2.64 / 0.83, C 1.37 / 1.75 / 0.56; TCN 2D A 1.43 / 3.14 / 21.4, B 0.99 / 1.33 / 0.41, C 0.93 / 1.32 / 0.33; **PatchTST without instance normalization A 13.7 / 8.5 / 16.3, B 0.986 / 1.378 / 0.429, C 0.911 / 1.357 / 0.328**; the models with a level pathway change little (ResidualLinearAR 2D A 0.975 / 1.354 / 0.352, C 0.971 / 1.350 / 0.336; LinearAR 1D 1.018 / 1.362 / 0.730 to C 0.907 / 1.294 / 0.321). **Large and direct models in regime A are as good as the best simple models and beat the lookback mean everywhere except COSMOS 2400-2600 ties it:** RevIN-MLP 0.887 / 1.315 / 0.305, Time-LLM 0.895 / 1.301 / 0.311, PatchTST 0.903 / 1.319 / 0.310, OFA 0.906 / 1.323 / 0.308, MOMENT 0.912 / 1.318 / 0.313, TimeRAN 0.918 / 1.347 / 0.324, OFA random 0.914, OFA scratch 0.897. 2400-2600 (baselines 0.946 / 1.452 / 0.087): RevIN-MLP 0.742 / 1.245 / 0.057, Time-LLM 0.750 / 1.270 / 0.057, OFA 0.749 / 1.282 / 0.058, PatchTST 0.763 / 1.295 / 0.059, MOMENT 0.761, TimeRAN 0.774 / 1.379 / 0.064; at COSMOS 2400-2600 the large direct models (0.057-0.064) **beat** the lookback mean (0.087) while the 1D/2D linear models do not (0.107-0.126 after A; 0.063-0.069 after C). So: transfer does not distinguish the language-model models from RevIN-MLP (RevIN-MLP is best or tied everywhere); regime C is the single most effective correction; the failures are level and scale failures of the source normalization.

### 39.5 June-July collection (`evaluation/analysis/transfer_july_all_table.py`, partial: complete groups only)
The A/B table of the 11 core models is in 36.5. The all-regimes table is partial (cheap direct models and C/Cm for linear models are in; the large models and 1D LSTMs are still running: `claude-uc-julyall-*` jobs). Example, 2400-2600, core four / other group, t+12: LinearAR 2D A 0.689 / 1.014, B 0.680 / 1.009, **C 0.612 / 0.935**; DLinear A 0.696 / 1.024, C 0.653 / 1.027; PatchTST A 0.646 / 1.002; MOMENT A 0.713 / 1.141; baselines 1.049 / 1.744.

### 39.6 15-minute spatial (`evaluation/analysis/spatial15_table.py`, now including `core4-krige`)
**Coverage:** linear 4D models 57 of 57 cells in both bands (permuted 25 cells each); lookback mean 19 of 19; **STS-PredNet 21 of 57 per band (seeds 41-42 and the rest of seed 40 are running)**; ResidualConvLSTM 0. **Kriging cells are finished** (5 seeds, both bands, LinearAR 4D and ResidualLinearAR 4D; lookback mean seed 42). **Kriging is much worse at the physical receivers than IDW even though its map-level error is lower:** 600-800 LinearAR 4D map MAE 0.451 (IDW power 2: 0.546) but receiver MAE with the receivers in the input map **2.378** (IDW power 2: 0.906; power 4: 0.794); ResidualLinearAR 4D 0.459 and 2.381 (power 2: 0.904, power 4: 0.785); 2400-2600 LinearAR 4D 0.461 and 1.194, ResidualLinearAR 4D 0.472 and 1.201 (IDW power 4 about 0.64); left-out receivers 5.3 (600-800) and 3.0 (2400-2600), added back 4.1-4.3 and 1.5-1.6. The near-average kriging map (no distance structure in the data, 37.9) loses receiver-specific levels; the map-level score again misleads about receiver accuracy. IDW sensitivity (S2) as in 28.3 (power 4/16 best at the receivers; power 1 and 5x5 grids worst). STS-PredNet in sample is worse than the linear models at every IDW variant (e.g. power 2: 1.203 vs 0.904). The paired comparison against the per-receiver models is still pending (`claude-uc-origins-s15-1`: 90 per-origin dump jobs left at 14:02).

### 39.7 Still running or not analyzed
Per-origin dumps for the map-construction comparison (`claude-uc-origins-s15-1`); STS-PredNet neighbour ablation (`claude-uc-nbr-sts-1`); STS-PredNet variants seeds 40-42 (final cluster); the offset test for Time-LLM, TimeRAN, MOMENT (`claude-uc-offset-llm-2`); June-July large and C/Cm sets; October guesthouse/law73 completion of the all-regimes transfer; TimeRAN/MOMENT at 1 minute (`claude-uc-timeran1min-1`, not tabulated); Time-LLM 1-minute finals (`claude-final-timellm1min-finals-3`; the 1-minute table shows up to 5 seeds); spectral seeds 43-44 (filler jobs, not re-summarized); the STS-PredNet offset smokes. None of the relaunched jobs is in the watchdog list.

### 39.8 Cluster check at 14:11 UTC (after section 39)
UC 11.5 of 12 GPUs reserved; final cluster 5.3 of 6. Running (started in the last 11 hours): UC \`claude-uc-julyall-cheap-1\`, \`-rolled-1\`, \`-timellm-1\`, \`-llm-3\`, \`claude-uc-tall-llm-2\`, \`claude-uc-offset-llm-2\`, \`claude-uc-nbr-sts-1\`, \`claude-uc-origins-s15-1\`; final cluster \`claude-final-s15sts-seeds-2-600_800\` and \`-2400_2600\`. No failed jobs. \`claude-uc-tall-direct-cheap-3\`, \`-rolled-3\` and \`claude-uc-tall-timellm-2\` (the October guesthouse/law73 completion runs) are no longer listed, 15-25 minutes after launch: finished or ended, not verified; rerun \`python evaluation/analysis/transfer_all_table.py --cache evaluation/results/transfer_all_raw.csv\` to see whether the "powder" group is now complete. Disk note: the local machine has 1.1 GB free; the repository's \`.git\` is 3.2 GiB including two leftover temporary pack files (313 MB); no results are downloaded locally (analyses read S3 in memory and write CSVs of at most 4.4 MB).

### 39.9 Done now from the open list (2026-11-11 14:21 UTC)

**October guesthouse/law73 transfer is complete.** `claude-uc-tall-direct-cheap-3`, `-rolled-3` and `claude-uc-tall-timellm-2` ended because they had nothing left (each logged "0 jobs", done: 0 ok, 0 failed). The "powder" column was missing from the all-regimes table only because the table counted the four core receivers' baselines as expected targets: fixed in `evaluation/analysis/transfer_all_table.py` (the October group is guesthouse and law73 only); also note the script **reads the cache file if it exists** (rerun without `--cache`, or delete the cache, to re-read S3). **Guesthouse and law73, 600-800 (MAE mean over t+4/12/24/48, dB; lookback mean 1.546, persistence 1.767):** regime A: RevIN-MLP **0.910**, OFA from scratch 0.916, OFA 0.919, PatchTST without patching 0.925, MOMENT 0.926, PatchTST 0.930, Time-LLM 0.931, ResidualNonlinear 2D 0.940, NLinear 0.968, TimeRAN 0.968, ResidualLinearAR 2D 0.976, DLinear 0.982, LinearAR 1D 1.008, TCN 1.106; failing frozen: LSTMAttn 4.61, VanillaLSTM 2D 4.91, Autoformer-CSA 4.73, PatchTST without instance normalization 2.12, VanillaLSTM 1D 2.28; after regime C: LSTMAttn 1.82, VanillaLSTM 2D 1.65, Autoformer-CSA 1.89 (B: 4.80), PatchTST without instance norm 0.939, VanillaLSTM 1D 1.01, TCN 1.016. **The picture of 39.4 holds on the held-out receivers:** the models with a level pathway transfer at about 0.91-0.98, the rest need regime C, and no language-model model beats RevIN-MLP.
**STS-PredNet offset response (smoke; 600-800, seed 40, 157 origins at stride 4; humanities, madsen, moran; `offset-response-sts-smoke/`, and +10 dB only at stride 40 in `offset-response-sts-smoke2/`):** **the gain is small: 0.03-0.07 at t+4 to t+48 for -20 dB, 0.07-0.15 at -10 dB, up to 0.25 at +10 dB and t+48** (shifts of 0.2-2.5 dB for a 10-20 dB input shift; share within 0.5 dB of the offset about 0-2%): STS-PredNet largely ignores the level of its inputs (a learned-mean forecast, consistent with its worse in-sample error, 1.203 vs 0.904 for the linear 4D models), opposite to RevIN-MLP, OFA and the residual models (gain 1.000). Caveats: one seed, three receivers (sagepoint not run), thinned origins; gain noise (sd 0.1-0.4) is large relative to the values. Full run: `claude-uc-offset-sts` command in 37.8 (6 jobs, GPU).
**STS-PredNet neighbour ablation (smoke; 600-800, seed 40, every 20th origin, `neighbor-ablation-smoke/`; MAE at the target receiver, mean over horizons):** humanities real 1.385, shift96 1.319, shift210 1.532 (+0.15), level 1.470 (+0.09); madsen real 1.178, shift96 1.227, shift210 1.523 (**+0.35**), level 1.200: breaking the neighbours' daily phase costs STS-PredNet 0.15-0.35 dB, **much more than the linear 4D models (+0.02, 33.4)**, so STS-PredNet does use its neighbours (at about 30 origins, one seed, two receivers shown here: preliminary). The full run (`claude-uc-nbr-sts-1`, 10 jobs) is still going.
**TimeRAN and MOMENT at 1 minute are far too slow:** `timeran_lp` 600-800 has 31 of 32 trials after about 11 hours, `moment_lp` and 2400-2600 have none; at 2 tasks at a time the remaining about 97 trials and 80 finals would take days (open decision: raise parallelism, restart `claude-uc-timeran1min-1` with `--max-parallel 8`, losing the in-flight trials).
**Spectral seeds 43-44 are partial** (1 minute 53 and 52 of 130 cells per seed; 15 minutes 27 and 30 of 130; filler jobs on the final cluster): not summarized; the 3-seed result in 33.6 stands.
**Not doable yet:** the paired map-construction comparison (per-origin dumps still running: `claude-uc-origins-s15-1`), the full STS-PredNet neighbour ablation, the STS offset full run, TimeRAN/MOMENT at 1 minute.

### 39.10 Status and newly processed results (2026-11-11 15:17 UTC)

**Cluster status (15:05 UTC):** final cluster 5.1 of 6 GPUs reserved, all six H100s at 98-100%; UC 11.5 of 12 reserved, most GPUs 95-100% (`uc-4` 71%, one V100 50%, **`uc-6` did not answer SSH (timeout during banner exchange)**: watch it). **Time-LLM 1-minute finals are complete: 15 + 15 of 15 per band** (the 1-minute table 39.2 already used up to five seeds; rerun confirms 5 seeds). **TimeRAN at 1 minute: moment_lp dropped (user: "only timeran_lp")**; `claude-uc-timeran1min-1` stopped and replaced by `claude-uc-timeran1min-2` (`--models timeran_lp --max-parallel 4`, watchdog entry replaced): 600-800 trials complete (32 of 32) so its 15 finals started; 30 trials left for 2400-2600; estimate about 8-12 h (the 1-minute final time is not measured). **Estimated drain** (rates since each job started; rough): final cluster about 15 h for the STS-PredNet seeds 40-42 runs (8 of 78 done after 1.7 h); the spectral seed 43-44 filler only runs in gaps (about 50 cells in 12 h: will not drain). UC: June-July rolled C/Cm 535 of 1,404 left (about 1 h), June-July Time-LLM 33 of 78 (about 1.3 h), October OFA/TimeRAN/MOMENT completion 25 of 53 (about 1 h), per-origin dumps 23 left (under 1 h), large-model offset test 38 of 49 (about 4 h), June-July OFA/TimeRAN/MOMENT 558 of 650 (about 10 h), TimeRAN 1 minute (8-12 h), STS neighbour ablation (10 jobs; no counter), ResidualConvLSTM tuning (36.8 h old, no visible progress). UC will have about 4-6 free GPUs in 1-2 h unless more is queued.
**TSS-LCD 15-minute tuning (600-800; 16 of 16 trials ok):** validation objective from 0.756 to 1.690; **authors' defaults (candidate 0) 1.294; winner `tsslcd-43aeb379e8` 0.756 (-42%): latent 32, learning rate 2.2e-4, batch 64, masking 0.2**; the tuned finals scored 0.898 on the test week (39.1). Plausible reason (not tested): the 15-minute training set has about 500 windows, so the default batch of 256 gives only 2 gradient steps per epoch (undertrained); a smaller batch and higher learning rate give more steps.
**Part 1 per-horizon 15-minute table refreshed (`evaluation/results/part1_per_horizon_15min.{csv,txt}`):** the 1D LSTMs at 600-800 now have all horizons (rescoring finished): ResidualVanillaLSTM 1D 0.467 / 0.586 / 0.758 / 0.882 / 0.929 (mean_4h 0.789), VanillaLSTM 1D 0.464 / 0.586 / 0.773 / 0.913 / 0.975 (0.812); at 2400-2600 0.670 and 0.762 (unchanged). Ranking unchanged: the direct models with a level pathway lead (RevIN-MLP 0.743), the 1D recurrent and linear models are 0.79-0.81, and both beat the tuned median (0.980) by 17-24%.
**ElectroSense reproduction table (`evaluation/analysis/es_reproduction_table.py --out evaluation/results/es_reproduction_table.csv`; 54 cells; window-averaged raw MSE, the article convention):** 4 channels at horizons 12 / 48 / 192: OFA (GPT-2) **0.240 / 0.265 / 0.297**, Time-LLM (BERT) 0.246 / 0.274 / 0.516 (sd 0.114 at 192: the documented BERT collapse in some seeds), Time-LLM (GPT-2) 0.250 / 0.285 / 0.344, PatchTST 0.270 / 0.313 / 0.358, DLinear 0.271 / 0.315 / 0.389, LightTS 0.285 / 0.316 / 0.402, Autoformer 0.372 / 0.389 / 0.569, Reformer 0.428 / 0.576 / 0.871, Informer 0.456 / 0.543 / 0.685; 25 channels: OFA 0.581 / 0.813 / 1.036, Time-LLM (GPT-2) 0.581 / 0.829 / 1.030, PatchTST 0.594 / 0.870 / 1.068, DLinear 0.603 / 0.981 / 1.544, LightTS 0.655 / 1.003 / 1.538, Time-LLM (BERT) 0.750 / 1.089 / 1.350, Autoformer 1.217 / 1.253 / 1.625, Reformer 1.233 / 2.111 / 2.474, Informer 3.253 / 2.532 / 2.880. **Caveat: the standard errors of the tuned TSLib baselines are exactly 0.000 (n=5): the five "seeds" are identical because TSLib's `run.py` hardcodes `fix_seed = 2021`; the seed fix (campaign 13.x item (a)) was not completed, so those rows are effectively one run each.** OFA and Time-LLM have real seed spread (sd 0.001-0.017).
**Completed but still not analyzed:** the 60 one-minute 4D/2D/1D checkpoints rescored by `claude-uc-rescore-fullrange-1` (`rescored/oct/`, stride 1 on the final range) are not tabulated; one-minute direct finals (RevIN-MLP, NLinear, the others) are scored on thinned windows and not rescored at stride 1; the 1-minute TimeRAN and moment_lp work is incomplete; the spectral seed 43-44 cells are partial; the October transfer and June-July cheap sets are tabulated (39.4, 39.5; the June-July table is partial until the C/Cm and large-model jobs finish).

### 39.11 One-minute table at stride 1 from the rescored records, the ElectroSense seed fix, the RevIN-MLP/NLinear rescoring (2026-11-11 15:23 UTC)

**Why the rescored checkpoints had not been tabulated:** my omission, not a blocker: they finished in the last hours and no script aggregated `rescored/checkpoints-oct2026*`. **New:** `evaluation/analysis/part1_per_horizon_1min_stride1.py` -> `evaluation/results/part1_per_horizon_1min_stride1.csv` (every test window, 40,084 per run; the 1D LSTMs are thinned to one window in ten, disclosed in the records; 4D rows are map-level and not comparable). **600-800, MAE dB at t+1 / 15 / 60 and the mean:** tuned median **0.851 / 0.900 / 1.002 = 0.918**; PatchTST (direct) 0.846 / 0.905 / 1.007 = 0.919; ResidualVanillaLSTM 1D 0.919; OFA 0.920; ResidualLinearAR 2D 0.924; ResidualLinearAR 1D 0.926; LinearAR 1D 0.928; Kalman 0.928; TCN 2D 0.930; LinearAR 2D 0.930; ResidualLinearAR 2D (direct) 0.932; VanillaLSTM 1D 0.932; ResidualNonlinear 2D 0.933; untuned median 0.934; tuned mean 0.934; DLinear 0.944; untuned mean 0.949; Autoformer-CSA 0.956; ResidualVanillaLSTM 2D 0.999; LSTMAttn 1.089; VanillaLSTM 2D 1.116; persistence 1.159; TSS-LCD 1.162. **2400-2600:** tuned median 0.734, OFA 0.738 (+0.5%), RevIN-MLP 0.742 (+1.1%), PatchTST 0.743, Kalman 0.745, tuned mean 0.749, ResidualNonlinear and ResidualLinearAR 2D 0.750, NLinear 0.752, untuned median 0.759, DLinear 0.763, untuned mean 0.768, TSS-LCD 0.852, persistence 0.932. **4D map-level (600-800; not comparable):** ResidualLinearAR 4D 0.655 / 0.703 / 0.799, LinearAR 4D 0.663 / 0.710 / 0.811, ResidualConvLSTM 0.694 / 0.719 / 0.813, DSwinLSTM 0.916 / 0.908 / 0.944, ConvLSTM 0.867 / 0.905 / 0.978. **Reading (stride 1 confirms the thinned table, 39.2):** at 1 minute every working model is within 1% of the tuned median at 600-800 and within about 1% at 2400-2600; nothing beats it. **Caveats:** the oct-checkpoint rows for the plain 2D LSTMs show 10 "seeds" because rounds 1 and 2 of the same checkpoints are both present (`checkpoints-oct2026` and `-r2`; they are pooled in the mean); 600-800 RevIN-MLP and NLinear and all Time-LLM 1-minute finals are not in this table yet (below).
**Stride-1 rescoring of the 1-minute RevIN-MLP and NLinear finals (user asked; item 2):** `claude-uc-rescore-1min-revin-1` (`rescore --sources direct --resolution 1min --models revinmlp nlinear --skip-done --no-pin --max-parallel 6 --gpu-fraction 0.125`): 60 checkpoints at about 12-16 s each with 6 at a time; 32 of 60 done after 6 minutes (about 5 per minute; about 6 minutes left at 15:50): rerun the stride-1 script when it ends. **Not rescored:** the Time-LLM 1-minute finals (30 checkpoints, 12-15 GB each when rescored, so about one per GPU) are still on thinned windows.
**ElectroSense seed fix: why it was not done and what is happening now:** the problem (TSLib's `run.py` hardcodes `fix_seed = 2021`, so seeds 40-44 are one run and the tuned-baseline standard errors are exactly 0.000) was written down on 2026-11-09 as an open item (13.x (a)) with no code, and it lost out to the weekend's 1-minute and 15-minute campaigns; nobody was blocked on it, and I did not schedule it. Fork task G is writing the fix (seed through an environment variable, rerun the finals with real seeds on new prefixes: 150 tuned and 180 default runs, reusing the validation-selected winners, no new search) and will report the verification, per-run time and the new table.

### 39.12 The Part 1 tables now carry all five metrics; tables run on the cluster (2026-11-11 15:41 UTC)

**Change (the result CSVs were MAE-only, so a paper table row would have mixed new MAE with old or missing MSE):** `evaluation/analysis/part1_common.py` (shared helpers: MAE, MSE, RMSE, MAPE in linear power and in dBm), `part1_per_horizon.py` (15 minutes), `part1_per_horizon_1min.py` (direct finals as trained, windows thinned 15x), `part1_per_horizon_1min_stride1.py` (every test window) and `tsslcd_table.py` (MAE and MSE) all write **long CSVs** (band, family, model, horizon, metric, mean, sd, n) in `evaluation/results/part1_per_horizon_{15min,1min,1min_stride1}.csv` and `tsslcd_all.csv` (text versions `.txt` print MAE and MSE; the heuristics already had all five metrics in `baselines_all_horizons.csv`; TSS-LCD has MAE and MSE only; the old MAE-only CSVs and `tsslcd_new_runs.csv` are superseded). **Run on the cluster, per the standing rule:** `training/ray/scripts/run_tables.sh` (Ray job `claude-uc-tables-2`; the first attempt failed because the tracked baselines CSV is not in the synced code, so the runner now fetches it from `s3://spectrum/analysis-inputs/`) uploads its outputs to `s3://spectrum/analysis-outputs/`; only the 8 small result files (about 280 KB) were fetched to this machine. The run also picked up the **stride-1 rescoring of the 1-minute RevIN-MLP and NLinear finals** (`claude-uc-rescore-1min-revin-1`: 60 of 60 ok, about 15 s each): 600-800 RevIN-MLP 0.849 / 0.906 / 1.006 = **0.920 MAE**, MSE 3.338; NLinear 0.935 MAE, MSE 3.369. Not rescored: the Time-LLM 1-minute finals.
**What MSE adds (this changes the 1-minute reading):** 1 minute, every window, 600-800, mean MSE dB^2 over t+1/15/60: LinearAR 2D 3.311, ResidualLinearAR 2D 3.318, PatchTST 3.329, ResidualNonlinear 2D 3.333, RevIN-MLP 3.338, OFA 3.339, Kalman 3.341, DLinear 3.359, NLinear 3.369, tuned mean 3.392, **tuned median 3.797** (14-15% worse than the best learned model), LSTMAttn 4.085, VanillaLSTM 2D 4.227. The tuned median **wins on MAE (0.918) but is the worst of the good models on MSE**: it gives up squared error for absolute error (as noted in Part 1's caveats). 2400-2600: Kalman 3.081, ResidualNonlinear 3.091, OFA 3.098, PatchTST 3.099, ResidualLinearAR 2D 3.111, tuned median 3.318 (+7.7% over Kalman). So "nothing beats the tuned median at 1 minute" holds for MAE only; on MSE the learned models and Kalman beat it by 7-15%. 15 minutes, 600-800, mean MSE over t+1..t+48: RevIN-MLP **1.744**, OFA 1.777, Time-LLM 1.777, OFA scratch 1.780, PatchTST 1.780, DLinear 1.835, linear 1D 1.878, TSS-LCD tuned 2.686, tuned median 2.826, tuned mean 2.849, untuned mean 3.869, Kalman 4.157, persistence 4.421, TSS-LCD defaults 4.242 (RevIN-MLP is 38% below the tuned median in MSE; Kalman and persistence degrade with the horizon: 9.1 and 9.5 at t+48). **The paper's tables should state which metric each claim is about.**

### 39.13 Discussion: what the 1-minute MAE and MSE results say together, and a correction to 39.12 (2026-11-11 15:44 UTC)

**Correction to 39.12.** It said that on MSE "the learned models and Kalman beat [the tuned median] by 7-15%" and that "nothing beats the tuned median at 1 minute holds for MAE only". The first statement is true but compares against the weakest MSE baseline. **The relevant simple baseline for MSE is the lookback mean, and against it the learned models gain almost nothing.** Every number below is from `evaluation/results/part1_per_horizon_1min_stride1.csv` (every test window, mean over t+1/15/60 minutes, seeds averaged, no intervals).
| MSE (dB^2) | 600-800 | 2400-2600 |
|---|---:|---:|
| best learned model | 3.311 (LinearAR 2D) | 3.091 (ResidualNonlinear 2D) |
| RevIN-MLP / PatchTST / OFA | 3.338 / 3.329 / 3.339 | 3.115 / 3.099 / 3.098 |
| Kalman | 3.341 | 3.081 |
| tuned lookback mean | 3.392 | 3.114 |
| untuned lookback mean (60 min) | 3.420 | 3.233 |
| tuned median | 3.797 | 3.318 |
**Facts.** (1) 600-800: the best learned model is 2.4% below the tuned mean, PatchTST, RevIN-MLP and OFA 1.6-1.9% below; at t+60 the margin is 0.5-1% (3.774-3.803 against 3.828). (2) 2400-2600: the tuned mean is within 0.7% of the best learned model; RevIN-MLP (3.115) and DLinear (3.132) are no better than it (3.114), and at t+60 the tuned mean (3.824) is better than RevIN-MLP (3.867), OFA (3.876), PatchTST (3.853), ResidualLinearAR 2D (3.842), NLinear (3.886) and DLinear (3.881). (3) Kalman is the best or second-best model on MSE at 2400-2600 (3.081) and ties the learned models at 600-800 (3.341). (4) The tuned median is the best model on MAE (0.918 and 0.734) and the worst of the good models on MSE (3.797 and 3.318). (5) On MAE the learned models are within about 1% of the tuned median at 600-800 and 0.5-1.3% above it at 2400-2600 (stride-1 table, 39.11; paired intervals in 34.4 on the thinned origins).
**Reading.** The tuned median minimizes absolute error and the mean minimizes squared error, so a table in one metric flatters the heuristic built for it; which heuristic is "the strongest" depends on the metric. Taking the best simple heuristic per metric (median for MAE, mean or Kalman for MSE), **no learned model beats it by more than about 2.4% in any 1-minute cell**; the learned models are inside what a tuned heuristic achieves. This agrees with the entropy analysis (31.8, 34.5: at 1 minute the learned models' gap to the lookback mean is at most 0.03-0.04 dB) and with the paired intervals (34.4). At 15 minutes the picture is different in both metrics (RevIN-MLP 0.743 vs 0.980 MAE and 1.744 vs 2.826 MSE against the tuned median; the tuned mean is 0.909 over five horizons). **Hypotheses, not tested:** the 1-minute series are close to a level plus noise at short horizons, so a good estimate of the local level is nearly all that can be predicted; the learned models learn that level estimate and the heuristics are tuned to it.
**What would make the statement firm (not done):** paired block-bootstrap intervals for MSE (the per-origin dumps of 34.4 hold MAE only; squared errors need a rerun of `training/ray/origin_errors.py` with an MSE output), and for the tuned mean and Kalman as comparators (34.4 compared the median, the 60-minute mean and Kalman on MAE). Until then the 1-minute claim should read "within about 1-2% of the best tuned heuristic in each metric", not "ties the median". The Time-LLM 1-minute finals are not in the stride-1 table (still thinned), and TimeRAN at 1 minute is incomplete.

### 39.14 Task G: the ElectroSense TSLib seed fix (fork; commits `f95b114` fix, `ee2e8f3` table option, `6c8d6a6` setup probe, watchdog `0105eaa`, `28e2d42`, `7ee407b`; local, synced to UC; `tests/test_tsl_seed.py`, 4 passing) (2026-11-11 15:49 UTC)

**Cause confirmed:** the pinned library's `run.py` hardcodes `fix_seed = 2021`; its `--seed` flag only reaches augmentation helpers that are unused at ratio 0, so seeds 40-44 of the tuned baselines were five identical runs (hence the 0.000 standard errors in 39.10). **Fix:** `patch_seed()` in `training/ray/tsl_runs.py` changes only that line to read `TSL_SEED` (default 2021, so old results stay reproducible); `setup_node` applies it per node, `tsl_job` sets `TSL_SEED`; `tune_tsl` gained `--winners-from` and `--seeds` (finals only, no new search); `tsl_runs run` gained `--max-parallel`, `--skip-done`, `--cpu`. New prefixes `electrosense-results/seedfix-tuned` and `seedfix-default` (old ones untouched). **Verification (DLinear, spec4, horizon 12, tuned winner, CPU):** `TSL_SEED=2021` gives MSE 0.27111053 (identical to the old number); seed 40 gives 0.27307, seed 41 0.27083. **Preliminary real-seed values (DLinear tuned, MSE, mean ± standard error over 5 seeds; old value):** spec4 h12 0.2723 ± 0.0005 (0.2711), h48 0.3170 ± 0.0018 (0.3146), h192 0.3895 ± 0.0010 (0.3892); spec25 h48 0.9675 ± 0.0234 (0.9813), h192 1.5087 ± 0.0299 (1.5435): at 25 channels the seed spread is large, so the old single-run numbers hid real variance. **Problems fixed on the way:** the setup check imported only the experiment class so PatchTST, Informer and Reformer failed on a missing `reformer_pytorch` (the check now imports every model; setup rerun on all nine UC nodes); PatchTST and Informer were far too slow on CPU (almost no completions in 25 minutes).
**Jobs (all in the watchdog):** `claude-uc-tsl-seedfix-tuned-cpu-3` and `-default-cpu-3` (DLinear, LightTS, Autoformer, CPU); `-tuned-gpu-3` and `-default-gpu-3` (PatchTST, Informer, and Reformer for the defaults, 0.25 GPU each). **Progress at the report:** 45 of 150 tuned runs and 18 of 180 default runs done (old failed records remain until overwritten; the table and `--skip-done` ignore records that are not ok); 22 tasks running, 10 pending (UC 12 of 12 GPUs reserved). **Run times:** on GPU (old records) DLinear, LightTS, PatchTST 20-35 s, Informer 76-143 s, Autoformer 190-220 s (the full set about 4-5 GPU-slot-hours); on CPU DLinear 20-26 s, Autoformer about 295 s, PatchTST 210-250 s at h12. Estimate 3-8 h (rough; about 0.6 runs per minute so far). **Table when the runs finish:** `python evaluation/analysis/es_reproduction_table.py --seedfix --out evaluation/results/es_reproduction_table_seedfix.csv`. **Caveats:** the old table's Reformer rows are untuned with seeds 2021-2023, so they differ from the new 6-model defaults set; small shifts from the old numbers are expected (the old five seeds were all the 2021 run).

### 39.15 The lookback median under TSS-LCD's own evaluation protocol (user task; `evaluation/analysis/tsslcd_protocol_baseline.py`, Ray jobs `claude-uc-tsslcd-protocol-1`/`-2` and `claude-final-tsslcd-protocol-1`; results `evaluation/results/tsslcd_protocol_baseline.csv` and `..._fullweek.csv`, columns band, savgol, inputs, eval_smooth, target, horizon, metric, value; commit `2e10606` local) (2026-11-11 16:22 UTC)

**Method (inside the TSS-LCD harness, nothing copied):** `build_partition` (profile 1min, scaler fit on the training rows), `window_starts(..., CONTEXT + FUTURE, 1)`, `score_grid` with eval smoothing off and 6:3 centered, against the raw target and the model target; prediction = the median of the last k_h input steps (k_h from `median-study/selected_k.json`: 600-800 {1: 20, 15: 30, 60: 30}, 2400-2600 {20, 20, 20}) per column, inverted with `scaler.inverse`; (i) inputs = `test_model` (the series TSS-LCD sees; with savgol centered it contains samples after the origin, the authors' leak) and (ii) inputs = `test_raw`; deterministic; CPU; run on the cluster.
**Window sets (a finding):** the stored 600-800 TSS-LCD runs with savgol off and centered (`tsslcd-authors-full/`, the 0.739 and 1.164 rows of 22.2) were scored on **9,888 windows**, whereas the current data give **9,962**: **those runs saw a test week that ended at 2026-11-03 18:46 UTC, 74 minutes early (the test files had been synced only that far)**; setting the range end to 18:46 reproduces exactly 9,888 windows (checked with the harness: 9,888 vs 9,962 windows, 10,007 vs 10,081 rows). The tuned TSS-LCD finals, the causal runs and the whole 2400-2600 set were scored on the full week (9,962). **So TSS-LCD results are on two slightly different window sets (74 of 9,962 windows, 0.7%)**: disclose or rerun the early runs. For like-for-like comparison the 600-800 median is scored on the 9,888-window set in the main CSV (the full-week version is in the second CSV); 2400-2600 uses 9,962 (its stored TSS-LCD runs do).
**Sanity checks:** (1) median, savgol off, no averaging, raw target, raw inputs, MAE at t+1 / 15 / 60: **600-800 0.849 / 0.899 / 1.002** (Table 3: 0.851 / 0.900 / 1.002; harness 9,888 windows) and **2400-2600 0.663 / 0.715 / 0.824** (Table 3: 0.663 / 0.716 / 0.824; 9,962 windows): within 0.002 dB (the harness window set differs from the pipeline's). (2) TSS-LCD from the stored runs under the same keys: 600-800 centered, 6:3 centered, model target **0.739**, raw target **1.156**; no averaging 0.755 / 1.166; savgol off 1.164 (and 0.916 / 1.152 with 6:3): reproduces 22.2 exactly.
**Results, MAE averaged over t+1/15/60 minutes (target: raw measurement / the model's smoothed target; eval averaging none / 6:3 centered):**
| 600-800, savgol centered (the authors' protocol) | raw, none | raw, 6:3 | smoothed, none | smoothed, 6:3 |
|---|---:|---:|---:|---:|
| TSS-LCD (authors' defaults, 5 seeds) | 1.166 | 1.156 | 0.755 | **0.739** |
| median (i) model inputs (same inputs as TSS-LCD, with look-ahead) | 0.927 | 0.928 | 0.398 | **0.399** |
| median (ii) raw inputs | 0.917 | 0.910 | 0.487 | 0.482 |
| 600-800, savgol off | raw, none | raw, 6:3 | model target, none | model target, 6:3 |
| TSS-LCD | 1.164 | 1.152 | 1.164 | 0.916 |
| median (i) = (ii) | 0.917 | 0.910 | 0.917 | 0.677 |
| 2400-2600, savgol off | raw, none | raw, 6:3 | | |
| TSS-LCD (no centered 6:3 row exists) | 0.852 | not run | | |
| median (i) = (ii) | 0.734 | 0.728 | | |
| 2400-2600, savgol centered (median only; TSS-LCD was never run with centered savgol at 2400-2600) | raw, none | raw, 6:3 | smoothed, none | smoothed, 6:3 |
| median (i) | 0.742 | 0.743 | 0.341 | 0.342 |
| median (ii) | 0.734 | 0.728 | 0.401 | 0.397 |
Per horizon at 600-800, centered, 6:3, smoothed target (t+1 / 15 / 60): median (i) 0.216 / 0.410 / 0.571; median (ii) 0.343 / 0.480 / 0.624; TSS-LCD 0.718 / 0.744 / 0.755.
**Reading:** under TSS-LCD's own protocol the lookback median scores **0.399 (with the same, look-ahead inputs) and 0.482 (raw inputs) against TSS-LCD's 0.739, i.e. 46% and 35% lower**; in every cell of the grid the median is better than TSS-LCD (raw target 0.91-0.93 vs 1.15-1.17; savgol off, 6:3 on the model target 0.677 vs 0.916). The headline 0.739 therefore does not show forecasting skill: a median over the last 30 minutes scores far better on the same protocol, and 0.09 dB of the median's smoothed-target score comes from the filter's look-ahead (0.398 with look-ahead inputs vs 0.487 without). The protocol also rewards smoothing as such (the smoothed target is easier: 0.40-0.49 for the median vs 0.92 against the raw target). **Caveats:** the median is tuned on validation (k per horizon) and TSS-LCD uses the authors' defaults (tuning at 1 minute changes it by 0.2%, 39.1); TSS-LCD at 2400-2600 has no centered or 6:3 centered rows (only savgol off and causal were run: centered at 2400-2600 would be 5 runs of about 3.7 h each, not launched); the two window sets above.

### 39.16 Paired map-construction comparison (task F's script, run on the cluster as `claude-uc-mapcmp-1`; `evaluation/results/map_construction_vs_per_receiver.{csv,txt}`; per-origin dumps `claude-uc-origins-s15-1`: 128 of 128 ok) (2026-11-11 16:23 UTC)

Paired differences on identical origins (4D minus other, dB, mean over t+4/12/24/48, pooled over the four core receivers, moving-block bootstrap block 96, 2,000 draws, 95% interval; negative = the 4D model is better; LinearAR 4D shown, ResidualLinearAR 4D within 0.01):
| band, map | 4D MAE | vs LinearAR 1D | vs LinearAR 2D | vs ResidualLinearAR 2D | vs 4D lookback mean (same map) | vs persistence |
|---|---:|---|---|---|---|---|
| 600-800, IDW power 2 (default) | 0.908 | +0.127 [+0.113, +0.143] | +0.105 [+0.092, +0.116] | +0.116 [+0.102, +0.127] | -0.307 | -0.357 |
| 600-800, power 4 | 0.794 | **+0.013 [+0.004, +0.027]** | **-0.009 [-0.016, -0.001]** | +0.002 [-0.009, +0.015] | -0.346 | -0.471 |
| 600-800, power 16 | 0.796 | +0.015 [+0.005, +0.029] | -0.008 [-0.015, +0.001] | +0.004 [-0.009, +0.016] | -0.345 | -0.469 |
| 600-800, grid 20x20 / linear power / grid 5x5 / power 1 | 0.866 / 1.055 / 1.372 / 2.033 | +0.085 / +0.274 / +0.591 / +1.252 | | | | |
| 600-800, **kriging** | **2.378** | **+1.596 [+1.521, +1.661]** | +1.574 | +1.585 | -0.131 | **+1.112 (worse than persistence)** |
| 2400-2600, IDW power 2 (default) | 0.937 | +0.284 [+0.221, +0.326] | +0.254 [+0.183, +0.314] | +0.261 [+0.184, +0.311] | -0.279 | -0.194 |
| 2400-2600, power 4 | 0.634 | -0.018 [-0.036, +0.018] | **-0.048 [-0.064, -0.011]** | **-0.041 [-0.058, -0.019]** | -0.317 | -0.496 |
| 2400-2600, power 16 | 0.636 | -0.017 [-0.035, +0.022] | -0.046 [-0.062, -0.006] | -0.039 [-0.056, -0.015] | -0.314 | -0.494 |
| 2400-2600, kriging | 1.194 | +0.541 [+0.456, +0.605] | +0.511 | +0.518 | -0.244 | +0.063 [-0.047, +0.163] |
**Reading: the earlier finding "the 4D linear models are 0.10-0.29 dB worse than per-receiver models" (33.4) holds only for the default IDW power-2 map.** With a sharper map (IDW power 4 or 16, close to nearest-neighbour) **the 4D model ties the per-receiver models at 600-800 (within +/- 0.015 dB, sign depends on the comparator) and beats the 2D per-receiver models by 0.04-0.05 dB (about 7%) at 2400-2600**, and it beats the 4D lookback mean by 0.31-0.36 dB and persistence by 0.47-0.50 dB under the same map. The map construction therefore moves the 4D result by more than the 4D-versus-per-receiver gap: smoothing maps (power 1, 5x5 grids, linear-power interpolation) are far worse (+0.25 to +1.75 dB), the 20x20 grid is in between, and **kriging is the worst construction tried (2.38 dB at 600-800, worse than persistence; 1.19 at 2400-2600)**. Why: at the receivers the sharper map reproduces the receiver's own value, so the 4D model can effectively read its own history through the map; smoother maps blend other receivers' levels into the cell (and kriging, fitted on four receivers with no distance structure, is close to a plain average, 37.9). It does NOT show that neighbours help: the neighbour ablation (33.4) found at most +0.024 dB when neighbours are broken, and a 4D model whose map at the receiver is its own value is a per-receiver model in a different guise. **For Part 4: the paper's spatial claim should state the map construction (and report power 4 alongside the default).** Caveats: seeds 40-44 for the base map and kriging, 40-42 for the other variants (unequal seed counts); intervals cover origin-sampling variation only; per-receiver 1D/2D references are the October models scored in distribution; the unpaired cross-check from the cells' result.json agrees (e.g. 2400-2600 power 4: 0.635 vs lookback mean 0.952).

### 39.17 ResidualConvLSTM 15-minute 4D tuning: the search is finished, the finals are running (2026-11-11 16:33 UTC)

The "[2 left] with no visible progress" noted since 31 (and in 35-39) was not a stall: **the trials are extremely slow, 13 to 28 hours each** (46,500-100,300 s; the last two, finished after 95,690 s and 100,310 s, were 26-28 h). **All 16 trials are done per band** (`tuning/powder15_4d/<band>/4d/residualconvlstm/`): best validation (map-level) objectives 600-800 **0.8456** (next 1.2245, 1.2392; worst 1.369) and 2400-2600 **0.7361** (next 0.7695, 0.8054; worst 1.062). The job `claude-uc-tune4d-neural-3` (37 h old) printed "[tune] 10 final seed runs" after its last trial (about 12:00 UTC 11-11): **5 seeds x 2 bands are training now**; if a final costs about as much as a trial the finals end between about 01:00 and 16:00 on 11-12 (guess, no final time seen yet). **Consequence for the spatial work:** the retrained spatial variants (S2/S3/S5: 19 variants per band, each a new training run) cannot be afforded for ResidualConvLSTM (hundreds of GPU-hours at 13-28 h per run); what is affordable is scoring the finished base models at the physical receivers and in the conditions that need no retraining (left-out and added receivers, permuted coordinates), as was done for the STS-PredNet base cell. Not decided; proposal only.

### 39.18 The protocol baseline extended to the causal filter and trailing 6:3 averaging (2026-11-11 16:38 UTC; Ray job `claude-uc-tsslcd-protocol-3`; `evaluation/results/tsslcd_protocol_baseline.csv` now 768 rows, `..._fullweek.csv`)

Added: savgol `causal` to the median grid and the eval setting 6:3 trailing (`p6g3cau`) for every savgol setting, both bands, inputs (i) and (ii); TSS-LCD rows for the causal-filter runs (both bands, 5 seeds each; the stored runs already held `off` and `p6g3cau` keys, so no rescoring from weights was needed) and the `p6g3cau` column for the off and centered runs. **The 384 earlier rows are unchanged (max difference 0.0).** Window sets: 600-800 savgol off and centered stay on the matched 9,888 windows (the stored TSS-LCD runs); the causal TSS-LCD runs and all of 2400-2600 were scored on the full week (9,962), so the 600-800 causal median uses 9,962 (the full-week CSV has all of them on 9,962).
**Causal filter + trailing 6:3 averaging, MAE averaged over t+1/15/60 (per horizon t+1 / 15 / 60 in parentheses):**
| band | target | TSS-LCD | median (i) model inputs | median (ii) raw inputs |
|---|---|---|---|---|
| 600-800 | smoothed (model) | 0.870 (0.845 / 0.888 / 0.877) | 0.618 (0.495 / 0.639 / 0.719) | **0.600** (0.439 / 0.637 / 0.725) |
| 600-800 | raw | 1.172 (1.155 / 1.180 / 1.181) | 0.969 (0.919 / 0.946 / 1.043) | **0.924** (0.860 / 0.904 / 1.006) |
| 2400-2600 | smoothed (model) | 0.591 (0.525 / 0.613 / 0.637) | 0.544 (0.413 / 0.576 / 0.644) | **0.499** (0.351 / 0.525 / 0.621) |
| 2400-2600 | raw | 0.826 (0.795 / 0.831 / 0.852) | 0.789 (0.730 / 0.767 / 0.869) | **0.739** (0.672 / 0.719 / 0.827) |
Other cells (causal filter, no averaging; raw / model target): 600-800 TSS-LCD 1.185 / 0.896, median (i) 0.965 / 0.616, (ii) 0.917 / 0.600; 2400-2600 TSS-LCD 0.836 / 0.613, median (i) 0.785 / 0.541, (ii) 0.734 / 0.497. The new `p6g3cau` column for the other filters: 600-800 centered (model / raw target) TSS-LCD 0.740 / 1.157, median (i) 0.408 / 0.935, (ii) 0.489 / 0.923; savgol off TSS-LCD 0.917 / 1.153, median 0.684 / 0.923; 2400-2600 savgol off TSS-LCD 0.668 / 0.834, median 0.556 / 0.739; 2400-2600 centered (median only) 0.351 / 0.749 (i), 0.404 / 0.739 (ii).
**Reading:** with a causal filter and causal averaging (no look-ahead anywhere) the lookback median still beats TSS-LCD in every cell: **600-800 by 31% (smoothed target) and 21% (raw target)** (0.600 vs 0.870; 0.924 vs 1.172), **2400-2600 by 16% and 11%** (0.499 vs 0.591; 0.739 vs 0.826). The gap is smaller at 2400-2600 and at the 60-minute horizon: there the median with the same filtered inputs, (i), ties or slightly loses to TSS-LCD at t+60 on the smoothed target (0.644 vs 0.637), while the median on raw inputs, (ii), is still ahead (0.621). The causal protocol removes the filter's look-ahead from TSS-LCD's inputs and targets, and TSS-LCD's score changes little (600-800 0.739 centered vs 0.870 causal, smoothed target) while the median's smoothed-target score rises from 0.399 to 0.618 (i): the centered filter's look-ahead was worth about 0.22 dB to the median and 0.13 dB to TSS-LCD.

### 39.19 Jobs that ended (checked 2026-11-11 17:27 UTC), the seed-fixed ElectroSense table, zero-shot MOMENT status

**Ended jobs (verified from their last log lines and S3):** `claude-uc-julyall-rolled-1` finished **1404 of 1404 ok, 0 failed**; `claude-uc-julyall-cheap-1` **1300 ok, 0 failed**; the ElectroSense seed-fixed tuned jobs (`-tuned-cpu-3`, `-tuned-gpu-3`) finished with **150 of 150 tuned records ok**; the default jobs wrote **175 of 180 ok, 5 failed with "no metrics in run.py output"** (PatchTST spec25 h12 seed 44, h192 seed 44, h48 seeds 41 and 43; Reformer spec4 h192 seed 43: process died, probably out of memory on the shared GPU shares), **rerun as `claude-uc-tsl-seedfix-default-gpu-4` (0.5 GPU, 3 at a time): 5 of 5 ok, so the seed-fixed set is complete (330 of 330)**; `claude-uc-mapcmp-1` and `claude-uc-tsslcd-protocol-3` finished (39.16, 39.18). The October guesthouse/law73 completion jobs finished earlier with nothing left (39.9). `claude-final-momentzs-choose-1` (task H) failed at launch and was replaced by `claude-final-momentzs-validate-2`.
**Seed-fixed ElectroSense table (`evaluation/analysis/es_reproduction_table.py --seedfix`, run on the cluster by `training/ray/scripts/run_es_table.sh` as `claude-uc-estable-2`; `evaluation/results/es_reproduction_table_seedfix.{csv,txt}`, 84 cells; window-averaged raw MSE, mean ± standard error of 5 real seeds; the first run of the runner failed because the table script builds its own boto3 client from the generic S3_* variables, which point elsewhere on the workers: fixed by mapping the SPECTRUM_* ones):** 4 channels, horizons 12 / 48 / 192: **OFA (GPT-2) 0.240 / 0.265 / 0.297** (unchanged: it always had real seeds), Time-LLM (BERT) 0.246 / 0.274 / 0.516 (sd large at 192: BERT collapse in some seeds), Time-LLM (GPT-2) 0.250 / 0.285 / 0.344, **PatchTST defaults 0.262 / 0.297 / 0.379, tuned 0.274 / 0.312 / 0.354**, DLinear tuned 0.272 / 0.317 / 0.389 (defaults 0.358 / 0.398 / 0.479), LightTS tuned 0.278 / 0.331 / 0.402 (defaults 0.359 / 0.429 / 0.558), Autoformer tuned 0.397 / 0.513 / 0.544 (defaults 0.575 / 0.681 / 0.714), Informer tuned 0.533 / 0.552 / 0.634 (defaults 0.512 / 0.625 / 0.840), Reformer defaults 0.424 / 0.613 / 0.812. 25 channels: OFA 0.581 / 0.813 / 1.036, Time-LLM (GPT-2) 0.581 / 0.829 / 1.030, PatchTST defaults 0.585 / 0.858 / 1.052 (tuned 0.592 / 0.881 / 1.078), DLinear tuned 0.606 / 0.967 / 1.509, LightTS tuned 0.656 / 1.119 / 1.623, Autoformer tuned 1.140 / 1.243 / 1.684, Reformer 1.180 / 2.057 / 2.515, Informer 2.856 / 2.582 / 2.567 (defaults 3.321 / 2.964 / 2.806). Standard errors are now real (0.000-0.067 over seeds; e.g. Autoformer tuned 0.397 ± 0.029; DLinear tuned 0.272 ± 0.000 means below 0.0005). **Findings for the paper:** (1) the old table's 0.000 standard errors were an artifact; the point estimates moved little (DLinear tuned spec4 h12 0.272 vs 0.271). (2) **Tuning the library baselines on the data helps a lot for DLinear (-24% at 4 channels, horizon 12), LightTS (-23%) and Autoformer (-31%), but the validation-selected PatchTST is no better than the library defaults (4 channels: +4.6% worse at horizon 12, +5% at 48, better only at 192; 25 channels +1% to +3% worse) and Informer is mixed**: tuning on a validation split does not always help. (3) OFA leads at 4 channels by 8% at horizon 12 over the best PatchTST (defaults) and ties it within 1.5% at 25 channels (0.581 vs 0.585 at horizon 12; 1.036 vs 1.052 at 192). Supports the tuning paragraph (39.x) with a second dataset.
**Zero-shot MOMENT (task H, interim):** the forecaster (forecasting as masked reconstruction) is built; the design **`chunk1` was chosen on the validation range and frozen (commit `cc13bc8`) before any test scoring**; the test-week run is going on the final cluster; no results yet.

### 39.20 Cluster status at 2026-11-11 17:57 UTC
UC 11.5 of 12 GPUs reserved (50 of 464 CPUs), GPUs at 63-100%: `claude-uc-nbr-sts-1` (4.6 h), `claude-uc-julyall-timellm-1` (4.6 h), `claude-uc-julyall-llm-3` (4.5 h), `claude-uc-offset-llm-2` (4.0 h), `claude-uc-tall-llm-2` (3.9 h), `claude-uc-timeran1min-2` (2.8 h), `claude-uc-tsslcd15-tune-2400-1` (1.3 h; TSS-LCD tuning at 15 minutes, 2400-2600). Final cluster 5.55 of 6 reserved (60 of 288 CPUs), GPUs at 99-100% except `final-6` (43%): STS-PredNet seeds 40-42 (`claude-final-s15sts-seeds-2-600_800` and `-2400_2600`, 4.6 h), Time-LLM 1-minute stride-1 rescoring (1.9 h), the spectral seed 43-44 filler jobs (15.2 h); `claude-final-momentzs-choose-1` (task H) still shows FAILED (replaced by `-validate-2`; the test-week run is in progress). No other failures.

### 39.21 Reformer tuning for the ElectroSense reproduction (queued 2026-11-11 18:07 UTC)
**Why Reformer was untuned:** no reason is recorded; the library-baseline tuning covered DLinear, PatchTST, Informer, Autoformer and LightTS, and 13.x lists Reformer and TimesNet as "other library models, not tuned" (TimesNet was also dropped from the defaults as too slow). **Queued:** `claude-uc-tsl-reformer-tune-1` (`tune_tsl --models Reformer --datasets spec4 spec25 --horizons 12 48 192 --output-prefix electrosense-results/seedfix-tuned-reformer --gpu-fraction 0.5 --max-parallel 4`; 64 trials to run = 2 datasets x 16 candidates x 2 tuning horizons (12 and 192), then 5 finals x 3 horizons x 2 datasets with real seeds; the library defaults are candidate 0; search seed 42, which is now a real seed through the TSL_SEED patch). New: `Reformer` in `DEFAULTS` and `SPACES` of `training/ray/tune_tsl.py` (learning rate, d_model, heads, layers, d_ff, dropout, batch size, epochs; the LSH bucket size and hash count are fixed in the library's constructor), a separate output prefix so the other models' `winners.json` is not touched, and `es_reproduction_table.py --seedfix` now also reads `seedfix-tuned-reformer`. Estimate about 2 GPU-hours (a default Reformer run takes about 70 s on GPU); in the watchdog; the tuned row appears in the seed-fixed table when it finishes (rerun `training/ray/scripts/run_es_table.sh`).

### 39.22 Zero-shot MOMENT (task H; built by the fork, tabulated by me because the fork ended without a final report) (2026-11-11 18:38 UTC)

**Method (`training/ray/moment_zeroshot.py`, `evaluation/analysis/moment_zeroshot_table.py`, commits `d05dbac` ... `cc13bc8`, `037fef1`):** the frozen pretrained MOMENT-1-small (no head, no training) forecasts by masked reconstruction: the 96-step context is placed at the start of the 512-step input and the future patches are masked and reconstructed. **Design chosen on the validation range only (2026-10-25 14:00 to 10-27 13:55 UTC; four core receivers, both bands, mean MAE over horizons 1/4/12/24/48) and frozen in `evaluation/analysis/moment_zeroshot_design.json` before the test week was scored (commit `cc13bc8`): `chunk1` (autoregressive, one masked patch per pass; validation MAE 1.155; candidates single6 1.181, chunk3 1.188, per_horizon 1.189; a last-value forecast scores 1.054 on the same validation range, i.e. all four designs are worse than last value on validation too).** Jobs: `claude-final-momentzs-setup-1`, `-choose-2`, `-choose-3`, `-validate-2`, `-test-1` all SUCCEEDED (`-choose-1`, `-validate-1` failed or stopped and were replaced); table `claude-uc-momenttable-1`; `evaluation/results/moment_zeroshot_15min.csv`, `moment_zeroshot_table.txt` (all five metrics, long format).
**Test week, 15-minute POWDER, four core receivers pooled (MAE dB at t+1 / 4 / 12 / 24 / 48 and mean over t+4/12/24/48):** 600-800 **zero-shot MOMENT 1.164 / 1.556 / 1.472 / 1.310 / 1.285, mean 1.406** against persistence 1.245, tuned median 0.981, RevIN-MLP 0.744, MOMENT linear probe 0.766, TimeRAN linear probe 0.790; MSE mean 5.792 (persistence 5.265, tuned median 3.286, RevIN-MLP 1.972). 2400-2600: **1.196** (0.971 / 1.246 / 1.197 / 1.193 / 1.148) against persistence 1.117, tuned median 0.870, RevIN-MLP 0.617; MSE 6.386. **Paired block bootstrap against the tuned median on identical origins (2,500 origins): +0.418 dB [+0.385, +0.445] at 600-800 and +0.319 dB [+0.288, +0.356] at 2400-2600**, i.e. the zero-shot model is 43% and 37% worse than the tuned median and also worse than persistence. It is worst at short horizons (t+1: 1.164 vs persistence 0.505 at 600-800) and relatively closer at t+48 (1.285 vs persistence 1.864, tuned median 1.155). Held-out receivers and external sites (MAE at t+12 / mean over t+4..48): POWDER guesthouse+law73 600-800 1.926 / 1.912, 2400-2600 1.916 / 1.915; AERPAW 1.620 / 1.578 and 1.396 / 1.291; ARA 1.962 / 1.940 and 2.157 / 1.924; COSMOS 0.541 / 0.544 and 0.125 / 0.116 (compare the trained models at the same targets in 39.4: e.g. RevIN-MLP AERPAW 600-800 0.887, MOMENT linear probe 0.912).
**Reading:** **a pretrained time-series foundation model used zero-shot, without even a linear head, does not forecast this spectrum data usefully**: it is worse than persistence at 15-minute resolution (except at the longest horizons) and 37-43% worse than the tuned median; the 400,000-series pretraining does not transfer by itself, and the linear probe (0.766) is what makes MOMENT competitive. **Caveats:** one model size (small; base and large not run), one construction family (masked reconstruction; four designs compared on validation only, all worse than last value there), context 96 steps (a longer-context variant was not run), 15 minutes only. No seeds (deterministic).
**Agent and job status:** task I (1-minute offset test): `claude-uc-offset1m-smoke-1` SUCCEEDED, the full run `claude-uc-offset1m-1` is RUNNING; task J (drift analysis): `claude-uc-leveldrift-coverage-1` and `-level-1` SUCCEEDED, `-model-1` RUNNING; the fuller 15-minute offset runs (`claude-uc-offset15-rolled-1`, `-cheap-1`, `-llm-1`) are RUNNING (launched 18:10 UTC).

### 39.23 Task J (drift versus local level variation, items 5-9 of the user's list): code built, cluster runs in progress, only item 9 computed (2026-11-11 18:39 UTC; fork wrapped up at the user's request)

**Built (local commits, synced to UC):** `training/ray/level_diagnostics_coverage.py` (item 9), `training/ray/level_diagnostics_origins.py` (level job: per band, receiver and range (train, validation, test, week2) and every origin, the level distance d = window mean minus the bin's training mean (dB and in training standard deviations), quantiles q10/q50/q90/q99 and per-day median and 90th percentile; model job: per checkpoint, every 10th origin, per-origin MAE of the model, tuned median, tuned mean and lookback mean on identical origins plus sufficient statistics for the regression of excess error on d), `evaluation/analysis/level_drift_analysis.py` (tables for items 5-8 and coverage), `training/ray/scripts/run_level_drift.sh` (cluster runner), `tests/test_level_drift.py` (7 tests). Models: plain 2D VanillaLSTM, LSTMAttn and ResidualVanillaLSTM (both checkpoint rounds at 600-800, `oct2026-band-runs` at 2400-2600, seeds 40-44); the 1D VanillaLSTM is skipped. **Jobs (prefix `s3://spectrum/level-drift-oct2026/`):** `claude-uc-leveldrift-coverage-1` finished; `-level-1` 8 of 8 ok; `-model-1` RUNNING at the report (77 left, 28-70 s per checkpoint, 8 at a time; in the watchdog, resumable). **To finish:** (a) `ray job submit ... claude-uc-leveldrift-model-2 -- python -u -m training.ray.level_diagnostics_origins model --skip-done --stride 10 --max-parallel 8` if `-model-1` is no longer running with work left; (b) `... claude-uc-leveldrift-analysis-1 -- bash training/ray/scripts/run_level_drift.sh` (the offset-gain input is at `analysis-inputs/offset_response.csv`), then fetch `analysis-outputs/level_drift_report.txt` and `level_drift_*.csv`.
**Item 9, coverage of the data after 2026-11-03 20:00 UTC (the cluster copy of the raw files):** **all four core receivers in both bands have 1,440 valid minutes (all 200 bins finite) on every UTC day from 10-27 through 11-08**; on 11-09 they have 139-148 minutes (600-800: humanities 141, madsen 148, moran 148, sagepoint 139; 2400-2600: 143, 140, 140, 142) and from 11-10 none: **the cluster copy ends about 11-09 02:20 UTC, so a contiguous stretch of about 5 days 6 hours after the test week exists on all four receivers in both bands** (unofficial: the collection runs to 11-10 20:00 and is not declared final; syncing the final files would add the last day). So item 9 (repeat item 8 on a second week) is possible as soon as the model job finishes (the job already includes a week2 range 11-03 20:00 to 11-09 02:00). **No results for items 5-8 yet.**
**Caveats (from the fork):** offset gains for the slope prediction come from task I's 1-minute offset test (`offset-response-1min-oct2026`, 2D, horizon 15) if its records have the format the reader assumes (not verified), otherwise from the 15-minute gains at t+12 (600-800: VanillaLSTM 0.126, LSTMAttn 0.123, ResidualVanillaLSTM 1.000; 2400-2600: 0.376, 0.073, 1.000); day-block bootstrap intervals have only 7 blocks for the test week; the training range is in-sample for the models and validation was used for selection; checkpoint rounds 1 and 2 are pooled; model origins are thinned to every 10th (about 1,000 per range); the median and mean windows are hardcoded copies of the baselines CSV (`TUNED_K` in `origin_errors.py`).

### 39.24 Task I (offset test on the 1-minute checkpoints): code built, full run in progress, smoke results (2026-11-11 18:39 UTC; fork wrapped up at the user's request)

**Why:** Table 5's gains come from 15-minute checkpoints, but Section 5.2 uses them to explain the 1-minute gap (the 2D LSTM and LSTMAttn are 13-16% worse). **Built (commits `345295e`, `9990d48`; synced to UC; `tests/test_offset_response_1min.py`, 2 tests):** `training/ray/offset_response_1min.py` runs the 1-minute checkpoints through the benchmark's own path (`load_chunk`, `build_model`, `load_checkpoint_into_model`, `make_window_starts`, `forecast`); offsets ±5, ±10, ±20 dB applied to the normalized windows as c / std per bin (equal to shifting the raw dB values); forecast shift compared with the unshifted forecast on identical origins; gain = shift / c; October 1-minute test week, the four core receivers pooled in one job per checkpoint (no per-receiver split); **test stride 20 for 2D models (about 500 origins per receiver of about 9,900) and 100 for 1D models** (recorded in each record); models: rolled-out 1D/2D checkpoints (`checkpoints-oct2026`, `-r2`, `oct2026-band-runs`) and the 1-minute direct finals (RevIN-MLP, PatchTST, NLinear, DLinear, ResidualLinearAR and LinearAR 2D, ResidualNonlinear 2D), horizons 1/15/60; OFA and Time-LLM skipped. **Jobs:** `claude-uc-offset1m-smoke-1` finished (17 ok); `claude-uc-offset1m-1` (301 jobs, 12 at a time, `s3://spectrum/offset-response-1min-oct2026/`) RUNNING at the report, 195 left, 0 failures, in the watchdog.
**Smoke results (600-800, seed 40, gain at +10 dB; three values = t+1 / 15 / 60):** ResidualLinearAR 2D 1.0 / 1.0 / 1.0, ResidualLinearAR 1D 1.0, **ResidualVanillaLSTM 2D 1.0**, Autoformer-CSA 2D 1.0, direct RevIN-MLP / PatchTST / NLinear / DLinear / ResidualNonlinear 2D (h1) 1.0, LinearAR 2D 0.987 / 0.971 / 0.934, LinearAR 1D 0.998 / 0.995 / 0.989, TCN 2D 0.993 / 0.985 / 0.962, **VanillaLSTM 2D 0.018 / 0.001 / -0.002, LSTMAttn 2D 0.044 / 0.041 / 0.039**: **at 1 minute, as at 15 minutes, the plain 2D LSTM and LSTMAttn ignore the level of their inputs and the models with a level pathway carry it exactly** (this is the smoke on one seed; not yet compared side by side with the 15-minute table).
**To finish:** (1) check `claude-uc-offset1m-1` is done (`ray job logs ... | grep "done:"`; rerun failures with `python -m training.ray.offset_response_1min --source both --skip-done`); (2) `ray job submit ... claude-uc-offset1m-table-1 -- bash training/ray/scripts/run_offset1m_table.sh` (runs `offset_response_table.py --prefix offset-response-1min-oct2026/` at horizons 1, 15, 60; uploads `offset_response_1min.csv` and `.txt` to `s3://spectrum/analysis-outputs/`; the runner is untested); (3) fetch those two small files into `evaluation/results/`. **Caveats:** pooled receivers (not per receiver as in the 15-minute test); each direct model's rows carry only its trained horizon; the code revision field reads "unknown".

### 39.25 Finished jobs, the 1-minute offset test, the drift analysis (items 5-9), TSS-LCD 2400-2600 tuning, Reformer (2026-11-11 20:44 UTC)

**Jobs finished since 39.19 (verified from their logs):** `claude-uc-offset1m-1` 301 of 301 ok; `claude-uc-leveldrift-model-1` 180 of 180 ok; `claude-uc-offset15-rolled-1` 304 ok; `-offset15-cheap-1` 560 ok; `claude-uc-offset-llm-2` 49 ok; `claude-uc-tall-llm-2` 53 ok, 0 failed; `claude-uc-tsslcd15-tune-2400-1` finished (search and 5 finals); `claude-uc-tsl-reformer-tune-1` finished (finals 0 failed); `claude-uc-julyall-timellm-1` **finished with 23 ok and 55 failed** (probably CUDA out of memory again): rerun as `claude-uc-julyall-timellm-2` (`--gpu-fraction 1.0 --max-parallel 2 --skip-done`, 55 jobs). Still running: `claude-uc-offset15-llm-1` (336 jobs), `claude-uc-nbr-sts-1`, `claude-uc-timeran1min-2`, `claude-uc-julyall-llm-3` (UC); STS-PredNet seeds 40-42, the Time-LLM 1-minute rescoring and the spectral filler (final cluster).
**1-minute offset test (task I; `evaluation/results/offset_response_1min.{csv,txt}`, 156 runs per horizon table; pooled receivers; gain at t+1, offsets ±5/±10/±20 dB; 5 seeds unless stated):** models with a level pathway carry the level exactly at 1 minute as at 15 minutes (gain 1.000: RevIN-MLP, PatchTST, NLinear, ResidualLinearAR 1D and 2D, ResidualNonlinear 2D, ResidualVanillaLSTM 1D and 2D, **Autoformer-CSA 1.000**); LinearAR 0.98-0.99 (2D), DLinear 0.97-0.99, LinearAR 1D 0.99-1.00. **The plain recurrent models do not:** 600-800 **VanillaLSTM 2D 0.064 / 0.128 / 0.253 at -20 / -10 / -5 dB and 0.151 / 0.080 / 0.041 at +5 / +10 / +20; LSTMAttn 2D 0.059 / 0.118 / 0.230 and 0.091 / 0.040 / 0.019; VanillaLSTM 1D 0.30-0.77; TCN 2D 0.32 / 0.51 / 0.65 for negative offsets and 0.99 for positive ones (saturating)**; **2400-2600 VanillaLSTM 2D and LSTMAttn have negative gain for positive offsets (-0.11 to -0.39: the forecast moves the wrong way) and 0.04-0.18 for negative offsets**; the t+15 and t+60 tables are the same (VanillaLSTM 2D 600-800 t+60: 0.071 / 0.132 / 0.248 and 0.122 / 0.055 / 0.027). **This closes the gap in Section 5.2: the 1-minute plain 2D LSTM and LSTMAttn, the two models 13-16% worse than the median, have a level gain of 0.02-0.25, i.e. they ignore the level of their inputs at 1 minute too.** Caveats: pooled receivers, thinned origins (stride 20 for 2D, 100 for 1D), VanillaLSTM 1D at 2400-2600 has one run.
**Drift versus local level variation (task J; `evaluation/results/level_drift_*.csv`, `level_drift_report.txt`; 1,630,800 origin rows; models VanillaLSTM 2D, LSTMAttn, ResidualVanillaLSTM, 15 checkpoints each; gap = model MAE / tuned median MAE - 1; the training range is in-sample; "week2" = 11-03 20:00 to 11-09 02:00, unofficial):**
(5) **600-800, gap at t+1 / 15 / 60:** LSTMAttn **train 0.046 / 0.050 / 0.035, validation 0.141 / 0.144 / 0.116, test 0.188 / 0.199 / 0.173, week2 0.232 / 0.247 / 0.227**; VanillaLSTM 2D train 0.095 / 0.079 / 0.049, validation 0.182 / 0.172 / 0.138, test 0.232 / 0.227 / 0.186, week2 0.288 / 0.286 / 0.251; **ResidualVanillaLSTM (level pathway) is flat across ranges (t+1 0.039-0.044, t+60 0.152-0.163)**. 2400-2600: weaker and slower (LSTMAttn t+1 0.045 train, 0.049 test, 0.066 week2; VanillaLSTM t+60 0.117, 0.110, 0.140). So **the plain models' deficit is small on the training range, larger on validation (2-4 days later), larger again on the test week and largest in the second week**: it grows with the time since training, as drift predicts, and is not present at full size in-sample as local level variation alone would give; the level-pathway model shows no such growth.
(7) **The level moves farther from the training mean on test and week2:** |window mean - training mean| at 600-800 median / q90 / q99 / mean: train 0.47 / 2.91 / 6.02 / 1.04 dB, validation 0.51 / 3.08 / 5.79 / 1.09, **test 0.59 / 3.30 / 6.29 / 1.19, week2 0.61 / 3.61 / 6.62 / 1.27**; in training standard deviations the q99 doubles (1.73 train, 3.27 test, 3.22 week2); 2400-2600 median 0.24 / 0.29 / 0.32 / 0.38 (train / validation / test / week2), mean 0.86 / 1.00 / 0.99 / 1.09. The shift is real but modest in dB (the mean moves by 0.15-0.23 dB).
(6) **But level distance does not explain the excess the way the mechanism predicts:** regression of the excess error over the median on d (slope in dB per dB; mechanism: 1 - gain, with the 1-minute gains 0.078 for LSTMAttn and 0.105 for VanillaLSTM at 600-800 and about -0.07 at 2400-2600): 600-800 LSTMAttn slope train 0.002 [-0.006, 0.012], validation 0.062, **test 0.131 [0.068, 0.195], week2 0.164 [0.138, 0.200]** against a predicted 0.92; VanillaLSTM train 0.032, validation 0.093, **test 0.160 [0.092, 0.225], week2 0.203** against 0.895; R^2 0.00-0.07; 2400-2600 slopes -0.009 to +0.017 (R^2 about 0) against a predicted 1.07; ResidualVanillaLSTM about 0.01 (predicted 0). The slopes are 6-8 times smaller than the mechanism predicts at 600-800 and nil at 2400-2600, but they rise from about zero in training to 0.13-0.20 on test and week2, so level distance accounts for a small, growing part of the excess.
(8) **Within the test week (600-800)** the relative excess of LSTMAttn grows from 0.116 / 0.113 / 0.133 on days 0-2 to 0.248 / 0.305 on days 4-5 (slope +0.030 per day, se 0.0075, 7 days), VanillaLSTM 0.141 to 0.304 / 0.332, then stays at 0.20-0.28 in week2, whereas the median level distance does not grow (0.65 on day 0, 0.50 on days 3-4, 0.99 on day 6; slope +0.022 per day, se 0.033): the excess error grows over the week while the level distance does not, so **the growth is not explained by the level distance**. (9) The second-week data exist (all four core receivers in both bands have full data to 11-08 and about 140 minutes on 11-09; 5 days 6 hours after the test week): the week2 gap (0.23-0.29 at 600-800) is larger than the test week's (0.19-0.23), no plateau yet.
**What the data support (stated cautiously):** a growing deficit with time since training (drift in a broad sense) rather than a fixed local-level-variation penalty, and at 600-800 only; the specific mechanism "excess error = (1 - gain) x level distance" explains little of it (R^2 at most 7%, slopes 6-8x too small); something other than the level (spectral shape or variance changes, an overfit to the training week) must carry most of the drift, which these tests do not identify. 2400-2600 shows a weak version. Caveats: the in-sample training gap understates the deficit; validation was used for model selection; day-block intervals have 2-8 blocks; origins thinned to every 10th for the models; checkpoint rounds pooled; level distances computed at every origin.
**TSS-LCD tuned at 15 minutes, 2400-2600 (`tsslcd-authors-15min-tuned-2400`; `evaluation/results/tsslcd_all.csv` regenerated on the cluster):** t+1 / 4 / 12 / 24 / 48 = **0.684 / 0.697 / 0.713 / 0.762 / 0.770, mean 0.725 (sd 0.034)** against the authors' defaults 0.989 (off; causal 0.975), i.e. tuning gives -27%; the tuned model is 8% better than the tuned median over the same five horizons (0.786) and the last trial's validation objective was 0.775. **Reformer tuned (seed-fixed ElectroSense table, MSE ± se):** 4 channels h12 / 48 / 192 **0.376 ± 0.012 / 0.486 ± 0.022 / 0.615 ± 0.043 vs defaults 0.424 / 0.613 / 0.812 (-11%, -21%, -24%)**; 25 channels 1.052 ± 0.010 / 2.064 ± 0.044 / 2.497 ± 0.036 vs 1.180 / 2.057 / 2.515 (-11%, +0.3%, -0.7%): tuning helps Reformer at 4 channels and at horizon 12 at 25 channels, not at the long horizons at 25 channels.

### 39.26 Paper tables: the 15-minute offset CSVs regenerated, TSS-LCD hyperparameters, search winners and timings (2026-11-11 21:20 UTC; regenerated on the cluster by `training/ray/scripts/run_paper_tables_2.sh`, Ray jobs `claude-uc-papertables2-1/-2`)

**Offset test, 15 minutes, now complete for the rolled-out models (`evaluation/results/offset_response.csv`, 1.6 MB, regenerated: 20 runs per cell = 4 core receivers x 5 seeds; TCN 16 = 4 seeds; the old file had 6 = 2 receivers x 3 seeds) and the direct models (`offset_response_direct.csv`, 509 runs; the large models are still being added by `claude-uc-offset15-llm-1`: OFA, OFA controls, Time-LLM, TimeRAN, MOMENT currently have 6 runs per cell, MOMENT 20 at 2400-2600).** Gain at t+12 (600-800 / 2400-2600; the gain is constant over offsets for the linear models): models with a level pathway 1.000 (ResidualLinearAR 1D and 2D, ResidualVanillaLSTM 1D and 2D, Autoformer-CSA, RevIN-MLP, PatchTST (and its joint and no-patching variants), NLinear, ResidualNonlinear 2D, OFA and controls, Time-LLM, TimeRAN, MOMENT); LinearAR 1D 0.989 / 0.979, LinearAR 2D 0.988 / 0.957 (direct 0.977 / 0.990), DLinear 0.998 / 0.945; **plain recurrent and TCN models: VanillaLSTM 2D 0.075 / 0.147 / 0.285 at -20 / -10 / -5 dB and 0.165 / 0.077 / 0.038 at +5 / +10 / +20 (600-800), 0.090 / 0.172 / 0.301 and 0.957 / 0.473 / 0.237 (2400-2600); LSTMAttn 0.066 / 0.126 / 0.252 and 0.174 / 0.080 / 0.040 (600-800), 0.127 / 0.246 / 0.426 and -0.168 / -0.084 / -0.046 (2400-2600); VanillaLSTM 1D 0.165 / 0.368 / 0.562 and 0.662 / 0.540 / 0.396 (600-800); TCN 2D 0.197 / 0.346 / 0.480 and 0.964 / 0.965 / 0.961 (600-800; 2400-2600 1.049 / 1.037 / 1.028 for positive offsets: it overshoots); PatchTST without instance normalization 0.408 / 0.563 / 0.674 and 0.651 / 0.534 / 0.367 (2400-2600 0.326 ... 0.322)**. **Change against the old 3-seed, 2-receiver table:** the models with a level pathway are unchanged (maximum change 0.000-0.001); the plain models move by up to 0.04-0.10 at 600-800 (gain at +10 dB, t+12: LSTMAttn 0.103 to 0.080, VanillaLSTM 2D 0.105 to 0.077, VanillaLSTM 1D 0.542 to 0.540, TCN 0.962 to 0.965) and by up to 0.20-0.34 at some offsets at 2400-2600 (LSTMAttn, VanillaLSTM 1D and 2D): the qualitative picture and every conclusion are unchanged, the caveat on seeds and receivers is removed.
**TSS-LCD for the selected-hyperparameters and model-configuration tables (`evaluation/analysis/tsslcd_config_table.py` -> `evaluation/results/tsslcd_config_timings.csv`, `tsslcd_search_winners.csv`; mean over seeds 40-44, savgol off):** fixed settings: three stages (context autoencoder, future autoencoder, latent diffusion) trained 300 / 300 / 2000 epochs, 1000 diffusion steps, Adam; context / future 60 / 60 steps at 1 minute and 100 / 48 at 15 minutes (the authors' 5 attention heads need a multiple of 5; one 60-step diffusion sample per window); parameters 79.1 M (latent 32) and 81.3 M (latent 64) at 1 minute, 112.4 M at 15 minutes (context 100). **Authors' defaults:** latent 32, learning rate 1e-4, batch 256, training-time context masking 0.1. **Search (equal budget, 16 candidates = the defaults plus 15 random draws from latent {16, 32, 48, 64}, learning rate log-uniform 3e-5 to 3e-4, batch {64, 128, 256}, masking {0, 0.1, 0.2}; search seed 42; objective = validation raw-target mean-horizon MAE, no smoothing; then 5 finals):** 1 minute 600-800: winner `tsslcd-7ec88604c4` **latent 64, learning rate 4.9e-5, batch 128, masking 0.1** (validation 1.1229 vs 1.1300 for the defaults; worst 1.82; mean trial 7,051 s); 15 minutes 600-800: winner `tsslcd-43aeb379e8` **latent 32, learning rate 2.23e-4, batch 64, masking 0.2** (validation 0.7556 vs 1.2940 for the defaults, -42%; worst 1.69; mean trial 916 s); 15 minutes 2400-2600: **the same candidate `tsslcd-43aeb379e8` won** (validation 0.6997; worst 1.55; mean trial 946 s) **but only 14 of 16 trials were ok (the defaults, candidate 0, among the missing ones, so the validation comparison with the defaults is not available there and the winner was picked from 14 candidates)**: the two missing trials were relaunched as `claude-uc-tsslcd15-tune-2400-2` (it reruns only the missing trials, re-picks the winner and redoes the finals if the winner changes). **Timings (mean per run, hardware not recorded: the runs shared GPUs, so no GPU model should be quoted):** 1 minute 600-800 defaults: training 7,477 s (context AE 2,388 + future AE 90 + diffusion 4,998), wall 7,545 s, inference over all 9,888 test windows 55 s = 5.6 ms per window; tuned: training 12,707 s (wall 12,827 s, inference 98.5 s = 9.9 ms per window; batch 128 doubles the steps); 1 minute 2400-2600 defaults: training 11,695 s (wall 11,790 s; slower because of shared GPUs, not recorded), inference 76.7 s = 7.7 ms; 15 minutes (519 training windows, 625 test windows): defaults 878 s training (wall 897 s) at 600-800 and 872 s (893 s) at 2400-2600, tuned 1,015 s (1,033 s) and 993 s (1,012 s), inference 5.1-5.8 s for all test windows = 8.2-9.2 ms per window. The two window sets of 22.x remain: the first 1-minute 600-800 defaults runs used 9,888 test windows, every other set 9,962 (1 minute) or 625 (15 minutes).

### 39.27 Launched just before the PC went off (2026-11-11 21:30 UTC): tuned 15-minute spectral ablation (task K), extra 1D VanillaLSTM seeds at 2400-2600, status of the Part 1 and Part 3 gaps

**Tuned 15-minute spectral-context ablation (Part 3 gap 1; task K; commit `590a07b`, synced to UC):** `training/ray/spectral_context_fixed.py` has a new option `--hyperparameters {inherited,tuned15}` (default `inherited`: old results unchanged); `tuned15` reads each model's winner from the tuned finals (`tuning/powder15/<band>/2d/<model>/final/seed-N.json`) and trains each cell through the tuned finals' own path (`tune_fixed.base_config` and `prepared`, then the regional config) on the full test week; records carry `config_source` (`tuned15:<candidate_id>`) and `peak_gpu_mb`; `tests/test_spectral_context_tuned15.py` (3 tests). **Jobs:** `claude-uc-spectral15t-1` (5 models x 13 regions x 2 conditions x seeds 40-42 = 390 cells; prefix `oct2026-spectral-context-15min-tuned`; `--max-parallel 6 --gpu-fraction 0.25 --skip-done`; in the watchdog) and the smoke `claude-uc-spectral15t-smoke-1` (region R5, seed 40, all five models, prefix `...-smoke`; output not read). **Seeds 43-44 (650 cells in all):** `ray job submit ... claude-uc-spectral15t-2 -- python -u -m training.ray.spectral_context_fixed --resolution 15min --hyperparameters tuned15 --output-prefix oct2026-spectral-context-15min-tuned --seeds 43 44 --max-parallel 6 --gpu-fraction 0.25 --skip-done`. **Not done:** the verification (full_context over a region's bins versus the tuned final's own per-bin error for the same seed; expect a gap because both conditions train with the loss restricted to the region while the tuned finals train on all 200 bins: report it, do not treat it as an error), the cost estimate (from the smoke records' `wall_seconds` and `peak_gpu_mb`), the analysis extension (`evaluation/analysis/spectral_ablation_final.py` must read the new prefix and print the tuned table next to the inherited one; run it as a Ray job like `run_paper_tables_2.sh` with a new small runner). **Caveats:** the tuned configs may need more GPU memory than the inherited ones at 0.25 GPU per task (if the smoke shows out-of-memory failures, relaunch with a larger share); `region_only` still changes the inputs and the training loss (gap 5 not addressed); 2400-2600 and a strong cross-frequency model are not covered (gaps 3 and 4). **2400-2600 region definitions:** only a single "whole band" region (`plan_regions_2400_2600_wholeband.csv`) and the automatic segmentation (`evaluation/results/statistical_analysis/fixed_frequency_regions.csv`: 21 regions in 2400-2600 across the nodes; `frequency_segments.csv`) exist; there is no curated, behaviour-annotated plan like `plan_regions_600_800.csv` (17 regions, 13 non-noise).
**Extra 1D VanillaLSTM seeds at 2400-2600 (Part 1 item 2; `claude-uc-band2400-lstm1d-seeds-1`, running, in the watchdog):** `fixed_train --band 2400_2600 --models vanillalstm1d --seeds 40 41 43 44 --no-include-reference --test-end 2026-11-03T20:00:00Z` (the tool's default test end is the preliminary 10-30 18:00 cutoff: overridden), the 600-800 hyperparameters, output `oct2026-band-runs`; its records still have to be rescored/tabulated (stride 1) into Table 3.
**Other open Part 1 / Part 2 items (status):** Time-LLM 1-minute stride-1 rescoring `claude-final-rescore-timellm1min-2` has written 0 of 30 records after 4.5 h: its 0.15 GPU tasks do not fit the fragmented free capacity (about 0.07 per H100) and wait for the STS-PredNet runs to end (about 01:00-02:00 UTC), or relaunch with a share that fits; TCN 15-minute seed 42 has no saved weights (tuned finals exist for all five seeds, rescoring found 40, 41, 43, 44): needs a retrain of one run per band, not queued; the entropy analysis (Fig. 2) for the direct models (RevIN-MLP, PatchTST, OFA, Time-LLM) is not started (per-bin files exist; the entropy-advantage script must read them); Part 2's offset columns are regenerated (39.26); the 2400-2600 transfer columns are in `transfer_all_regimes.csv`, not yet formatted.

### 39.28 Entropy-advantage analysis extended to the direct finals (2026-11-12 01:38 UTC; `evaluation/analysis/entropy_advantage_october.py` and `entropy_advantage_regions_october.py`, run on the cluster by `training/ray/scripts/run_entropy_tables.sh` as `claude-uc-entropy-1`; `evaluation/results/entropy_advantage_october.csv` and `entropy_advantage_october_regions.csv` regenerated and committed)

**Change:** `model_tables` now also returns the direct finals RevIN-MLP, PatchTST, OFA, Time-LLM, NLinear and DLinear (labels `direct <model>`; both resolutions and both bands) from their per-bin tables, **each horizon scored from its own direct checkpoint** (the union over horizons and seeds of `rescored/per-bin/direct/<res>/<band>/2d/<model>/h<h>/seed-N.csv`, i.e. the stride-1 rescoring; where that does not yet cover all horizons (Time-LLM at 1 minute) the finals' own tables `direct-tuning/.../per-bin/` are used, **1-minute windows thinned 15x**); `n_seeds` counts distinct seeds; baseline = the validation-tuned lookback median (`--baseline median-tuned`, the default); terciles on the 1-minute entropy at both resolutions; common horizons 4/12/24/48 (15 minutes) and 1/15/60 (1 minute). Note: the working copy of these scripts already held uncommitted changes (the tuned-median baseline, gap_db columns) from before this task; they are committed together with the extension.
**15 minutes, gap_db = model minus tuned median (low / mid / high 1-minute-entropy tercile; negative = model better):**
| 600-800 | low | mid | high | | 2400-2600 | low | mid | high |
|---|---:|---:|---:|---|---|---:|---:|---:|
| direct OFA | +0.007 | -0.229 | -0.472 | | direct RevIN-MLP | **+0.001** | **-0.247** | **-0.514** |
| direct Time-LLM | +0.007 | -0.222 | -0.472 | | direct Time-LLM | +0.002 | -0.238 | -0.497 |
| direct RevIN-MLP | +0.007 | **-0.233** | **-0.487** | | direct OFA | +0.002 | -0.239 | -0.505 |
| direct PatchTST | +0.007 | -0.224 | -0.461 | | direct PatchTST | +0.002 | -0.232 | -0.493 |
| direct NLinear | +0.014 | -0.177 | -0.442 | | direct NLinear | +0.012 | -0.213 | -0.470 |
| direct DLinear | +0.020 | -0.167 | -0.441 | | direct DLinear | +0.015 | -0.191 | -0.421 |
| ResidualLinearAR 2D | +0.013 | -0.133 | -0.394 | | ResidualLinearAR 2D | +0.011 | -0.178 | -0.379 |
| LinearAR 2D / 1D | +0.018 / +0.020 | -0.120 / -0.148 | -0.372 / -0.413 | | LinearAR 2D / 1D | +0.022 / +0.013 | -0.171 / -0.195 | -0.383 / -0.432 |
| ResidualVanillaLSTM 1D / 2D | +0.023 / +0.046 | -0.159 / +0.379 | -0.438 / +0.001 | | ResidualVanillaLSTM 1D / 2D | +0.016 / +0.059 | -0.180 / +0.257 | -0.435 / +0.518 |
| VanillaLSTM 1D / 2D | +0.046 / +0.091 | -0.139 / -0.002 | -0.412 / -0.099 | | VanillaLSTM 1D / 2D | +0.027 / +0.042 | -0.084 / +0.125 | -0.268 / +0.205 |
| TCN 2D (4 seeds) | +0.027 | -0.131 | -0.315 | | TCN 2D (4 seeds) | +0.040 | -0.170 | -0.417 |
| LSTMAttn 2D | +0.114 | +0.100 | +0.001 | | LSTMAttn 2D | +0.053 | +0.146 | +0.256 |
| Autoformer-CSA 2D | +0.076 | +0.402 | +0.397 | | Autoformer-CSA 2D | +0.033 | +0.278 | +0.542 |
(1D LinearAR and ResidualLinearAR 1D: 600-800 +0.020 / +0.021, -0.148 / -0.148, -0.413 / -0.414; 2400-2600 +0.013 / +0.013, -0.195 / -0.198, -0.432 / -0.441.) **Best model per tercile:** 600-800 low OFA (+0.007; Time-LLM, RevIN-MLP and PatchTST tie at +0.007), mid RevIN-MLP (-0.233), high RevIN-MLP (-0.487); 2400-2600 RevIN-MLP in all three (+0.001, -0.247, -0.514). **Does any model beat the median in the low-entropy third at 15 minutes? No: every model has gap_db > 0 there (best +0.001 for RevIN-MLP at 2400-2600, +0.007 for OFA, Time-LLM, RevIN-MLP and PatchTST at 600-800).** The advantage of the actual best models (direct) **does grow with entropy: from +0.007 / +0.001 dB in the lowest third to -0.47 to -0.49 / -0.51 dB in the highest** (about half a dB), confirming the earlier statement with the models that matter; the direct models lead the 1D/2D models by 0.05-0.10 dB in the top third. Share of bins better than the median: 63-78% for the direct models at 15 minutes.
**1 minute (gap_db low / mid / high):** 600-800: PatchTST +0.001 / +0.004 / +0.001, Time-LLM +0.001 / +0.001 / **-0.004**, OFA +0.002 / +0.005 / +0.001, RevIN-MLP +0.003 / +0.006 / 0.000, NLinear +0.006 / +0.024 / +0.023, DLinear +0.013 / +0.037 / +0.029, ResidualLinearAR 2D +0.004 / +0.009 / +0.006, ResidualLinearAR 1D +0.005 / +0.013 / +0.007, TCN +0.010 / +0.016 / +0.011, LSTMAttn +0.115 / +0.192 / +0.222, VanillaLSTM 2D +0.152 / +0.211 / +0.269; best per tercile PatchTST (+0.001), Time-LLM (+0.001), Time-LLM (-0.004). 2400-2600: Time-LLM +0.003 / +0.038 / +0.003, OFA +0.004 / +0.024 / **-0.018**, PatchTST +0.004 / +0.031 / -0.011, RevIN-MLP +0.005 / +0.030 / -0.011, ResidualLinearAR 2D +0.007 / +0.035 / -0.014; best per tercile Time-LLM (+0.003), OFA (+0.024), OFA (-0.018). **At 1 minute no model is more than 0.02 dB from the tuned median in any tercile; only a few beat it in the highest-entropy third at 2400-2600 (by 0.005-0.018 dB).** The models without a level pathway (2D VanillaLSTM, LSTMAttn) are 0.11-0.27 dB worse at 600-800 in every tercile. Caveats: tercile grouping chosen after seeing the table (as before); Time-LLM 1-minute rows use thinned windows; no intervals; 600-800 regions CSV (13 non-noise regions) regenerated with the same models.

### 39.29 STS-PredNet offset response, full run (2026-11-12 01:54 UTC; `claude-final-offset-sts-1`: 6 of 6 ok; table built on the cluster by `training/ray/scripts/run_sts_offset_table.sh` as `claude-uc-stsoffsettable-1`; `evaluation/results/offset_response_sts.csv`, `offset_response_sts_h{4,12,24,48}.txt`)
Setting: the 15-minute STS-PredNet base variant (tuning finals, seeds 40-42, both bands); all four core receivers' series shifted by c dB at once (`series_transform`), response measured at the four physical receivers; 12 runs per cell (4 receivers x 3 seeds), thinned origins (stride 4), offsets -20 / -10 / -5 / +5 / +10 / +20 dB. **Gain (forecast shift / offset), mean over receivers and seeds:**
| band | horizon | -20 | -10 | -5 | +5 | +10 | +20 |
|---|---|---:|---:|---:|---:|---:|---:|
| 600-800 | t+4 | 0.019 | 0.003 | 0.053 | 0.022 | 0.020 | 0.011 |
| 600-800 | t+12 | -0.007 | -0.025 | -0.026 | 0.009 | 0.029 | 0.022 |
| 600-800 | t+24 | -0.057 | -0.106 | -0.009 | -0.020 | 0.074 | 0.063 |
| 600-800 | t+48 | -0.040 | -0.087 | 0.138 | -0.004 | 0.117 | 0.068 |
| 2400-2600 | t+4 | 0.022 | 0.041 | 0.070 | -0.074 | -0.033 | -0.014 |
| 2400-2600 | t+12 | 0.011 | 0.021 | 0.074 | -0.096 | 0.004 | 0.021 |
| 2400-2600 | t+24 | 0.005 | 0.010 | 0.085 | -0.240 | 0.031 | 0.044 |
| 2400-2600 | t+48 | 0.020 | 0.047 | 0.238 | -0.036 | 0.098 | 0.073 |
**STS-PredNet does not carry the level of its inputs at all: every gain is between about -0.24 and +0.24 and almost all are within +/- 0.1** (the forecast moves by under 1-2 dB for a 10-20 dB input shift), at every horizon and in both bands; this is the opposite of RevIN-MLP, PatchTST, OFA, Time-LLM and the residual models (gain 1.000) and similar to or weaker than the plain 2D LSTM and LSTMAttn (0.04-0.25). It is consistent with its in-sample error (1.20 dB against 0.90 for the linear 4D models at IDW power 2, 39.6) and with its neighbour-ablation behaviour (39.9): it forecasts from a learned typical level. Caveats: base variant only (the retrained spatial variants were not tested); gains for positive and negative offsets differ in sign at a few cells (the noise of a near-zero response: standard deviations of the per-value gain are large relative to the means); thinned origins; two bands, 12 runs per cell.

### 39.30 Strong cross-frequency models (task L, done), direct-model spectral ablation (task M, running), DSwinLSTM-I (task N, running), wider parallelism, TCN seed-42 finding (2026-11-12 02:50 UTC)

**Strong cross-frequency models, 15 minutes, direct, five seeds, test week (Part 3 gap 4; task L; commits `956e6e0`, `8ea099e`, `74a3695`, `17fb478`; `evaluation/results/strong_models_15min.{csv,txt}`; all tuning jobs `claude-final-strong-tune-*` succeeded on the final cluster):** `revinmlp_mix` is RevIN-MLP plus one zero-initialised residual `Linear(200, 200)` across the bins (`channel_mixing`, default off); `itransformer` is a self-contained port of the Time-Series-Library model (shapes, level behaviour and a training step are tested; **not compared numerically against the library**, the cluster has no TSL checkout). Both use 16-candidate searches selected on the mean validation objective at horizons 4 and 48 (winners: mix 600-800 #14, 2400-2600 #11; iTransformer 600-800 #10, 2400-2600 #11). Smoke: an iTransformer trial takes 13-24 s and peaks at 0.24 GB.
| MAE (dB), mean over 5 seeds | band | t+4 | t+12 | t+24 | t+48 | mean |
|---|---|---:|---:|---:|---:|---:|
| RevIN-MLP | 600-800 | 0.5775 | 0.7180 | 0.8103 | 0.8685 | 0.7436 |
| RevIN-MLP + channel mixing | 600-800 | 0.5809 | 0.7170 | 0.8095 | 0.8703 | 0.7444 |
| iTransformer | 600-800 | 0.5823 | 0.7135 | 0.8144 | 0.9678 | 0.7695 |
| PatchTST | 600-800 | 0.5905 | 0.7274 | 0.8238 | 0.8783 | 0.7550 |
| RevIN-MLP | 2400-2600 | 0.5221 | 0.6084 | 0.6516 | 0.6844 | 0.6166 |
| RevIN-MLP + channel mixing | 2400-2600 | 0.5242 | 0.6024 | 0.6478 | 0.6799 | 0.6136 |
| iTransformer | 2400-2600 | 0.5622 | 0.6347 | 0.6876 | 0.7487 | 0.6583 |
| PatchTST | 2400-2600 | 0.5365 | 0.6333 | 0.6630 | 0.6842 | 0.6292 |
**Finding:** cross-bin channel mixing does not help RevIN-MLP in distribution (mean +0.0008 dB at 600-800, -0.0030 dB at 2400-2600, each within about one seed sd; per-horizon signs mixed). iTransformer is worse than both RevIN-MLP variants on the mean and clearly at t+48 in both bands (+0.10 and +0.06 dB; seed sd 0.018-0.020), PatchTST slightly worse than RevIN-MLP. Caveats: MAE-only table here (other metrics are in the CSV); TSMixer not attempted; iTransformer's t+48 weakness may come from the 4/48 validation pairing or its small search space.

**Direct-model spectral-context ablation (task M; code in `0f25e90`, `950c2c7`, `3b24e1f`; driver `training/ray/spectral_context_direct.py`, analysis `spectral_ablation_final.py --strong`, cluster runner `run_strong_ablation_table.sh`):** one model per horizon (4/12/24/48) for RevIN-MLP, RevIN-MLP+mixing, PatchTST-joint (seeds 40-42) and iTransformer (seed 42 only), 13 regions at 600-800 and 5 at 2400-2600, conditions `full_context`, `region_only` and a new `input_only` (inputs masked, targets real: `data.mask.real_targets`, defaults unchanged). Smoke (48 cells, 0 failed): `full_context` matches the tuned finals' per-bin MAE over the region (mean difference +0.009 dB, seed sd 0.020 dB). `input_only` is far worse than `region_only` inside the region (RevIN-MLP 600-800 R3: 4.30 vs 1.58 dB), which looks like a real effect (loss over unpredictable out-of-region bins dominates), to be interpreted before use in the paper. Running on final: `claude-final-spectral-direct-mlp-1` (1,296 cells), `-ptj-1` (432 cells), `-itr-1` (waits on the iTransformer tuning job id). **No ablation results yet.** Table job to run when done: `claude-final-strong-ablation-table-1` (outputs `spectral_ablation_strong*.csv`).

**Tuned 15-minute spectral ablation (task K), status:** UC `claude-uc-spectral15t-1` seeds 40-42 at 200 of 390 cells after 4.9 h (TCN and Autoformer-CSA cells not started, out-of-memory risk); final `-2` seeds 43-44 at 40 of 260; the final 2400-2600 LSTM-family part is finished (30 of 30 cells, 0 failed), TCN and Autoformer-CSA (`-2400-big`) running.

**DSwinLSTM-I at 15 minutes, both bands, one seed (task N, final cluster):** still in smoke (`claude-final-dswin15-smoke-3`); table code and GPU origin scoring committed (`cb3f81f`). No results yet.

**Parallelism (the long jobs were capped low):** `--max-parallel` was 3 for the two final seed ablations (`spectral-seeds-1min-1`, `-15min-1`; 146 and 201 cells left at ~2 cells/h), 3 x 0.15 GPU for the Time-LLM 1-minute stride-1 rescoring (29 of 30 tasks left at ~8.7 h each, so ~84 h) and 2 for UC `offset15-llm-1` (264 tasks left). Relaunched with wider limits by `training/ray/relaunch_wider.py` (a cluster-side job, so it needs no operator): the two seed jobs 3 -> 6 and `offset15-llm` 2 -> 4 were restarted at 02:40 / 02:45 UTC as `-w2` (cells in flight lost; finished cells skipped); the Time-LLM rescoring (3 -> 10, `-w2`) restarts right after its next task completion (job `claude-final-relaunch-3`). New ids are in `watchdog/{final,uc}.json`. Lesson: `sync-cluster-code.py` defaults to `origin/integrate`, so unpushed commits are not synced unless `--revision HEAD` is given (the UC relaunch failed three times for that reason; the branch is now pushed).

**TCN seed 42 (Part 1 item 3):** search trials save no checkpoints, so the seed-42 weights cannot be taken from tuning; the seed-42 final ran out of memory and only 12 of 16 (600-800) and 13 of 16 (2400-2600) search trials succeeded, so the winner was chosen from a partial set. Open: a finals-only path for seed 42 and a decision on the failed trials.

**TSS-LCD weights:** the 15-minute finals save `ae_ctx`, `ae_fut` and `diffusion` weights to S3 (`save_weights: True`); the search trials do not (`False`). Existence of the objects in S3 not checked.

**GPU outlook (02:22 UTC):** UC 11.5 of 12 and final 5.7 of 6 GPUs reserved; small jobs free GPUs within 1-5 h, but the final seed ablations, the Time-LLM rescoring and the UC large-model offset runs run well past 11-13 at the old limits.

### 39.31 Stale-code failures after my sync, repaired (2026-11-12 03:00 UTC)
I synced the final and UC clusters at 02:35-02:39 with the default revision (`origin/integrate`, the older pushed `7d9268a`), which overwrote the newer committed code that tasks M and N had synced. Tasks started then failed: `spectral-direct-mlp-1` 884 of 1,296 cells (`KeyError: 'revinmlp_mix'`, `regional_config() got an unexpected keyword argument 'noise_ranges'`), `-ptj-1` 196 of 432, `-itr-1` 120 of 144. Separately, the 02:46 restart of the Time-LLM rescoring put tasks on `ray-final-2`, which lacked `/workspace/pylibs` (transformers), so every new task failed with an ImportError (the old driver had never been scheduled there). Fixes: final synced with `--revision HEAD` (always use it), `llm_powder --setup` run on all six final nodes (job `claude-final-llm-setup-2`, transformers 4.44.2 ok on all), drivers restarted as `spectral-direct-mlp-2` (parallel 6), `-ptj-2` (3), `-itr-2`, `rescore-timellm1min-w3` (10); the earlier failed cells have no records and were recomputed (0 failed after the restart). The in-flight Time-LLM tasks of the old driver were lost at the 02:46 restart (up to ~9 h each). Seeds `-w2` and tuned-spectral jobs were not affected as far as the logs show (no failures logged).
**Stopped on request (03:10 UTC):** `claude-final-spectral-seeds-1min-w2` and `-15min-w2` (seeds 43-44 of the old ablations: the 1-minute one has 3 seeds, the 15-minute inherited one is superseded by the tuned rerun; finished cells kept) and `claude-uc-offset15-llm-w2` (large-model 15-minute offset test, not in Table 5). **Changed:** Time-LLM 1-minute stride-1 rescoring restarted as `claude-final-rescore-timellm1min-w4` with `--gpu-fraction 0.5 --max-parallel 5` (was 0.15 x 10), because at ~9 h per task it looked starved for GPU.
