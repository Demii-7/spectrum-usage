# Spatial (4D) experiments: plan for the RTX 6000 cluster

Written 2026-11-09. Goal: answer the reviewers' spatial concerns and the paper's open spatial questions with the October data, using the six
receivers that have aligned training, validation and test runs. Paper due 2026-11-13, so the plan is ordered by value per GPU-hour.

## 1. What has to be answered, and why

| # | Question | Who asked | Why it matters |
|---|---|---|---|
| Q1 | How much of a 4D model's apparent accuracy comes from scoring against an interpolated (smoothed) map instead of the measurements? | paper section on spatial learning; both reviewers | the paper's main spatial claim ("the 4D advantage is largely an artifact of smoothed maps") must hold on the new data |
| Q2 | Is that conclusion an artifact of the particular interpolation (IDW power 2, dB domain, 10x10 grid)? | Reviewer 1 (weakness 3 and feedback), Reviewer 2 (feedback) | both reviewers asked for a sensitivity study and Reviewer 2 asked to "quantify its smoothing effect before training" |
| Q3 | Can a spatial model forecast a location it has never seen, from its neighbours? And can it use a new receiver's data without retraining? | the stated motivation for 4D models; the paper's holdout conditions | this is the use case that would justify a map representation at all |
| Q4 | Do the models use the geography, or only the set of series? | commented-out "permuted geometry" test in the paper | a model that is indifferent to where the receivers are has not learned spatial structure |
| Q5 | Does adding receivers help forecast a fixed set of receivers? | new | a direct dose-response test of spatial information |

## 2. Data and geometry (checked 2026-11-09)

Six October receivers with aligned runs in both bands (600-800 and 2400-2600 MHz): training `20261015T1434Z` (Oct 15 14:34 to Oct 22 13:13),
validation and test `20261025T1355Z` (validation Oct 25 13:55 to Oct 27 13:55, test Oct 27 20:00 to Nov 3 20:00). Core four: humanities, madsen,
moran, sagepoint. Additional: **guesthouse** and **law73**. `web` is excluded (it stopped mid-test). Coordinates are in `data/locations/powder.json`
("guest house", "law 73", "sage point").

Positions relative to the six-receiver centroid: humanities (-323, 105) m, madsen (255, -667), moran (127, 677), sagepoint (737, -123),
guesthouse (257, 261), law73 (-1053, -252).

* **guesthouse lies inside the core-four bounding box** (436 m from moran): forecasting it from the core four is spatial *interpolation*.
* **law73 lies outside it** (813 m from humanities, the nearest): forecasting it is *extrapolation*. Having one of each is scientifically useful.
* The receivers are sparse: the closest core pair is 727 m apart, while a 10x10 cell is about 118 x 149 m (core box) or 199 x 149 m (six-receiver
  box). About 94-96 of the 100 map cells are interpolated values. This is the quantitative core of the reviewers' concern.
* The map builder (`training/common/map_builder.py`) does **global** IDW over all receivers in the map, `power` default 2, interpolation in dB
  (or linear power with `power_domain: linear`), grid extent = the bounding box of the receivers used, centred on their mean. Receivers are
  scored at their **nearest** grid cell, so even a "receiver cell" holds an interpolated value unless a receiver sits exactly on a node.
* Implication for the holdout design: if a receiver is excluded, the grid's origin and extent move. **The grid must be fixed to the
  six-receiver box for every holdout condition** (small change: an origin and extent override in the map config).

## 3. Resolution and models

**Primary resolution: 15 minutes.** Justification: at 1 minute the best model is only about 3% better than the lookback mean and white noise
is 66-85% of the error, so an effect of spatial context would be below the noise; at 15 minutes the linear models are about 35% better than the
lookback mean, so there is room for spatial information to show (the same reasoning that limited the spectral-context ablation). 15-minute 4D
training is also roughly 10-15x cheaper (15x fewer windows; estimated, not measured). **1 minute: evaluation only** on the existing October
4D checkpoints (Q1), so the paper's existing table can be updated without retraining.

**Models** (each scored at the physical receivers, so all are comparable):

| model | why | cost at 1 minute (measured, shared H100) |
|---|---|---|
| LookbackMean 4D | reference, no parameters | - |
| LinearAR 4D, ResidualLinearAR 4D | the strongest 4D models so far (0.728 and 0.719 dB on the full grid), CPU only | 1.5 and 2.0 h on CPU |
| ResidualConvLSTM 4D | the best neural 4D model, fits 24 GB easily (1.9 GB peak) | 3.7 h |
| ConvLSTM 4D | the standard neural 4D model (5.2 GB peak) | 8.8 h |
| DSwinLSTM | **excluded** from the new training (11.7 h per seed at 1 minute); its existing checkpoints are evaluated in Q1 | |
| 2D comparators: ResidualLinearAR 2D pooled over the six receivers, and per receiver; LookbackMean 2D | a model that never sees a map, scored on the same measurements | small |

Hyperparameters: inherited from the October 1-minute 600-800 search, as for the other 15-minute inherited results (disclosed). Seeds: 3 per
cell (40, 41, 42) for the training experiments, 5 where a headline number comes from it.

## 4. Experiments

### S1. Map versus receiver scoring (Q1)

Score every 4D model three ways, exactly as the paper defines them: **full-grid** (all cells against the IDW target), **receiver-cell** (only
the cells nearest the receivers, against the IDW target), **physical receiver** (the same cells against the receiver's measured values). Add a
fourth number that needs no model: the **IDW round-trip floor**, the error between the IDW map built from the true future values and the
measurements at the receivers. It is the error a perfect map forecaster would still have at the receivers, i.e. the representation's own cost.
Compare 4D and 2D models on identical physical-receiver windows. **Prediction from the old data**: full-grid < receiver-cell << physical,
and 2D beats 4D at the receivers.

* 1 minute: existing October checkpoints (LinearAR/ResidualLinearAR/ConvLSTM/ResidualConvLSTM x 5 seeds, DSwinLSTM round 1 and 2),
  evaluation only (about 70 minutes and 50 GB host memory each; 192 GiB nodes take 3 at a time).
* 15 minutes: the S2 default-configuration runs.

### S2. Interpolation sensitivity (Q2)

Vary only how the map is built; keep data, models, splits and the scoring at physical receivers fixed. One factor at a time around the
default (p=2, dB, 10x10):

| factor | values | rationale |
|---|---|---|
| IDW power p | 1, 2, 4, 16 | p=1 smooths most; p=16 is effectively nearest-neighbour (Voronoi), the least smoothing an interpolation can do |
| interpolation domain | dB (default), linear power | averaging in linear power is physically motivated and weights strong signals more |
| grid size | 5x5, 10x10, 20x20 | fewer cells = fewer interpolated values per measured one |

That is 7 configurations (default included once). For each:

1. **Smoothing diagnostics before any training** (Reviewer 2's request): IDW round-trip error at the receivers; fraction of the map's
   variance (over time, per cell) retained relative to the receivers; spatial autocorrelation of the target map; share of cells within one
   cell of a receiver. Pure numpy over the training range; minutes.
2. **Train** LinearAR 4D, ResidualLinearAR 4D and ResidualConvLSTM 4D (3 seeds) and score with S1's four numbers.

**Falsifiable prediction**: the full-grid error moves strongly with the smoothing (lowest at p=1, highest near nearest-neighbour), while the
physical-receiver error stays roughly constant and above the 2D comparators. If the physical-receiver ranking of 4D vs 2D flips for some
configuration, the paper's conclusion is configuration-dependent and must say so.

A non-IDW method beyond the nearest-neighbour limit (Gaussian-process / kriging or RBF) is **optional** and only if time allows; p=16 already
gives a qualitatively different method (piecewise-constant Voronoi).

### S3. Unseen receivers (Q3)

Two designs:

* **Core four to the two held-out receivers.** Train on maps of the core four (grid fixed to the six-receiver box), evaluate at guesthouse
  (interpolation) and law73 (extrapolation) under two conditions from the paper: **excluded** (the held-out receiver is not in the input map;
  pure spatial generalization) and **included at inference** (its recent observations are added to the input map without retraining).
* **Leave one out within the six** (6 folds): train on five, test on the sixth, condition "excluded". Gives six independent spatial tests
  with a range of distances (436-813 m to the nearest other receiver) instead of two.

Baselines at the held-out receiver, which separate spatial knowledge from local history:
(a) **IDW nowcast** from the neighbours' last observed values (spatial interpolation without forecasting);
(b) the neighbours' lookback mean, interpolated;
(c) the held-out receiver's **own lookback mean** (uses its own history, which a spatial-only model cannot);
(d) the six-receiver pooled 2D model applied to the held-out receiver's own history (zero-shot new receiver);
(e) a per-receiver 2D model trained on that receiver (upper reference).

**Prediction from the old data**: excluded-condition models land near (a) and (b) and far above (c); "included" helps the linear models more
than the neural ones.

### S4. Permuted geometry (Q4)

Permute which coordinates each receiver is given before interpolation, keep everything else fixed, score against the true measurements.
Two variants: **frozen** (trained on the correct geometry, tested on a permuted one; evaluation only) and **retrained** (trained and tested on
the same permutation; tests whether the true geography carries information at all). 5 sampled permutations of the six receivers.
**Interpretation**: no degradation means the model does not use where the receivers are.

### S5. Number of receivers (Q5)

Train with k = 2, 3, 4, 5 receivers in the map (3 random subsets each, always excluding the evaluation receivers) plus the full set, and score
a fixed evaluation set (guesthouse and law73). Plot error against k with the baselines. Cheapest dose-response test of spatial information.

## 5. Statistics

Paired comparisons on identical test windows; mean +- standard deviation over seeds; 95% confidence intervals from a block bootstrap over test
days (7 blocks, 1,000 resamples) for the key differences; "no effect" claims stated as equivalence within +-2% with the interval shown.
All five metrics per horizon, per-bin tables and weights saved for every run (as for the other campaigns).

## 6. Compute and order

New cluster: 8 instances, each one Quadro RTX 6000 (24 GB) and 192 GiB host memory. Linear 4D models train on CPU. Rough counts at 15
minutes (one band, 600-800; 2400-2600 as a replication only if time remains):

| block | trainings | notes |
|---|---|---|
| S2 | 7 configs x 3 models x 3 seeds = 63 | includes S1's 15-minute default |
| S3 | (1 + 6 folds) x 3 models x 3 seeds = 63 | core-four runs reuse the S2 default if the grid is the six-receiver box |
| S4 | 5 permutations x 2 models x 3 seeds = 30 retrained, plus frozen evaluations | ResidualLinearAR 4D and ResidualConvLSTM |
| S5 | 4 k-values x 3 subsets x 2 models x 3 seeds = 72 | |

About 230 trainings, two thirds of them CPU-only linear models. With 15-minute ResidualConvLSTM at an estimated 20-40 minutes on a shared H100
(unmeasured; an RTX 6000 is slower), the GPU part is about 80 runs, so roughly 10-20 hours over 8 GPUs. Order: (1) S1 at 1 minute on the A100
now (evaluation only); (2) 15-minute 4D pipeline and smoke; (3) S2 diagnostics (no training); (4) S2 training; (5) S3; (6) S4; (7) S5.
If the cluster arrives late, S5 and the S4 retrained variant are dropped first.

## 7. Code needed

1. `apply_powder15_4d`: the 15-minute configuration with the map kept (separate cache name per map configuration, lookback 96); smoke test.
2. Map configuration overrides: fixed origin and extent (six-receiver box), power, domain, grid size, selected/excluded sites, permutation
   seed. Most exist in the map builder; origin/extent override is new.
3. A spatial runner (like `tune_direct`, bounded queue): builds the variants, trains, evaluates through `training/common/spatial_ablation_eval.py`
   (which already computes full-grid, receiver-grid and physical-receiver errors on frozen checkpoints), adds the baselines (a)-(e), saves results.
4. The smoothing-diagnostics script (S2 step 1).
5. Check before use: that the evaluator's site-name matching finds "guest house" and "law 73"; that the evaluator runs on four-receiver
   October checkpoints (it was written for the older six-receiver set).

## 8. How the results map to the paper

| result | paper change |
|---|---|
| S1 (both resolutions) | replace the 0.861 / 1.134 / 1.982 dB numbers; add the round-trip floor |
| S2 | new paragraph and small table for Reviewers 1 and 2; quantify the smoothing (diagnostics) |
| S3 | rewrite the holdout paragraph with interpolation (guesthouse) vs extrapolation (law73) and leave-one-out |
| S4 | restore the commented-out permuted-geometry paragraph with numbers |
| S5 | one figure if time allows |

## 9. Additions (2026-11-09)

* **Kriging** joins S2 as an eighth map construction: ordinary kriging with one variogram per band, fitted on the training range and
  then frozen. Six receivers give only 15 pairs of distances, so the variogram will be rough; its per-cell prediction variance is
  reported as a direct measure of how much of the map is guessed.
* **STS-PredNet** has no October checkpoints. Train it (registry search, then 5 seeds, both bands, 1 minute, recent-history branch
  only) as part of the main 4D comparison and include it in S1; then redo the daily-history pair (one and two previous days, with
  LinearAR given the same inputs). The paper's daily-history numbers (1.401 / 1.230 dB) are from the July data.
* **ConvLSTM-FM** stays out. Its source paper pretrains on about 24 s of millisecond-scale IQ spectrograms from one testbed with no
  released weights, and our adaptation never trained sensibly (best July trial 2.47 dB vs 0.96 for the lookback mean).
* **web**: training and validation 100% covered, test 70% (gap 2026-10-28 15:20 to 10-30 18:24). Use it as a seventh input receiver
  (S5) and as a held-out receiver scored only on its covered test windows, reported separately; never in headline numbers.

## 10. After collection ends (2026-11-10 15:00): drift and transfer

* **Second test week** (2026-11-03 20:00 to the last timestamp, about 2026-11-10 15:00): rescore every frozen checkpoint and the
  lookback means. Report error per day against days since training ended (about 5 to 19 days). The drift measure is each model's
  error divided by the lookback mean's error on the same day (the lookback mean has no training and cannot age), so a harder week
  does not masquerade as degradation. Evaluation only.
* **Transfer tests** (new receiver, new bands, new testbeds, as in the paper's transferability table) on the frozen October
  checkpoints, including the TimeRAN/MOMENT/OFA finals. Before reporting AERPAW transfer for TimeRAN, confirm its pretraining corpus
  (AERPAW datasets 18, 20, 23, 24, 25: phone and UAV cellular measurements) does not include the AERPAW spectrum-monitoring data.

### Transfer protocol: adaptation ladder (agreed 2026-11-09; extends REVISION_NOTES section 2)

Every transferred model, all three representations, scored on identical target test windows under:

| step | target data | change | isolates |
|---|---|---|---|
| A zero-shot | none | source normalizer | the current (harsh) protocol |
| B per-window centering | none | input level removed and restored | shape forecasting given the level |
| C refit normalizer | first X hours | per-bin mean/std refit, weights frozen | normalization shift alone |
| D fine-tune | first X hours | weights updated, one fixed recipe | target dynamics beyond the level |
| E references | first X hours | lookback mean; LinearAR and the same architecture trained only on those hours | transfer vs starting from scratch |

* Calibration: POWDER targets (guesthouse, law73, web, other bands) use their own validation run (2026-10-25 to 10-27) and are scored
  on the core test week; ARA, COSMOS and AERPAW use the first X hours, with the test segment fixed as everything after the largest X.
* X in {1, 6, 24} hours where the trace allows.
* Fine-tune recipe, never tuned on target data: up to 5 epochs, learning rate 0.1x the source winner's, early stopping on the last
  20% of the calibration window; head-only for OFA and TimeRAN (linear probing), full otherwise.
* Report error, skill vs the lookback mean, and a bias/shape split for every step (answers Reviewer 1's request to diagnose the failures).
