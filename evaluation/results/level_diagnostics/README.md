# Why don't learned corrections beat simple baselines on 1-minute POWDER data? (CPU-only diagnostics, Nov 2026)

## Motivation

From working with SpectrumLLM and its data, "learnability" seemed to depend strongly on sampling resolution:

- 1-minute data (AERPAW/TSS-LCD, our POWDER) shows little learnable structure beyond simple baselines.
- 15-minute data (SpectrumLLM/ElectroSense, or POWDER collapsed to 15 min) looks more learnable.
- The DeepSPred 1-second claims look implausible.

Models with an output baseline/skip pathway (`ResidualLinearAutoregressive` and the residual LSTM and ConvLSTM models) already close most of the gap between poor and competitive models. So the open question is narrower: **why does the learned correction on top of the pathway add so little at 1 minute, and what changes when data are aggregated?**

Everything here uses transparent, CPU-only predictors: rolling means and medians, EWMA, local-level Kalman filters, ridge and gradient-boosting probes. No neural checkpoints were evaluated, and the plan's audit of real models' pathway and correction is still open (see [Open items](#open-items)).

Plan: `~/.claude/plans/come-up-with-a-streamed-coral.md` (not in the repo).

## Data and protocol

- **Receivers and bands:** POWDER `humanities`, `madsen`, `moran`, `sagepoint` (`-nuc1`). Bands 600–800 and 2400–2600 MHz, 200 one-MHz bins each. That makes 8 receiver × band cells, called cells below.
- **Split** (October 2026 protocol, the same one the models use):

  | split | interval (UTC) | source run |
  |---|---|---|
  | train | 2026-10-15 14:34 → 10-22 13:13 | `20261015T1434Z` |
  | validation | 2026-10-25 13:55 → 10-27 13:55 | `20261025T1355Z` |
  | test | 2026-10-27 20:00 → 11-03 20:00 | `20261025T1355Z` |

- **Selection:** every hyperparameter is chosen on validation only and frozen before test. This covers the lookback length and the kind of simple baseline (the "strong" baseline), ridge penalty, shrinkage α of a correction, and GBM iterations.
- **Metric:** MAE in dB. Skill = 1 − MAE / MAE_ref. CIs are 95% paired moving-block bootstraps over time origins (block of 240 origins).
- **Notation:**
  - Δ = input sampling interval.
  - H = lead time to the target endpoint, in minutes.
  - W = target averaging width.
  - "point" target = the 1-minute value at t+H (W=0, as in our model evaluation).
  - "avg15" target = the mean over (t+H−15, t+H] (W=15).
- **Averaging:**
  - Averaged targets and blocks average in **linear power**, then convert to dB.
  - A "dB mean" averages the dB values directly. That is a different and, because of Jensen's inequality, biased predictor of a linear-power average.
- **Pre-registered threshold:** a practical improvement is ε = 0.02 (2% skill). "No practical improvement" is a one-sided claim, meaning the CI upper bound is below ε. No multiple-comparison correction has been applied, so treat per-cell CIs as descriptive.

**Code:**
- `training/ray/level_diagnostics.py` (commits `a54a1c0`, `031af6c`), run as CPU-only Ray tasks on the mixed-GPU cluster:
  - `claude-leveldiag-full-1`: `--part main`, output in `full-1/`.
  - `claude-leveldiag-sampling-1`: `--part sampling`, output in `sampling-1/`.
- Tests: `tests/test_level_diagnostics.py` (synthetic local level with known variances).
- Summaries and derived tables: `scripts/`.

## Summary of findings

1. **Within each one-minute series, the data closely follow a local-level model.** That is a slowly drifting level plus short-timescale fluctuation that is close to white:
   - Changes have lag-1 autocorrelation ≈ −0.48 in every cell, with lags 2–3 ≈ 0.
   - The fit is stable across train, validation and test.
   - Forecast calibration is 0.7–1.2.
2. **Most of what can be predicted is estimating that level.** A validation-tuned 15–30-minute **median** is the best simple predictor.
   - It beats the 60-minute mean by 3–8% and persistence by about 20–23% (MAE).
   - The local-level Kalman filter is within 1% of it.
   - The 24-hour seasonal naive predictor is far worse.
3. **Linear (MSE-trained) corrections on top of the level add nothing over the strong baseline.** In 22–23 of 24 cells the CI upper bound on skill is below ε, and 13–15 are significantly *worse*.
   - Whatever the anchor (last value or window mean), more than 90% of the learned correction is "pull toward the window mean", which is just rebuilding a smoother.
   - Training, validation and test gains match, so this is not overfitting. There is simply little linear structure left.
4. **The leftover structure is nonlinear and asymmetric.** Dips persist, while spikes revert almost completely.
   - A gradient-boosted model trained on *absolute* error beats the strong baseline in all 24 cells.
   - The gain is +1–3% at H = 1 and 15, and +3–10% at H = 60 (point estimates; no CI vs strong yet).
   - This is the only learned gain that clears ε, and it is modest.
5. **Aggregation mainly changes the target, not the learnability.**
   - Averaging the target over 15 minutes removes most of the white noise. That noise is 65–80% of the best achievable point-target MSE but only 11–21% for 15-minute averages, so absolute errors fall a lot.
   - Measured against a validation-tuned baseline *for the same target*, ridge on the averaged target gains only 0–3%.
   - Larger "gains" (8–17%) appear only against a mismatched dB-mean baseline.
   - Aggregated *inputs* add no information, since blocks can be computed from 1-minute history. They are within about 3% of 1-minute history for averaged targets (slightly better at 600 MHz, which is a regularisation effect for ridge) and 4–14% worse for point targets.
   - 15-minute **decimation** (keep every 15th sample) is much worse than averaging: 21–44% for averaged targets.
6. **Coarse inputs make simple baselines weak, which flatters learned models.** On 15-minute blocks a ridge model does beat the best simple block baseline, by up to about 6%. But it still loses to a simple 1-minute median. The "learnability" here is the model recovering what the coarse baseline threw away.
7. **The SpectrumLLM ElectroSense data look like temporal averages, not samples.**
   - The ElectroSense API serves time- and frequency-aggregated data (`aggTime`, `aggFreq`, `aggFun` = AVG/MAX).
   - The noise statistics of the SpectrumLLM files match POWDER 15-minute averages, not 15-minute samples.
   - Those results combine an averaged target, multi-hour horizons where the daily cycle is visible, and weak baselines. All three make models look better than at 1 minute.

## 1. Local-level structure (`full-1/statespace.csv`)

The model is y_t = μ_t + ε_t with μ_{t+1} = μ_t + η_t.

- Under this model, first differences have autocovariances γ0 = 2σε² + ση² and γ1 = −σε², with γ_k = 0 for k ≥ 2. So the lag-1 correlation ρ1 lies between −0.5 and 0, and ρ1 ≈ −0.5 means mostly white noise on a slow level.
- Measured, as medians over bins, in all 8 cells and all three splits:
  - ρ1 is between −0.47 and −0.50.
  - ρ2 is between −0.02 and 0.00.
  - ρ3 is between −0.01 and 0.01.
  - Every bin has ρ1 < 0.
- Fit parameters:
  - The prediction-error fit of a steady-state Kalman gain on train gives a signal-to-noise ratio q = ση²/σε² ≈ 0.003–0.016 and gain K ≈ 0.05–0.12. In other words, the level moves about 0.05–0.13 σε per minute.
  - Plugging moments into γ0 and γ1 gives larger, noisier q (0.02–0.12). The filter driven by the moment estimate is noticeably worse (`kalman_moments` in the table below), so we use the prediction-error fit.
- Calibration: test MSE divided by predicted variance is 0.87–1.24 at H = 1 and 0.70–1.10 at H = 60. The model slightly *over*-predicts long-horizon error, which is consistent with a level that is bounded or mean-reverting rather than a pure random walk (see §5).

## 2. The baseline ladder (`full-1/results.csv`, experiment `baselines`)

Test MAE in dB, mean over the 4 receivers. All predictors use only history up to the origin.

| predictor | 2.4 GHz H=1 | H=15 | H=60 | 600 MHz H=1 | H=15 | H=60 |
|---|---|---|---|---|---|---|
| training-set constant | 1.175 | 1.175 | 1.175 | 1.527 | 1.527 | 1.527 |
| seasonal naive 24 h | 1.066 | 1.066 | 1.066 | 1.437 | 1.437 | 1.437 |
| persistence | 0.845 | 0.931 | 1.024 | 1.064 | 1.166 | 1.250 |
| 60-min mean, linear power | 0.838 | 0.875 | 0.982 | 1.003 | 1.040 | 1.137 |
| 60-min mean, dB (`mean60`) | 0.715 | 0.754 | 0.866 | 0.899 | 0.936 | 1.038 |
| 60-min median | 0.693 | 0.734 | 0.851 | 0.873 | 0.912 | 1.017 |
| local-level Kalman (PE fit) | 0.671 | 0.729 | 0.836 | 0.855 | 0.914 | 1.015 |
| **strong** (validation-selected) | **0.664** | **0.718** | **0.826** | **0.854** | **0.900** | **1.003** |

- **Tracking the level is the big gain.** Going from persistence to the strong baseline cuts MAE 19–23%. Going from the training-set constant cuts it 30–44%.
- **Median beats mean.** Because the metric is MAE, the conditional median is the optimal point forecast. The 1-minute power distribution is right-skewed (skew 0.5–2.2), so medians win.
- **The strong baseline is a 15–30-minute median** in 23 of 24 cell × horizon combinations. The other one is EWMA with α = 0.2.
  - It beats `mean60` by 2.5–8.1% in every cell.
  - Its per-day skill vs `mean60` is positive on every test day (`full-1/daily.csv`).
- **Seasonal naive is poor.** The daily cycle is small relative to minute-scale variation.

## 3. Do learned corrections add anything? (`full-1/results.csv`, experiments `probe_*`; `derived_probe_vs_strong.csv`)

**Probes.** Every probe predicts the 1-minute target from the last 60 minutes of the same bin, with validation-selected ridge penalty and correction shrinkage α:

- **anchored ridge**, ŷ = b + w·(x − b), with the anchor b being:
  - the window mean (`mean60`, the same parameterisation as `ResidualLinearAutoregressive` with `baseline: mean`);
  - the last value (`last`, the new `baseline: "last"` option);
- **plain affine AR** (`none`): the same ridge regression, but the weights are not forced to sum to 1;
- **pooled ridge across bins** (`pooled_linear`), with an extra-covariates variant (`pooled_rich_linear`: local slope, variability, time of day, cross-bin context);
- **pooled HistGradientBoosting trained on absolute error** (`pooled_gbm_mae`), using the same features.

Skill vs the strong baseline, min / median / max over the 4 receivers:

| predictor | 2.4 GHz H=1 | H=15 | H=60 | 600 MHz H=1 | H=15 | H=60 |
|---|---|---|---|---|---|---|
| ridge, last anchor | −.046 / −.015 / −.005 | −.047 / −.021 / −.009 | −.047 / −.015 / +.007 | −.008 / +.001 / +.008 | −.021 / −.011 / −.007 | −.015 / −.008 / +.001 |
| ridge, mean anchor | −.043 / −.014 / −.006 | −.044 / −.021 / −.009 | −.044 / −.016 / +.005 | −.008 / .000 / +.007 | −.021 / −.012 / −.007 | −.015 / −.008 / +.001 |
| affine AR (`none`) | −.062 / −.019 / −.009 | −.075 / −.026 / −.009 | −.091 / −.009 / +.016 | −.017 / −.005 / +.005 | −.040 / −.022 / −.011 | −.044 / −.019 / −.003 |
| pooled ridge | −.041 / −.012 / −.003 | −.043 / −.019 / −.007 | −.046 / −.015 / +.005 | −.010 / −.002 / +.001 | −.024 / −.012 / −.010 | −.019 / −.010 / −.005 |
| **GBM, absolute-error loss** | **+.009 / +.027 / +.033** | **+.010 / +.025 / +.033** | **+.027 / +.066 / +.102** | **+.028 / +.030 / +.036** | **+.020 / +.022 / +.025** | **+.038 / +.049 / +.062** |

**One-sided ε test** for the per-bin anchored ridge corrections, with bootstrap CI vs strong:

| correction | cells with CI upper bound < 0.02 | cells significantly worse | cells significantly better |
|---|---|---|---|
| mean anchor | 23 / 24 | 14 | 1 |
| last anchor | 23 / 24 | 13 | 1 |
| plain affine AR (`none`) | 22 / 24 | 15 | 1 |

For the GBM, we only have CIs vs `mean60`: 0.054–0.163, with every lower bound above 0.03. A CI vs strong has not been computed yet.

### What the correction learns

- **Mean anchor and last anchor give identical forecasts.** With the inputs centred on the anchor, both are an affine map whose weights sum to 1, so they are the same function class. Unregularised they give identical forecasts (max difference 0.0000 dB, `full-1/equivalence.csv`), and regularised they are within 0.004 in skill.
  - This class is level-shift equivariant: add c dB to the history and the forecast moves by c.
  - The plain affine AR (`none`) is a strictly larger class that can also pull toward a fixed training level. Its forecasts differ by up to 12 dB, and it is the *worst* of the three. Equivariance helps here.
- **The last-anchored correction is mostly the cheap correction** c_mean = mean60 − last (`full-1/cheap.csv`). Regressing the learned last-anchor correction on that pull gives a slope of 0.77–0.95 with R² = 0.89–0.99. The correction is worth 0.15–0.25 dB of MAE over persistence on its own (`correction gain` in `full-1/summary.txt`). That is the learned model rebuilding a smoother.
- **No overfitting.** The mean-anchor correction's MAE gain over its anchor is the same on train, validation and test (0.024–0.054 dB). Validation chose α ≈ 1–1.25. What the correction finds is real but small, and it does not beat a better-chosen *simple* level estimate.
- **The residual is asymmetric** (`full-1/condmedian.csv`). This is the median of (y − mean60) by decile of the last value's deviation (last − mean60), median over cells, at H = 1:

  | deviation decile | 0 (deep dip) | 1 | 2 | … | 8 | 9 (big spike) |
  |---|---|---|---|---|---|---|
  | deviation (dB) | −2.05 | −0.82 | −0.34 | | +0.54 | +2.34 |
  | residual median (dB) | −0.91 | −0.42 | −0.21 | | +0.03 | +0.11 |

  - **Dips persist:** about 45% of a dip is still there at H = 1, and about 40% at H = 60.
  - **Spikes revert:** less than 5% of a spike remains, and at H = 15–60 it slightly overshoots.

  A symmetric linear correction cannot represent this, and a median of the window partly does. A tree model trained on absolute error can learn it directly, which is the most likely explanation for the GBM's 2–10%. It is untested which part matters: the loss (MAE vs MSE) or the nonlinearity.

## 4. Aggregation, sampling and like-for-like horizons (`sampling-1/`, `full-1/` experiment `aggregation`)

### Input representations

Every option uses the same origins (aligned to 15-minute boundaries) and the same targets, with history L = 60 or 240 min:

- **1min:** the raw 1-minute history.
- **block:** non-overlapping 15-minute linear-power averages.
- **decim:** every 15th 1-minute sample. Results are averaged over all 15 phases; the spread across phases is small, SD ≈ 0.01–0.02 dB.

### Predictors

For each representation, H ∈ {15, 30, 60} and target ∈ {point, avg15}:

- **strong:** a validation-selected simple predictor from that representation (last value, mean or median over k steps, or linear mean);
- **ridge:** a validation-tuned ridge on the full history.

### Results

Test MAE (dB, mean over receivers), L = 60 min. Each cell gives strong / ridge.

| target | input | 2.4 GHz H=15 | H=30 | H=60 | 600 MHz H=15 | H=30 | H=60 |
|---|---|---|---|---|---|---|---|
| point | 1min | **.733** / .752 | **.775** / .785 | **.833** / .858 | **.896** / .917 | **.932** / .951 | **1.001** / 1.017 |
| point | block | .831 / .774 | .868 / .802 | .942 / .881 | 1.006 / .948 | 1.033 / .982 | 1.098 / 1.048 |
| point | decim | .817 / .822 | .852 / .855 | .928 / .930 | 1.003 / 1.002 | 1.035 / 1.034 | 1.098 / 1.096 |
| avg15 | 1min | .447 / **.422** | .510 / **.488** | .614 / **.591** | .486 / .479 | .549 / .543 | .648 / .638 |
| avg15 | block | .450 / .434 | .509 / .500 | .616 / .608 | .480 / **.463** | .542 / **.529** | .643 / **.628** |
| avg15 | decim | .621 / .564 | .667 / .614 | .750 / .703 | .698 / .656 | .743 / .703 | .820 / .783 |

**Sampling vs averaging at 15 minutes:**

- For the averaged target, block averages are about as good as 1-minute history: the best predictor is within ±3% (slightly worse at 2.4 GHz, slightly better at 600 MHz).
- Decimated samples are 21–44% worse than the 1-minute strong baseline. A 15-minute *sample* is a single noisy minute. Averaging is an efficient summary for an averaged target. Sampling throws most of the information away.
- For the point target, the best predictor on blocks is 4–6% worse than the 1-minute strong baseline, and on samples 10–14% worse. The most recent minutes matter for the next minute's value.

**Like-for-like 60-minute-ahead forecast** (`derived_like_for_like_best_mae.csv`, best predictor per receiver):

| target | 1-min history | 15-min blocks | 15-min samples |
|---|---|---|---|
| point, 2.4 GHz / 600 MHz | **0.83** / **1.00** | 0.88 / 1.05 | 0.93 / 1.09 |
| avg15, 2.4 GHz / 600 MHz | **0.59** / 0.64 | 0.61 / **0.63** | 0.69 / 0.76 |

Predicting 60 minutes ahead from 1-minute data is never worse than from 15-minute data. The 15-minute target is just easier: about 0.6 dB vs 0.8–1.0 dB.

### Learnability = skill over the best simple predictor *of the same representation*

Ridge skill vs that representation's own strong baseline (`sampling-1/summary.txt`):

| target | input | L=60: 2.4 GHz H=15/30/60 | L=60: 600 MHz H=15/30/60 | cells with CI lower bound > 0 |
|---|---|---|---|---|
| point | 1min | −.032 / −.031 / −.031 | −.024 / −.022 / −.016 | 0 / 24 (24 / 24 have UCB < ε) |
| point | block | +.040 / +.046 / +.040 | +.059 / +.051 / +.047 | 24 / 24 |
| avg15 | 1min | +.027 / +.013 / .000 | +.018 / +.015 / +.017 | 9 / 24 |
| avg15 | block | +.034 / +.019 / +.014 | +.034 / +.025 / +.023 | 21 / 24 |
| avg15 | decim | +.040 / +.035 / +.029 | +.063 / +.057 / +.048 | n/a |

This is the mechanism behind "15-minute data are more learnable":

- **Coarse inputs make the best simple predictor weak.** A block baseline can only use 2–4 blocks, and a sample baseline only a few noisy points. A learned model recovers part of what the coarse summary lost, so its skill *vs that baseline* goes up. It still loses to the simple 1-minute median: ridge on blocks for point targets is 4–6% worse than 1-min strong.
- **Against an appropriate baseline for the same target, gains on averaged targets are small.** With 1-minute history the gain is 0–3%, and with L = 240 it is about 0.
  - The `full-1` aggregation arms show ridge "skill" of 6–17% on avg15 targets. That is against a **dB mean** of the history, which is biased low for a linear-power average.
  - The validation-selected baseline here is a 15–30-minute **linear-power** mean, which removes most of that apparent gain.
  - With L = 240 the dB mean also averages over 4 hours. That is why `full-1` arm C shows 32–42% "skill", 28–38 points of which a single last block also achieves.

## 5. How much error is irreducible? (`sampling-1/sampling.csv` rows `part=floor`; `derived_floors.csv`)

Under the fitted local-level model (train parameters), the best possible MSE of a causal forecast of a target averaged over W minutes is:

  floor = σε²/W (white noise in the target; for a point target, W = 1)
        + K·σε² (error in estimating the current level; steady-state Kalman)
        + ση²·V(H, W) (random-walk drift of the level, averaged over the target window).

Shares of the **best achieved test MSE** across all predictors and representations, median over 8 cells:

| target | H | white noise σε²/W | noise + level estimation |
|---|---|---|---|
| point | 15 | 0.79 | 0.85 |
| point | 30 | 0.74 | 0.80 |
| point | 60 | 0.65 | 0.70 |
| avg15 | 15 | 0.21 | 0.49 |
| avg15 | 30 | 0.16 | 0.36 |
| avg15 | 60 | 0.11 | 0.25 |

- **Point targets.** About two thirds to four fifths of the best achievable squared error is the unpredictable minute-to-minute fluctuation, and level estimation adds another 5–6%. Only 15–30% is left for any model to win.
- **Averaged targets.** Averaging divides the white part by about 15, so most of the remaining error is level movement. That gives a model far more room relative to persistence, though not relative to a good level estimator, which is already capturing it.
- **The random-walk drift term breaks down beyond about 15 minutes.**
  - Already at H = 30, the total "floor" exceeds the best achieved MSE in 5 of 8 cells for point targets. At H = 60 it does so in 6 of 8 cells for both targets (`floor_over_best` > 1). The level is bounded or mean-reverting, not a random walk.
  - So the full sum is **not** a valid lower bound at long horizons. The white-noise term on its own still is, provided the noise is independent.
- **Exception: `moran-nuc1` 2400–2600.** The train-fitted σε² exceeds the test best MSE there (share > 1). That cell is near the noise floor, with very low variance (σε² ≈ 0.03 dB²) and occasional bursts. The train noise estimate does not carry over, so treat this cell's floor as uninformative.

These are "irreducible" only relative to the local-level decomposition. They are not a certified bound for all predictors. The entropy/Fano analysis would be a separate, model-free check, and it hasn't been run.

## 6. SpectrumLLM's 15-minute ElectroSense data (`electrosense_noise_check.csv`)

### Is it averaged?

The SpectrumLLM paper only says "sampled at 15-minute intervals" from the ElectroSense open API (see `training/SpectrumLLM/MODEL_CARD.md`). The ElectroSense API documentation shows how such data are produced:

- **Endpoint:** non-owners can only fetch `/spectrum/aggregated`, which is aggregated "in time & frequency". The parameters are:
  - `aggTime`: time resolution in seconds, required;
  - `aggFreq`: frequency resolution in Hz, required;
  - `aggFun`: `MAX` or `AVG`.
- **Resolution floor:** for sensors you don't own, data come at 60 s and 100 kHz or coarser. Raw FFT data are available for your own sensor only, and at most 3 minutes per request.
  - Sources: [electrosense/api-examples](https://github.com/electrosense/api-examples), its `doc/electrosense-openapi.pdf`, and the [API spec](https://electrosense.org/app.html#!/api-spec).
  - The official example scripts use `aggFun=AVG`.
- **Sensor-side averaging:** sensors already average 5 FFT frames before upload ([arXiv 1903.05408](https://arxiv.org/pdf/1903.05408)).
- **Other ElectroSense papers** use 60 s and 10 kHz ([1707.08908](https://arxiv.org/pdf/1707.08908)) or 1-minute Madrid data ([2412.09849](https://arxiv.org/pdf/2412.09849), [2508.17872](https://arxiv.org/pdf/2508.17872)). None documents a 15-minute product.

So even ElectroSense "1-minute" data are 60-second aggregates. A 15-minute file is most naturally `aggTime=900`. Its 4/25/151-channel versions (100 / 12.5 / 2 MHz) imply a large `aggFreq` too.

The data's own statistics agree. Below are the noise fingerprints from `scripts/electrosense_noise_check.py`: three ElectroSense files vs the POWDER train week converted three ways, 8 cells each (medians over channels, then the range over files or cells).

| series | lag-1 autocorr. of changes | white-noise share of variance | white-noise share of change variance | skew |
|---|---|---|---|---|
| ElectroSense spec4 / 25 / 151 | −0.35 to −0.37 | 0.075–0.13 | 0.70–0.74 | −0.13 to +0.08 |
| POWDER, 15-min linear-power averages | −0.30 to −0.38 | 0.04–0.15 | 0.60–0.77 | −0.30 to +0.66 |
| POWDER, every 15th minute | −0.47 to −0.50 | 0.29–0.61 | 0.93–0.99 | 0.34–1.27 |
| POWDER, 1-minute | −0.47 to −0.49 | 0.26–0.51 | 0.94–0.99 | 0.48–2.24 |

ElectroSense matches the *averaged* series in every column and the *sampled* series in none. Its values are quantized to 0.1 and positive (5.5–50.4, units unstated). It is very likely `AVG` aggregation in time and frequency. That is not confirmed: the `aggTime`/`aggFreq`/`aggFun` used, and whether averaging is in linear power or dB, would need the SpectrumLLM authors.

### Why it looks more learnable

1. **The target is averaged.** By §5, white noise drops from about 65–80% of the best squared error to about 10–20%, so models have far more room relative to persistence.
2. **Long horizons and lookback.** The setup forecasts 3–48 h ahead from 48 h of history, so the daily cycle is visible. In our 15-minute repro, a 24-hour seasonal naive predictor beats several published models (`training/SpectrumLLM/repro/RESULTS_PAPER_REGIME.md`). At our 1-minute ≤ 60-minute setting, seasonal naive is useless (§2).
3. **Weak baselines.** The paper compares against persistence, a 48-hour lookback mean and seasonal naive, never a validation-tuned short mean or median. On POWDER, that kind of baseline removes most learned gains (§3, §4).
4. **Metric.** The paper reports MSE in raw units, dominated by the high-variance channel with step changes. Persistence is punished at every step.

## Implications

- At 1 minute, the right reference is a **validation-tuned short-window median or local-level filter**, not persistence or a long mean. Learned models should be reported as skill over that.
- A level-preserving, equivariant output pathway is the right inductive bias: its centred correction behaves the same at every signal level. Linear corrections on top of it are not worth their cost. What is left is nonlinear and asymmetric (dips persist, spikes revert), and models should be trained on the reported metric (MAE).
- "15-minute data are more learnable" is mostly:
  - (a) an easier, averaged target;
  - (b) weaker simple baselines on coarse inputs;
  - (c) at SpectrumLLM's horizons, a visible daily cycle.

  It is not more information. Aggregated inputs are at most about 3% better than 1-minute history on the same target (block averages at 600 MHz), and usually worse. Decimation, the literal "sample every 15 minutes", is clearly harmful.
- For the DeepSPred 1-second claim, the same logic predicts an even larger white-noise share at 1 s unless the data are averaged. That needs checking on their data.

## Open items

- **Real-model audit not done.** No neural checkpoints were evaluated. The pathway-alone / full / α-scaled / strong table for the trained residual models is still to do (GPU-free, but needs checkpoint inference).
- **GBM:** bootstrap CI vs strong. Also test which part produces the gain: (i) GBM trained on MSE, (ii) a median-anchored ridge, (iii) a GBM with robust-quantile features only.
- **ElectroSense:** rerun the baselines in the paper's regime with a validation-tuned short mean/median and a local-level filter. Ask the authors for the `aggTime`/`aggFreq`/`aggFun` they used.
- **Coverage:** AERPAW / TSS-LCD has not been analysed with this pipeline. POWDER is one test week, 4 receivers and 2 bands.
- **Entropy diagnostics:** the Ding et al. 2015 level-based Q = 8 entropy with Fano bound, plus surrogates and estimator calibration, have not been run. Note that the repo's entropy code uses LZ78 phrase counting, not Ding's match-length estimator.

## Files

- `full-1/`: main diagnostics output.
  - `results.csv`: all predictors, all splits, MAE and skills with CIs.
  - `statespace.csv`, `cheap.csv`, `condmedian.csv`, `daily.csv`, `equivalence.csv`, `cells.csv`, `run.json`.
  - `summary.txt`: the output of `scripts/summarize_main.py`.
- `sampling-1/`: input representation × target × H × L grid, with validation-selected baselines and error floors.
  - `sampling.csv`, `cells.csv`.
  - `summary.txt`: the output of `scripts/summarize_sampling.py`.
- `derived_probe_vs_strong.csv`, `derived_floors.csv`, `derived_like_for_like_best_mae.csv`: from `scripts/derived_tables.py`.
- `electrosense_noise_check.csv`: from `scripts/electrosense_noise_check.py`, run in the cluster head container.
- To reproduce the runs on the cluster:
  `python -m training.ray.level_diagnostics --output-dir runs/<name> [--part sampling]`, submitted with `ray job submit` as described in `training/ray/AGENTS.md`.
