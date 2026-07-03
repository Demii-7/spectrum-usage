# Statistical Analysis Plan

This document describes the descriptive statistics we will compute for each spectrum trace. The goal is to identify the structure available to a predictor and to separate easy prediction cases from cases that require temporal, spectral, or spatial learning.

## Analysis Units

We will compute statistics at three levels.

1. **Trace level**: one row per dataset, site, band, and run. Trace-level statistics summarize coverage, missingness, and broad measurement conditions.
2. **Frequency-bin level**: one row per measured frequency bin. Bin-level statistics provide the input features for frequency segmentation.
3. **Frequency-segment level**: one row per detected contiguous frequency region. Segment-level statistics are the main unit for task characterization and paper figures.

A **frequency segment** is a contiguous set of frequency bins within one site-band trace that has similar power distribution, occupancy dynamics, and temporal predictability statistics. A segment is not necessarily one transmitter or one licensed allocation. It is a homogeneous prediction region: smaller than a swept measurement band, larger than a single frequency bin, and useful for describing what a prediction model has to learn.

## Workflow

The first analytic output is the per-bin feature table. Frequency segmentation depends on those per-bin features, so segmentation is not the first step.

The workflow is:

1. Put each trace on a documented time and frequency grid.
2. Preserve the original missing-value mask and create a forward-filled analysis series.
3. Compute trace-level missingness and gap diagnostics.
4. Compute per-bin descriptive features.
5. Run binary segmentation over frequency using standardized per-bin features.
6. Compute segment-level statistics and task-regime labels.
7. Compute spectral and spatial relationship statistics where the trace supports them.

## Trace-Level Statistics

For each trace, compute:

- dataset, site, run, band, start time, end time, duration
- frequency range and frequency resolution
- time resolution
- number of timestamps and frequency bins
- missing value rate, both overall and per frequency bin
- number of gaps and longest gap duration
- number of forward-filled samples and back-filled leading samples
- long-gap flag
- power range and quantiles over the whole trace
- number of detected frequency segments
- count of segments by task regime

Trace-level statistics support dataset summaries and help catch measurement problems before model evaluation.

## Preprocessing Assumptions

All lagged, burst-duration, and differenced statistics require a regular time grid. Before computing these statistics, resample each trace to a documented time resolution. For our PAWR traces, the default grid is one minute.

For descriptive statistics, create an analysis series by forward-filling missing power values within each frequency bin after placing the trace on the regular time grid. Preserve the original missing-value mask before filling. Use the forward-filled series for statistics that require continuity, including autocorrelation, first differences, entropy, burst duration, occupancy transition counts, and baseline-difficulty statistics.

Leading missing values may be back-filled only when needed to make the analysis series complete, and the number of back-filled samples must be reported. Do not silently fill long gaps. Record the longest gap, number of gaps, per-bin missingness, overall missingness, number of forward-filled samples, and number of back-filled leading samples in the trace-level output. If a gap exceeds the configured maximum fill length, mark the affected trace, bin, or segment with a `long_gap_flag`.

Forward-fill is a pragmatic choice for descriptive statistics because it preserves the time grid, but it can create artificial constant intervals. Long filled gaps can reduce variance, inflate autocorrelation, and distort burst durations. Missingness diagnostics should therefore be reported alongside all statistics computed from the filled series.

Frequency segmentation assumes ordered bins on a uniform frequency grid. If a source dataset uses nonuniform frequency spacing, first resample or aggregate it to a uniform grid appropriate for that dataset. Record the original resolution and the analysis resolution in `trace_summary.csv`.

Power values are not always calibrated across receivers or datasets. Trace-level outputs should include a calibration or gain-consistency flag. Occupancy-rate comparisons across sites require comparable receiver gain, antenna configuration, and preprocessing; otherwise, compare only within a site or use normalized/residual features that do not depend on absolute power.

## Frequency-Bin Features

For each frequency bin, compute robust features that can support segmentation and later diagnostics.

Distribution features:

- mean power
- 5th, 50th, and 95th percentile power
- interpercentile range: 95th percentile minus 5th percentile
- standard deviation and variance
- short-timescale standard deviation, computed as `std(diff(power)) / sqrt(2)`
- missing value rate before filling
- number of gaps and longest gap duration
- number of forward-filled samples and back-filled leading samples
- long-gap flag

Occupancy features:

- local noise-floor estimate
- occupancy threshold, such as local noise floor plus 3 dB
- occupancy rate
- transition rate between idle and occupied states
- burst count per hour
- mean occupied duration
- mean idle duration

Temporal features:

- lag-1 autocorrelation
- lag-5, lag-15, and lag-60 autocorrelation for minute-resolution traces
- lag-1440 autocorrelation for daily repeatability when the trace is long enough
- daily profile range, computed as the 95th minus 5th percentile of the mean minute-of-day profile
- weekday/weekend contrast when the trace spans enough days
- first-half versus second-half distribution distance as a stationarity check

Entropy features:

- binary occupancy entropy, `H(X_t)`
- first-order conditional entropy, `H(X_t | X_{t-1})`
- higher-order conditional entropy, `H(X_t | X_{t-1}, ..., X_{t-k})`, when enough data are available
- entropy-rate estimate using a documented estimator
- normalized entropy rate
- predictability upper bound derived from entropy, following the style of the entropy-based spectrum predictability literature

For the first implementation, compute entropy on binary occupancy states rather than quantized power levels. Estimate `H(X_t)` and first-order conditional entropy with plug-in counts plus a finite-sample correction. Higher-order conditional entropy should only be reported when each context has enough support; otherwise leave the value missing. If we use an entropy-rate estimator such as Lempel-Ziv complexity or context-tree weighting, record the estimator name, state alphabet, sample count, and minimum trace-length rule in the output metadata. Entropy-based predictability bounds are not comparable across segments unless the estimator, state alphabet, and sample length requirements are the same.

## Descriptive Artifacts

This analysis is descriptive. It characterizes the traces we have collected and produces summary tables and figures for understanding the spectrum prediction task. It does not define train, validation, or test splits.

Frequency segmentation should use the full trace by default. The resulting segment boundaries and task-regime labels support dataset characterization, paper figures, and qualitative discussion of spectrum behavior.

If later model-evaluation code uses these statistics to stratify scores, that code should define its own split-aware procedure. That concern belongs in the model-evaluation workflow, not in this descriptive analysis plan.

## Frequency Segmentation

We will group adjacent frequency bins into frequency segments using one-dimensional binary segmentation over frequency.

The segmentation procedure is:

1. Compute the frequency-bin feature table for a site-band trace.
2. Select segmentation features. The initial feature set should include median power, lower-tail power, upper-tail power, variance or standard deviation, short-timescale standard deviation, occupancy rate, lag-1440 autocorrelation, daily profile range, and entropy rate when available.
3. Robustly standardize each feature across frequency bins within the trace, using median and interquartile range where possible.
4. Run binary segmentation over the ordered frequency bins. At each step, choose the split that gives the largest reduction in within-segment feature variance.
5. Stop when the best split falls below a configured improvement threshold, the segment would become narrower than the minimum width, or the maximum number of segments is reached.
6. Recompute all characterization statistics over each detected segment.

The minimum segment width should be configurable. For 1 MHz bins, an initial default of 5 bins is reasonable for broad task characterization, but narrower segments may be useful for higher-resolution datasets such as Electrosense.

Binary segmentation is interpretable and already matches the kind of analysis we have started in `evaluation/scripts/segment_cc2_frequency_bands.py`. If later results show unstable boundaries, we can compare against PELT or bottom-up adjacent-bin agglomeration, but binary segmentation should be the first implementation.

Spectrum behavior can change during a trace, so a single set of frequency boundaries may hide time-varying structure. As a diagnostic, rerun binary segmentation on sliding windows or chronological chunks and report boundary stability. A simple boundary-stability score can compare segment boundaries across windows using overlap or Jaccard similarity after mapping each bin to a segment label. Low stability should contribute to the `unstable_or_shifted` label and should be visible in the trace-level summary.

## Segment-Level Statistics

For each frequency segment, compute the same distribution, occupancy, temporal, and entropy statistics used at the bin level, but aggregate over the full segment.

Segment identity:

- segment ID
- start frequency, end frequency, center frequency, and bandwidth
- number of bins
- parent dataset, site, band, and run

Distribution:

- mean power
- 5th, 50th, and 95th percentile power
- interpercentile range
- standard deviation and variance
- median per-bin standard deviation
- short-timescale standard deviation of the segment mean time series

Occupancy:

- segment-level occupancy threshold
- occupancy rate
- burst count per hour
- mean occupied and idle durations
- transition rate
- duty-cycle variance across hours or days

Temporal predictability:

- autocorrelation at short lags
- lag-1440 autocorrelation when valid
- daily profile range
- weekday/weekend contrast when valid
- trend over the trace, such as the slope of hourly median power
- stationarity score between early and late portions of the trace
- frequency-boundary stability across sliding time windows

Entropy and predictability:

- occupancy entropy
- conditional entropy
- entropy-rate estimate
- normalized entropy rate
- predictability upper bound

Baseline difficulty:

- persistence error
- historical mean error
- same-time-yesterday error when valid
- seasonal mean error when valid
- best simple baseline for the segment

Baseline difficulty statistics tell us whether a model beats the structure exposed by simple predictors, not only whether the segment is variable.

## Task-Regime Labels

Each segment should receive a task-regime label derived from its statistics. Initial labels:

- `noise_floor`: low mean power, low variance, low occupancy rate
- `constant_occupancy`: high occupancy rate and low variance
- `intermittent_occupancy`: high variance or frequent state transitions without strong daily repeatability
- `diurnal_pattern`: lag-1440 autocorrelation and daily profile range above configured thresholds
- `bursty_short_timescale`: high short-timescale variance or high transition rate
- `mixed_activity`: no single dominant structure
- `unstable_or_shifted`: strong nonstationarity across the trace

These labels should come from measured behavior, not from FCC or NTIA allocation tables. Allocation information can help interpret a segment after classification, but it should not define the task regime.

## Spectral Statistics

Some prediction models claim to learn relationships across frequency. We should measure those relationships explicitly.

Within each site-band trace, compute:

- within-segment pairwise frequency correlation, on raw and deseasonalized series
- adjacent-bin correlation, on raw and deseasonalized series
- adjacent-segment correlation, on raw and deseasonalized series
- spectral coherence width, defined as the frequency distance over which correlation remains above a threshold
- strongest positively correlated segment for each segment
- strongest negatively correlated segment for each segment
- fraction of frequency pairs that are positively correlated, negatively correlated, or near zero
- the same quantities for binary occupancy state, not only received power

Correlation, anti-correlation, and non-correlation across frequency should all be reported. Positive correlation can reflect a wideband transmitter, shared traffic load, receiver gain variation, common time-of-day structure, or a noise-floor artifact. Negative correlation can reflect activity moving between channels, frequency hopping, or scheduler behavior. These interpretations require caution, so spectral statistics should be paired with spectrogram inspection for representative cases.

Compute correlations on both raw series and residual series. The residual series should remove the mean daily profile when the trace is long enough, or a lower-order trend/seasonal component when it is not. Raw correlation measures shared behavior as observed by the receiver. Residual correlation reduces the effect of shared diurnal structure and better tests whether one frequency provides information beyond a common daily pattern.

Spectral claims in model evaluation should be tested by comparing against independent per-frequency models or ablations that remove cross-frequency inputs.

## Spatial Statistics

For simultaneous multi-receiver traces, compute spatial statistics at the segment level.

Spatial features:

- pairwise receiver time-series correlation for each segment, on raw and deseasonalized series
- mean, median, minimum, and maximum pairwise receiver time-series correlation
- Spearman correlation between receiver distance and pairwise receiver time-series similarity
- within-site temporal variance and between-site mean variance
- held-out receiver interpolation error when using IDW or another map-building method

For each frequency bin or segment, define two separate quantities. The first is cross-site temporal similarity:

```text
time_spearman_r(i, j, f) = SpearmanCorr_t(x_i(f, t), x_j(f, t))
```

Summarize this over all receiver pairs as `mean_pair_time_spearman_r` and `median_pair_time_spearman_r`. This tells us whether receivers observe similar time-varying behavior, without using their locations.

The second is distance-dependent spatial structure:

```text
spatial_spearman_r(f) = SpearmanCorr_pairs(distance(i, j), time_spearman_r(i, j, f))
```

A negative `spatial_spearman_r` means nearby receivers have more similar time series than distant receivers. This distinguishes shared temporal activity from spatial structure. A bin can have high variance and high cross-site temporal similarity without a distance trend.

These statistics distinguish spatial learning from interpolation artifacts. If a model predicts an interpolated map, evaluation should also withhold a receiver from interpolation and compare predictions against that receiver's actual measurements.

Spatial statistics should also carry the trace calibration or gain-consistency flag. For uncalibrated receivers, absolute-power differences may reflect hardware rather than propagation or spectrum use. In those cases, residual or rank-based similarity is safer than comparing raw occupancy rates across receivers.

The existing POWDER nine-site analysis in `evaluation/SPATIAL.md` gives reference cases for these metrics. The 648--655 MHz region has high temporal variation and strong negative distance-similarity statistics, with median `mean_pair_time_spearman_r` near 0.65 and median `spatial_spearman_r` near -0.64. The 2497--2595 MHz region has a wider shared time-varying pattern, with distance-dependent structure strongest in subregions. The 769--775 MHz public-safety region is a contrast case: it has high variance in several bins, but median `spatial_spearman_r` near -0.01. These examples show why variance alone is not a spatial-structure metric.

The current script for recomputing these bin-level statistics is:

```bash
python3 evaluation/scripts/compute_spatial_correlation.py --write-pairs
```

## Outputs

The analysis should produce machine-readable tables and a small set of diagnostic figures.

Recommended tables:

- `trace_summary.csv`
- `frequency_bin_features.csv`
- `frequency_segments.csv`
- `segment_statistics.csv`
- `segment_spectral_correlation.csv`
- `segment_spatial_correlation.csv`

Recommended figures:

- spectrogram with segment boundaries overlaid
- per-bin feature profiles across frequency with segment boundaries
- count of segments by task regime
- temporal examples for each task regime
- spectral correlation heatmap for selected traces
- receiver-distance versus correlation plots for simultaneous multi-receiver traces

## Relation to Model Evaluation

The statistical analysis should inform evaluation design, even though this document does not define train/test splits.

- Report model performance by task regime, not only as one aggregate metric.
- Report performance by prediction horizon and task regime together.
- Compare deep models against simple baselines within each segment class.
- Test temporal generalization on held-out chronological periods.
- Test spectral claims with per-frequency baselines and cross-frequency ablations.
- Test spatial claims with held-out receiver evaluation.
- If model-evaluation code uses segment labels for scoring, implement the split-aware version there.

This workflow makes the prediction task explicit before model training. It lets us say whether a model succeeds on stable spectrum, intermittent activity, diurnal structure, spatially correlated regions, or frequency regions with exploitable cross-channel relationships.
