# Spatial Correlation in POWDER Spectrum Data

The POWDER nine-site dataset shows spatial structure in the 648-655 MHz and 2497-2595 MHz regions. These bins vary over time and the temporal patterns depend on sensor distance.

This note uses the common POWDER time range from `evaluation/powder/README.md`:

```text
2026-06-30T19:49:00Z .. 2026-07-02T15:12:00Z
```

The analysis uses all nine POWDER sites in `evaluation/map_settings/powder_20260628_20260630_9node_map_set.json`.

## Metrics

For each frequency bin, the script compares every pair of sites.

First, it computes the Spearman correlation between the two power time series:

```text
time_spearman_r(i, j, f) = SpearmanCorr_t(x_i(f, t), x_j(f, t))
```

Then it computes the relationship between site distance and pairwise time-series similarity:

```text
spatial_spearman_r(f) = SpearmanCorr_pairs(distance(i, j), time_spearman_r(i, j, f))
```

A negative `spatial_spearman_r` means nearby sites have more similar time series than distant sites. If you want a positive score for spatial structure, use `-spatial_spearman_r`.

The analysis also computes total variance for each bin across all sites and timestamps. High total variance means the bin changes over time, across space, or both.

## POWDER Finding

The 648-655 MHz region has both temporal variation and distance-dependent spatial structure. Treating the region as the one-MHz bin centers from 648.5 MHz through 655.5 MHz, the region has these per-bin summary statistics:

```text
total_variance_db2:          35.3 .. 50.3, median 42.6
mean_pair_time_spearman_r:    0.62 .. 0.67, median 0.65
spatial_spearman_r:          -0.68 .. -0.60, median -0.64
```

This means the full region, not only isolated bins, has time-varying power and a negative distance-similarity relationship.

Several bins in the region also pass this stricter filter:

```text
total_variance_db2 > 40
spatial_spearman_r < -0.6
```

Selected bins:

```text
649.5 MHz
650.5 MHz
652.5 MHz
655.5 MHz
```

These selected bins are representative examples from the region. They have pairwise site time-series correlations around 0.64 on average, while the distance-similarity Spearman statistic is below -0.6. The full 648-655 MHz region has the same pattern with a median distance-similarity Spearman value near -0.64.

The 2497-2595 MHz region in the `2400_2600` band also shows a shared time-varying pattern across the nine sites. Visual review of the spectrograms supports treating this as one wider band:

```text
2497.0 MHz .. 2595.0 MHz
```

Within this interval, the bin centers from 2497.5 MHz through 2594.5 MHz have total variance between about 20.0 and 31.5 dB². The mean pairwise time-series Spearman correlation ranges from about 0.59 to 0.66, with a median near 0.62. That supports a temporal-correlation interpretation across the whole band.

The distance relationship is strongest in subregions of the band. Across the 2497-2595 MHz interval, 82 of 98 one-MHz bins have `spatial_spearman_r < -0.35`, and 59 bins have `spatial_spearman_r < -0.4`. The strictest cluster appears around 2497-2525 MHz, where many bins have both high variance and negative distance-similarity Spearman values. The higher-frequency portion still has visible temporal structure, but the distance trend weakens in some bins.

The 769-775 MHz public-safety region provides a contrast case. Treating the region as bin centers from 769.5 MHz through 774.5 MHz, the region has high variance in most bins but little distance-dependent structure:

```text
total_variance_db2:           0.1 .. 93.6, median 80.5
mean_pair_time_spearman_r:    0.19 .. 0.42, median 0.36
spatial_spearman_r:          -0.10 .. 0.13, median -0.01
```

This band has visible activity, including the highest-variance `600_800` bin at 772.5 MHz, but the pairwise correlations do not decrease with distance. It is a useful example of a band with power variation but no spatial relationship under this metric.

Per-bin statistics for the 769-775 MHz region:

| Frequency (MHz) | Total Variance (dB²) | Mean Pair Time Spearman r | Spatial Spearman r |
| ---: | ---: | ---: | ---: |
| 769.5 | 40.35 | 0.19 | 0.09 |
| 770.5 | 0.15 | 0.31 | -0.09 |
| 771.5 | 80.98 | 0.36 | 0.07 |
| 772.5 | 93.57 | 0.35 | -0.10 |
| 773.5 | 87.29 | 0.36 | -0.10 |
| 774.5 | 79.97 | 0.42 | 0.13 |

## Evidence

The analysis separates temporal correlation from spatial structure.

Temporal correlation asks whether sites see similar time behavior in a frequency bin, without using site locations. The `mean_pair_time_spearman_r` column measures this by averaging the Spearman time-series correlation across all site pairs. The full 648-655 MHz region has median mean pairwise time correlation near 0.65, and the 2497-2595 MHz band has a median near 0.62, so the sites share time-varying activity in both regions.

Spatial structure asks whether those pairwise time correlations depend on distance. The `spatial_spearman_r` column measures the relationship between physical site distance and pairwise time-series similarity. The full 648-655 MHz region has `spatial_spearman_r` values from about -0.68 to -0.60, so nearby sites have more similar time series than distant sites throughout the region. The 2497-2595 MHz band has a broader and weaker distance relationship: most bins have negative distance-similarity Spearman values, but some bins near the upper part of the band fall closer to zero.

These two results argue against a purely temporal explanation. If all sites saw the same time-varying signal with no distance-dependent differences in SNR or local interference, `mean_pair_time_spearman_r` would be high while `spatial_spearman_r` would be near zero or undefined. In the full 648-655 MHz region, the pairwise correlations are high and they decrease with distance, which supports a spatial relationship beyond shared temporal activity. In the 2497-2595 MHz region, the evidence supports a wide temporally correlated band with distance-dependent structure concentrated in subregions.

The variance filter adds a third check. The full 648-655 MHz region has total variance above about 35.3 dB², and the 2497-2595 MHz bins have total variance above about 20.0 dB². The correlations come from bins with visible power variation rather than flat noise-floor behavior.

The 769-775 MHz region shows why variance alone is not enough. Its median total variance is about 80.5 dB², but its median `spatial_spearman_r` is about -0.01. Strong power variation can come from activity that does not follow a distance-decay pattern across the nine POWDER sites.

## Artifacts

The per-frequency spatial correlation table is here:

```text
evaluation/results/spatial_correlation_refreshed/powder_20260628_20260630_9node_common_spatial_correlation_by_frequency.csv
```

The variance versus spatial correlation plot is here:

```text
plots/real_world_spectrum/powder_spatial_correlation_vs_total_variance_refreshed.png
```

The selected 600-800 MHz spectrogram grids are here:

```text
plots/real_world_spectrum/selected_spatial_bins_600_800/
```

The `652.5 MHz` pairwise distance-versus-correlation plot is here:

```text
plots/real_world_spectrum/selected_spatial_bins_600_800/powder_652p5mhz_pairwise_distance_vs_correlation_refreshed.png
```

That plot shows the spatial relationship more directly than the stacked time-series plot. The stacked traces show the shared on/off temporal pattern, but pairwise correlations can look similar by eye. The distance-versus-correlation scatter plots all 36 site pairs and shows whether time-series similarity decreases with physical distance. At 652.5 MHz, the all-pairs distance-similarity Spearman value is about -0.61, while the Humanities-only reference slice is about -0.91.

The `772.5 MHz` pairwise distance-versus-correlation plot is here:

```text
plots/real_world_spectrum/selected_spatial_bins_600_800/powder_772p5mhz_pairwise_distance_vs_correlation_refreshed.png
```

At 772.5 MHz, the total variance is about 93.6 dB², but the all-pairs distance-similarity Spearman value is about -0.10. This plot is the clearest counterexample in the `600_800` band: strong activity does not imply spatial correlation.

The selected 2400-2600 MHz spectrogram grids and wider 2497-2595 MHz context plots are here:

```text
plots/real_world_spectrum/selected_spatial_bins_2400_2600/
```

The spectrograms let you inspect each selected bin across all nine sites. They are useful because the spatial-correlation statistic captures a distance trend across site pairs, not a visual claim that every site has the same spectrogram.

## Script

Use this script to recompute the per-bin statistics:

```bash
python3 evaluation/scripts/compute_spatial_correlation.py --write-pairs
```

The script name and arguments are dataset-neutral. POWDER is the current default because the existing settings and map-set JSON files describe the POWDER data layout.
