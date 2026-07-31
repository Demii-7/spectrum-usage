# Validation Results By Representation

Lower is better. Values are validation metrics. Entries formatted as
`mean +/- SEM` are means over completed replication seeds `40, 41, 42, 43, 44`;
`SEM` is the sample standard deviation divided by `sqrt(n)`. `n/a` means that
five-seed replication is not complete. Rows are ordered from best to worst
mean validation dB MAE within each representation. `*` marks a search that is
still running. `provisional` marks a single seed-42 search result, not a
replication estimate.

## 1D Models

| Model | Previous historical val loss | Current val loss | Mean validation dB MAE | t+1 dB MAE | t+15 dB MAE | t+60 dB MAE | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResidualLinearAR1D |  | 0.254198 +/- 0.000187 | 0.98500 +/- 0.00073 | 0.90424 +/- 0.00049 | 0.96394 +/- 0.00104 | 1.08682 +/- 0.00072 | 5 |
| LinearAR1D |  | 0.253539 +/- 0.000288 | 0.98711 +/- 0.00075 | 0.90474 +/- 0.00017 | 0.96606 +/- 0.00039 | 1.09053 +/- 0.00186 | 5 |
| LookbackMean1D |  | 0.258704 (provisional) | 1.02057 +/- n/a | 0.96013 +/- n/a | 0.99189 +/- n/a | 1.10969 +/- n/a | 1 |

## 2D Models

| Model | Previous historical val loss | Current val loss | Mean validation dB MAE | t+1 dB MAE | t+15 dB MAE | t+60 dB MAE | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResidualLinearAR2D | 0.247884 | 0.269470 +/- 0.000207 | 0.97309 +/- 0.00073 | 0.88064 +/- 0.00043 | 0.95494 +/- 0.00077 | 1.08371 +/- 0.00127 | 5 |
| TemporalConvNet | 0.279404 | 0.268290 +/- 0.000146 | 0.97804 +/- 0.00046 | 0.88820 +/- 0.00054 | 0.95929 +/- 0.00069 | 1.08662 +/- 0.00123 | 5 |
| LinearAR2D |  | 0.267452 +/- 0.000254 | 0.97796 +/- 0.00041 | 0.88371 +/- 0.00016 | 0.95858 +/- 0.00028 | 1.09160 +/- 0.00083 | 5 |
| LookbackMean2D | 0.213887 | 0.274179 (provisional) | 1.00677 +/- n/a | 0.94301 +/- n/a | 0.97919 +/- n/a | 1.09810 +/- n/a | 1 |
| Autoformer-CSA | 0.527293 | 0.273838 +/- 0.000095 | 1.01981 +/- 0.00093 | 0.94943 +/- 0.00188 | 0.99388 +/- 0.00124 | 1.11612 +/- 0.00219 | 5 |
| ResidualVanillaLSTM | 0.286421 | 0.328229 +/- 0.022559 | 1.05243 +/- 0.01956 | 0.93973 +/- 0.00218 | 1.00996 +/- 0.00877 | 1.20761 +/- 0.05092 | 5 |
| LSTMAttn | 0.280021 | 0.272541 +/- 0.003054 | 1.11198 +/- 0.00352 | 0.98381 +/- 0.00434 | 1.09086 +/- 0.00276 | 1.26126 +/- 0.01004 | 5 |
| VanillaLSTM | 0.275254 | 0.285572 +/- 0.001680 | 1.17259 +/- 0.00510 | 1.00288 +/- 0.00852 | 1.14906 +/- 0.00372 | 1.36583 +/- 0.01111 | 5 |

## Test Set

Lower is better. These results use the completed replication campaigns at
`s3://spectrum-usage/spectrum-usage/ray/production-20260729-replicate-v2/`.
Values are mean +/- SEM over seeds `40, 41, 42, 43, 44`. Mean test error is
the unweighted mean of the t+1, t+15, and t+60 errors for each seed.
The deterministic LookbackMean baseline is evaluated once and has no SEM.

### 1D Test MAE

| Model | Mean test dB MAE | t+1 dB MAE | t+15 dB MAE | t+60 dB MAE | n |
|---|---:|---:|---:|---:|---:|
| ResidualLinearAR1D | 0.81870 +/- 0.00100 | 0.76749 +/- 0.00061 | 0.80682 +/- 0.00120 | 0.88179 +/- 0.00126 | 5 |
| LinearAR1D | 0.82136 +/- 0.00050 | 0.76799 +/- 0.00025 | 0.80929 +/- 0.00039 | 0.88679 +/- 0.00103 | 5 |
| LookbackMean1D | 0.84398 | 0.80506 | 0.82753 | 0.89936 | 1 |

### 1D Test RMSE

| Model | Mean test dB RMSE | t+1 dB RMSE | t+15 dB RMSE | t+60 dB RMSE | n |
|---|---:|---:|---:|---:|---:|
| LinearAR1D | 1.27689 +/- 0.00054 | 1.21390 +/- 0.00011 | 1.26481 +/- 0.00030 | 1.35195 +/- 0.00124 | 5 |
| ResidualLinearAR1D | 1.27861 +/- 0.00041 | 1.21542 +/- 0.00036 | 1.26666 +/- 0.00054 | 1.35375 +/- 0.00064 | 5 |
| LookbackMean1D | 1.29329 | 1.24874 | 1.27672 | 1.35441 | 1 |

### 2D Test MAE

| Model | Mean test dB MAE | t+1 dB MAE | t+15 dB MAE | t+60 dB MAE | n |
|---|---:|---:|---:|---:|---:|
| ResidualLinearAR2D | 0.85999 +/- 0.00092 | 0.78385 +/- 0.00049 | 0.84476 +/- 0.00091 | 0.95136 +/- 0.00163 | 5 |
| TemporalConvNet | 0.86836 +/- 0.00068 | 0.79355 +/- 0.00119 | 0.85252 +/- 0.00106 | 0.95902 +/- 0.00126 | 5 |
| LinearAR2D | 0.86916 +/- 0.00073 | 0.78740 +/- 0.00018 | 0.85162 +/- 0.00046 | 0.96846 +/- 0.00157 | 5 |
| LookbackMean2D | 0.88840 | 0.83619 | 0.86622 | 0.96280 | 1 |
| Autoformer-CSA | 0.90100 +/- 0.00092 | 0.84357 +/- 0.00219 | 0.88028 +/- 0.00126 | 0.97915 +/- 0.00159 | 5 |
| ResidualVanillaLSTM | 0.94117 +/- 0.02145 | 0.83788 +/- 0.00156 | 0.89991 +/- 0.01184 | 1.08571 +/- 0.05320 | 5 |
| LSTMAttn | 1.16500 +/- 0.01430 | 1.01543 +/- 0.00617 | 1.14835 +/- 0.01484 | 1.33122 +/- 0.02977 | 5 |
| VanillaLSTM | 1.22673 +/- 0.01038 | 1.03097 +/- 0.01631 | 1.18274 +/- 0.01104 | 1.46648 +/- 0.01372 | 5 |

### 2D Test RMSE

| Model | Mean test dB RMSE | t+1 dB RMSE | t+15 dB RMSE | t+60 dB RMSE | n |
|---|---:|---:|---:|---:|---:|
| LinearAR2D | 1.80799 +/- 0.00037 | 1.69999 +/- 0.00005 | 1.78581 +/- 0.00018 | 1.93815 +/- 0.00088 | 5 |
| ResidualLinearAR2D | 1.81062 +/- 0.00051 | 1.70178 +/- 0.00023 | 1.78860 +/- 0.00041 | 1.94148 +/- 0.00096 | 5 |
| TemporalConvNet | 1.82203 +/- 0.00114 | 1.71639 +/- 0.00152 | 1.79834 +/- 0.00122 | 1.95136 +/- 0.00122 | 5 |
| LookbackMean2D | 1.84211 | 1.76253 | 1.81126 | 1.95253 | 1 |
| Autoformer-CSA | 1.85356 +/- 0.00043 | 1.76678 +/- 0.00222 | 1.82306 +/- 0.00031 | 1.97086 +/- 0.00215 | 5 |
| ResidualVanillaLSTM | 1.90328 +/- 0.02262 | 1.75904 +/- 0.00360 | 1.84035 +/- 0.00765 | 2.11044 +/- 0.06322 | 5 |
| LSTMAttn | 2.32362 +/- 0.05040 | 2.03621 +/- 0.01520 | 2.30754 +/- 0.06426 | 2.62712 +/- 0.08897 | 5 |
| VanillaLSTM | 2.39836 +/- 0.02952 | 2.04462 +/- 0.03027 | 2.37451 +/- 0.04403 | 2.77596 +/- 0.03945 | 5 |

### 4D Test MAE

| Model | Mean test dB MAE | t+1 dB MAE | t+15 dB MAE | t+60 dB MAE | n |
|---|---:|---:|---:|---:|---:|
| ResidualLinearAR4D | 0.62183 +/- 0.00044 | 0.55162 +/- 0.00015 | 0.60675 +/- 0.00035 | 0.70713 +/- 0.00085 | 5 |
| LinearAR4D | 0.63975 +/- 0.00164 | 0.56360 +/- 0.00159 | 0.61926 +/- 0.00137 | 0.73639 +/- 0.00265 | 5 |
| ResidualConvLSTM | 0.65053 +/- n/a | 0.60021 +/- n/a | 0.62773 +/- n/a | 0.72365 +/- n/a | 2 * |
| ConvLSTM | 0.97493 +/- 0.00739 | 0.89763 +/- 0.00498 | 0.96353 +/- 0.00525 | 1.06365 +/- 0.01752 | 5 |
| DSwinLSTM-I |  |  |  |  | 0 |
| ConvLSTM-FM |  |  |  |  | 0 |

### 4D Test RMSE

| Model | Mean test dB RMSE | t+1 dB RMSE | t+15 dB RMSE | t+60 dB RMSE | n |
|---|---:|---:|---:|---:|---:|
| ResidualLinearAR4D | 1.28748 +/- 0.00027 | 1.18320 +/- 0.00016 | 1.26299 +/- 0.00026 | 1.41624 +/- 0.00054 | 5 |
| LinearAR4D | 1.29424 +/- 0.00108 | 1.19061 +/- 0.00133 | 1.26620 +/- 0.00064 | 1.42592 +/- 0.00176 | 5 |
| ResidualConvLSTM | 1.32270 +/- n/a | 1.24306 +/- n/a | 1.28848 +/- n/a | 1.43655 +/- n/a | 2 * |
| ConvLSTM | 1.76990 +/- 0.00586 | 1.63836 +/- 0.00678 | 1.77457 +/- 0.00499 | 1.89678 +/- 0.01764 | 5 |

## 4D Models

| Model | Previous historical val loss | Current val loss | Mean validation dB MAE | t+1 dB MAE | t+15 dB MAE | t+60 dB MAE | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResidualLinearAR4D | 0.266781 | 0.267402 +/- 0.000214 | 0.93042 +/- 0.00020 | 0.84933 +/- 0.00020 | 0.91790 +/- 0.00019 | 1.02402 +/- 0.00045 | 5 |
| LinearAR4D | 1.057217 | 0.260147 +/- 0.000669 | 0.93776 +/- 0.00128 | 0.86253 +/- 0.00187 | 0.92292 +/- 0.00114 | 1.02784 +/- 0.00139 | 5 |
| LookbackMean4D | 0.175999 | 0.272255 (provisional) | 0.96284 +/- n/a | 0.90874 +/- n/a | 0.94025 +/- n/a | 1.03954 +/- n/a | 1 |
| ResidualConvLSTM | 0.265573 | 0.272255 | 0.96308* +/- n/a | 0.90890* +/- n/a | 0.94044* +/- n/a | 1.03990* +/- n/a | 2 * |
| ConvLSTM | 0.552738 | 0.298986 | 1.24080 +/- 0.00323 | 1.19769 +/- 0.00495 | 1.24422 +/- 0.00315 | 1.28048 +/- 0.00876 | 5 |
| DSwinLSTM-I |  | 0.295602 | 1.20062 +/- n/a | 1.25006 +/- n/a | 1.17404 +/- n/a | 1.17776 +/- n/a | 0 |
| ConvLSTM-FM |  | 1.025627 (provisional; 1 error) | 2.47286 +/- n/a | 1.60842 +/- n/a | 2.92097 +/- n/a | 2.88918 +/- n/a | 0 |

The deterministic LookbackMean baselines are reported once and therefore have
no seed-based SEM. Rows with `*` are incomplete (not all five seeds finished).
ResidualConvLSTM seeds 42--44 failed: seed 42 hit the evaluation-directory
bug and seeds 43--44 failed on a corrupt map cache (`BadZipFile`); a targeted
retry is pending. The DSwinLSTM-I replication has not been submitted; its value
is the best HPO trial (50 epochs). ConvLSTM-FM completed with one trial error
and several divergent trials; its listed value is the best completed trial.

The previous TemporalConvNet search used joint-frequency models and is
excluded. Its replacement search uses only the independent-series model family.

## Daily-History Evaluation (STS-PredNet lp=2)

Lower is better. These results compare STS-PredNet (2 layers, 64 hidden, batch
size 2, lr 0.0002) against a matched Linear AR baseline, both using 60 recent
frames plus daily history at 1440 and 2880 minutes. All horizons use the same
target set (same-N) within each split. T6-only uses no T4 context.

### Daily-History Test MAE

| Model | Split | Mean test dB MAE | t+1 dB MAE | t+15 dB MAE | t+60 dB MAE | n |
|---|---|---:|---:|---:|---:|---:|
| Daily-history Linear AR | T6 only | 0.691 | 0.566 | 0.658 | 0.850 | 781 |
| Daily-history Linear AR | T4 validation | 0.740 | 0.637 | 0.716 | 0.868 | 1440 |
| Daily-history Linear AR | T4→T6 | 0.732 | 0.611 | 0.702 | 0.881 | 1533 |
| STS-PredNet | T6 only | 1.042 | 0.841 | 1.055 | 1.230 | 781 |
| STS-PredNet | T4 validation | 0.838 | 0.751 | 0.832 | 0.931 | 1440 |
| STS-PredNet | T4→T6 | 1.047 | 0.895 | 1.049 | 1.196 | 1533 |

### Daily-History Test RMSE

| Model | Split | Mean test dB RMSE | t+1 dB RMSE | t+15 dB RMSE | t+60 dB RMSE | n |
|---|---|---:|---:|---:|---:|---:|
| Daily-history Linear AR | T6 only | 1.310 | 1.185 | 1.268 | 1.478 | 781 |
| Daily-history Linear AR | T4 validation | 1.404 | 1.271 | 1.370 | 1.571 | 1440 |
| Daily-history Linear AR | T4→T6 | 1.364 | 1.226 | 1.325 | 1.542 | 1533 |
| STS-PredNet | T6 only | 1.785 | 1.472 | 1.816 | 2.066 | 781 |
| STS-PredNet | T4 validation | 1.537 | 1.389 | 1.527 | 1.693 | 1440 |
| STS-PredNet | T4→T6 | 1.786 | 1.571 | 1.791 | 1.994 | 1533 |

## Daily-History Evaluation (STS-PredNet lp=1)

Lower is better. These results compare STS-PredNet (2 layers, 64 hidden, batch
size 2, lr 0.0002) against a matched Linear AR baseline, both using 60 recent
frames plus one daily history frame at 1440 minutes. All horizons use the same
target set (same-N) within each split. T6-only uses no T4 context.

### lp=1 Daily-History Test MAE

| Model | Split | Mean test dB MAE | t+1 dB MAE | t+15 dB MAE | t+60 dB MAE | n |
|---|---|---:|---:|---:|---:|---:|
| Daily-history Linear AR | T6 only | 0.676 | 0.553 | 0.638 | 0.836 | 2221 |
| Daily-history Linear AR | T4→T6 | 0.690 | 0.570 | 0.655 | 0.847 | 2567 |
| Daily-history Linear AR | T4 validation | 0.749 | 0.633 | 0.715 | 0.898 | 1440 |
| STS-PredNet | T6 only | 0.976 | 0.793 | 0.922 | 1.214 | 2221 |
| STS-PredNet | T4→T6 | 0.986 | 0.806 | 0.927 | 1.227 | 2567 |
| STS-PredNet | T4 validation | 0.868 | 0.729 | 0.840 | 1.035 | 1440 |

### lp=1 Daily-History Test RMSE

| Model | Split | Mean test dB RMSE | t+1 dB RMSE | t+15 dB RMSE | t+60 dB RMSE | n |
|---|---|---:|---:|---:|---:|---:|
| Daily-history Linear AR | T6 only | 1.312 | 1.174 | 1.267 | 1.495 | 2221 |
| Daily-history Linear AR | T4→T6 | 1.330 | 1.190 | 1.286 | 1.513 | 2567 |
| Daily-history Linear AR | T4 validation | 1.422 | 1.267 | 1.377 | 1.622 | 1440 |
| STS-PredNet | T6 only | 1.747 | 1.476 | 1.691 | 2.073 | 2221 |
| STS-PredNet | T4→T6 | 1.749 | 1.491 | 1.685 | 2.072 | 2567 |
| STS-PredNet | T4 validation | 1.595 | 1.371 | 1.573 | 1.840 | 1440 |
