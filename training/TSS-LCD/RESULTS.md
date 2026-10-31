# TSS-LCD reproduction results

Generated from `training/TSS-LCD/repro/results/*/metrics.json`. All values dBm, t+1 only.
Narrative and chronology: `EXPERIMENTS.md`. Paper audit: `MODEL_CARD.md`.

Authors' published Table III: MSE 0.1541, RMSE 0.3926, MAE 0.2486, MAPE 0.2075%.

| arm | impl | split | smoothing | MSE | persistence | lookback mean | AE ceiling | model/persistence |
|---|---|---|---|---|---|---|---|---|
| `abl_temfe` | theirs | random | on (centred) | **0.2146** | 0.00677 | 0.5984 | - | 31.7x |
| `pf_chrono` | ours | chrono | - | **38.9486** | 4.14122 | 2.8603 | 1.4400 | 9.4x |
| `pf_chrono_s1` | ours | chrono | off | **7.3379** | 4.14122 | 2.8603 | 1.4187 | 1.8x |
| `pf_chrono_s2` | ours | chrono | off | **17.7880** | 4.14122 | 2.8603 | 1.6936 | 4.3x |
| `pf_chrono_smooth` | ours | chrono | on | **5.4458** | 0.00619 | 0.4683 | 0.7607 | 880.2x |
| `pf_random` | ours | random | - | **18.4655** | 4.24153 | 3.0750 | 1.6642 | 4.4x |
| `pf_random_s1` | ours | random | off | **10.2182** | 4.08177 | 2.9647 | 1.7025 | 2.5x |
| `pf_random_s2` | ours | random | off | **12.9849** | 4.28563 | 3.0682 | 1.6409 | 3.0x |
| `pf_random_smooth` | ours | random | on | **24.3888** | 0.00676 | 0.5997 | 0.3754 | 3609.6x |
| `theirs_random` | theirs | random | on (centred) | **0.1548** | 0.00677 | 0.5984 | - | 22.9x |

## Table III reproduction (`theirs_random`)

| metric | paper | ours | difference |
|---|---|---|---|
| mse | 0.1541 | 0.15477 | +0.43% |
| rmse | 0.3926 | 0.39340 | +0.20% |
| mae | 0.2486 | 0.24536 | -1.30% |
| mape_pct | 0.2075 | 0.20770 | +0.10% |

## Table IV ablations (their code, leaky random-split configuration)

| ablation | paper MSE | ours | ratio |
|---|---|---|---|
| w/o TemFE | 0.3027 | 0.2146 | 0.709 |
| w/o SpeFE | 0.2910 | not run | - |
| w/o SpaFE | 0.2861 | not run | - |
| w/o FFM | 0.3139 | not run | - |

Note: the paper text calls w/o TemFE the most severe ablation, but its own Table IV
ranks w/o FFM worse (0.3139 > 0.3027). The prose contradicts the table.

## Seed spread (our paper-faithful model, unsmoothed)

- chronological: [38.95, 7.34, 17.79] - spread 5.31x
- random pooled: [18.47, 10.22, 12.98] - spread 1.81x

The seed spread exceeds the chronological-vs-random difference (Welch p = 0.51),
so single-run comparisons in this project are not reliable.
