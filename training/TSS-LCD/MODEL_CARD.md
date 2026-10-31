# Model Card — TSS-LCD

Companion to `training/TSS-LCD/README.md` (build, tensor shapes, layer-by-layer architecture) and
`TSSLCD_REPRODUCTION.md` (how the upstream repo was obtained and run). This card is the
paper-and-reproduction audit. Source tags: `[P:n]` = PDF page n, `[C:file:line]` = their released code,
`[O:file]` = our code, `[M]` = measured by us, `[?]` = unknown.

```yaml
model: tss_lcd
display_name: TSS-LCD
paper_cite: "S. Cheng, X. Li, X. Lin, H. Ding, Y. Sun, 'TSS-LCD: A Temporal-Spectral-Spatial-Guided Latent Conditional Diffusion Model for Spectrum Prediction', IEEE Trans. Cogn. Commun. Netw., vol. 12, 2026"
doi: 10.1109/TCCN.2026.3683213
zotero_key: 3FYBL4KU          # PDF at Zotero/storage/VUFP7PZM
year: 2026
venue: IEEE TCCN
benchmarked: false            # implemented in our repo, not in the manuscript's tables
representation: 2d
residual_variant: false
params: 65504442              # [M] their released code, config defaults: 62.3M context AE + 2.68M future AE + 0.53M diffusion
code_available: full          # spectrum-specific, authors' own
data_available: full          # train+test+source RECOVERED from commit d7f1e51263 (deleted from HEAD)
checkpoints_available: false  # never committed in any of the 14 commits
license: none-declared        # no LICENSE file in the repo
reproducible: partial
repro_blocker: "released predictions do not match the paper's own Table III; no checkpoints; GPU node down"
compares_vs_trivial: false
beats_persistence: false      # [M] their own published predictions lose to persistence by 60x in MSE
released_artifacts_match_paper: false   # 1.59x worse MAE than Table III
leakage_found: confirmed      # centred savgol(51) applied before windowing
```

## 0. At a glance

| | |
|---|---|
| Paper | TSS-guided latent conditional diffusion; two stages (TSS-CC condition, LCD-SP latent diffusion) |
| Code | Full and spectrum-specific: `github.com/Xlab2024/TSS-LCD`, pushed 2025-12-03, 8 Python files |
| Data | AERPAW CC1/CC2/LW1. **Complete**: training split, test split and the 6839-row source CSV all recovered from deleted commit `d7f1e51263` |
| Reproducible | **partial** — architecture, full data and the exact preprocessing chain yes (ground truth regenerated to 1.7e-5 dB); checkpoints and Table III no |
| Beats a trivial reference | **No.** Their published predictions score MSE 0.3866 dB²; last-value persistence on their own ground truth scores 0.0064 dB² `[M]` |
| Headline claim | "consistently outperforms all baselines across all metrics" `[P:12]` |
| Our verdict | Claim holds *within the baseline set they chose*; the set contains no trivial reference, and every row of Table III is beaten by persistence by ≥24x in MSE |
| Capacity | 65.5M parameters against 3,710 training windows — ~17,700 parameters per training sample `[M]` |

## 1. Task as the paper formulates it

- **Lookback / horizon**: T1 = 50 steps in, T2 = 10 steps out; at 1-min sampling, 50 min → 10 min `[P:10]`
- **Sampling interval**: 1 minute `[P:10]`
- **Multi-step, direct** — the model emits all 10 steps in one shot from a single latent.
  **But `test.py` keeps only `fut[:, 0, :]` and `pred[:, 0, :]`** `[C:test.py]`, so every released number and
  every figure is **t+1 only**. The 10-step horizon is trained but not reported.
- **Quantity predicted**: RSS in dBm, min-max normalised to [0,1] per frequency, then Savitzky-Golay
  smoothed; inverse-transformed to dBm before writing results.
- **Spatial scope**: 3 fixed nodes x 250 frequency bins, flattened to D = 750 columns.
- **Our benchmark's regime**: 60-min lookback, horizons 1/15/60 min, per-bin z-score, no smoothing.
  Matches on sampling interval and roughly on lookback; **differs on target definition** (we predict the
  measurement, they predict a smoothed measurement), which is the dominant difference — see §3.

## 2. Dataset used in the paper

| Field | Value | Source |
|---|---|---|
| Testbed | AERPAW | `[P:10]` |
| Nodes | 3 fixed: CC1, CC2, LW1 | `[P:10]` |
| Location | North Carolina State University campus | `[P:10]` |
| Collection hardware | USRP B205mini, UHD-based monitoring software | `[P:10]` |
| Frequency range | 85–335 MHz | `[P:10]` |
| Bin width / count | 1 MHz, F = 250 bins | `[P:10]` |
| Time span | one full week, continuous | `[P:10]` |
| Sampling interval | 1 min | `[P:10]` |
| Time points per sensor | 60 x 24 x 7 = 10,080 | `[P:10]` |
| Tensor shape | 10,080 x 250 x 3 → flattened 10,080 x 750 | `[P:10]` |
| Sliding windows | 10,021 samples (window 60, stride 1) | `[P:10]` |
| Train/test split | 7:3, no validation set named | `[P:10]` |
| Split chronological? | **not stated** — "divided into training and test sets using a 7:3 split ratio" | `[P:10]` `[?]` |
| Public? | Yes — `aerpaw.org/dataset/february-2022-cc1-cc2-lw1-spectrum-measurements/`; also Dryad `10.5061/dryad.hmgqnk9zn` | `[P:10]` |

**Collection procedure caveats.** The paper gives the sensor model and cadence but not antenna type, gain
calibration, or whether dBm is absolute or uncalibrated receiver power; not stated whether the three nodes
are time-synchronised sample-for-sample, which the 750-column flattening assumes. Dead/clipped bins are not
discussed. `[?]`

### Recovered data (commit `d7f1e51263`, 2025-12-02, "split dataset and divide the training part and the test part")

The training split was deleted from `HEAD` but survives in history. All three files were recovered `[M]`:

| File | Size | Shape | Role |
|---|---|---|---|
| `dataset/merged_power_data_sub6GHz_avg_per_minute.csv` | 56.41 MB | 6839 x 750 | the source series |
| `dataset/trainDataset.csv` | 31.06 MB | 3761 x 750 | = source rows 0–3760, verified byte-exact `[M]` |
| `dataset/testDataset.csv` | 25.36 MB | 3079 x 750 | rows 1.. = source rows 3761–6838, verified byte-exact `[M]` |

**The actual split is chronological, 55:45 — not the 7:3 the paper states**, and the source has **6839 rows,
not the 10,080 the paper claims** (6839 min = 4.75 days, not "one full week"). There is no overlap between
the splits once each file's corrupt first row is dropped, so the split is methodologically clean; it is
simply not the split described in the paper.

Both `trainDataset.csv` and `testDataset.csv` carry the malformed value `-134.20303.1` at **row 0,
column 672** `[M]` — the same corruption in both, absent from the source file. Because their loader consumes
row 0 as a header, the corrupt row is silently discarded in both cases, so it never reaches the model.
Effective data: train = source rows 1–3760 (3760), test = source rows 3761–6838 (3078).

**Their source CSV is not our AERPAW CSV, despite the identical filename.** Both are 6839 x 750 at 1-min
cadence — the same campaign and the same averaging window — but the values differ: mean −2.16 dB,
sd 3.33 dB, range −17.05 … +19.14 dB, with the difference concentrated per frequency bin (per-column mean
difference spans −15.8 … +16.3 dB) `[M]`. It is **not** a column permutation (sorted column means differ by
up to 5.32 dB) and **not** a row shift (every offset tested leaves mean |diff| ≈ 2.77 dB). So the two are
different derivations — different band selection, calibration or dB conversion — of the same AERPAW
measurements. Both are staged side by side under `training/data/tsslcd_original/` with a `PROVENANCE.txt`
warning not to interchange them. This supersedes the earlier "closest row differs by 14.9 dB" note, which
compared our file against their *test slice* rather than their source.

`dataset/data.csv` (1.42 MB) is a **USDCNY foreign-exchange series** (`code,date,open,low,close,high`,
26,277 rows from 2019) — an unrelated dataset left in a spectrum-prediction repository.

**What ships in the repo at HEAD** (not the same as the above):

| File | Shape | Note |
|---|---|---|
| `dataset/testDataset.csv` | 3079 lines x 750, headerless, dBm | one malformed cell, row 0 col 672 = `-134.20303.1` (two decimal points) `[M]` |
| `dataset/trainDataset.csv` | **absent** | `config.py` defaults to this path; README tells the user to supply it |
| `results/gt_mat.csv` | header + 3018 x 750 | pandas-written header row `0,1,…,749`; read it with `header=0` or the first row parses as +749 dBm `[M]` |
| `results/pred_matours.csv` | header + 3018 x 750 | the authors' published predictions |

Released test values span −136.42 … −108.24 dBm `[M]` — a 28 dB window, i.e. a quiet band with no strong
emitters, which is relevant to how easy the forecasting task is.

## 3. Data transformations

Chain, in order, from CSV to model input `[C:DataSetPrepare.py]`:

1. `pd.read_csv(path)` **without `header=None`** — row 0 of a headerless file is consumed as column names.
   3079 → 3078 usable rows. `[C:data_utils.py]`
2. Per-frequency min-max to [0,1], `MinMaxScalerCustom` `[C:DataSetPrepare.py:24]`.
3. `savgol_filter(window_length=51, polyorder=2, axis=0, mode='interp')` along time `[C:DataSetPrepare.py:50]`.
4. Stride-1 windowing into (context 50, future 10) by `MultiBandDataset` `[C:DataSetPrepare.py:65]`.
5. Random zeroing of context entries, `mask_ratio=0.1` `[C:DataSetPrepare.py:68,85]`.

- **Normalisation statistics fit on**: the file currently being loaded. `get_dataloader` constructs a fresh
  `MinMaxScalerCustom` per call `[C:DataSetPrepare.py:47]`, so the test file is normalised by **test
  statistics**, and train and test live on different scales.
- **Scaling granularity**: per-frequency column, per-file.
- **Smoothing applied before windowing**: yes — step 3 precedes step 4.
- **Leakage audit — CONFIRMED, two findings.**
  1. *Target smoothing with future samples.* A centred 51-sample Savitzky-Golay filter at index *i*
     consumes *i* ± 25. Applied before any split or windowing, so (a) the forecast target is a smoothed
     quantity rather than the measurement, and (b) the 50-step context already contains information from up
     to 25 steps **after** the forecast origin. This is the single most consequential finding about this
     paper: it is why consecutive ground-truth rows differ by only ~0.05 dB `[M]`, and therefore why
     persistence is extraordinarily strong on their released ground truth.
  2. *Test-set statistics.* Per-file min-max means the normalisation constants are derived from the
     evaluation data.
- **Metric space**: dBm. Table III's MAPE of 0.2075% against a mean level of −125.6 dBm implies an absolute
  error of ~0.26 dB, consistent with its MAE of 0.2486 `[M]`, so the table is in physical units and is
  directly comparable to our measurements. `test.py` does inverse-transform before writing.
  Note MAPE is computed on dBm, a logarithmic quantity with an arbitrary zero: shifting the reference level
  changes MAPE while leaving the physical error untouched. The MAPE column is not a meaningful relative error.

## 4. What the model needs besides the spectrum

| Requirement | Needed? | How we supply it | Risk |
|---|---|---|---|
| Time-of-day / calendar encoding | No | — | — |
| Sensor coordinates / geometry | No (implicit in column order) | column ordering convention | silent if column order differs |
| Observation mask | Yes — zeroed entries at train *and* test | our forecaster applies none | changes the input distribution |
| Neighbouring-sensor input | Yes — all 3 nodes enter one tensor | 750-column merged CSV | single-site data cannot be used |
| **Fixed number of sites** | **Yes — hardcoded 3** | `freq_pool = nn.Linear(seq_len * input_dim // 3, input_dim)` `[C:Context2CondNew.py:261]`, `loc_out.view(B, T * D // 3)` `[C:...:330]` | **will not accept our 6-site POWDER layout without surgery** |
| Fixed bin count | Effectively yes (D = 750) | — | D must be divisible by 3 |
| Pretrained weights | No | — | — |
| Per-dataset tuning | Not stated | — | `[?]` |
| External libraries | torch, numpy, pandas, scipy, matplotlib | all in `spectrum-smoke:latest` | none |

## 5. Architecture and capacity

Two stages. **TSS-CC** builds a condition from the context: a time branch (positional encoding +
transformer over the 50 steps), a frequency branch (per-location transformer over frequency tokens, pooled,
then a 3-token location-level transformer), and cross-attention fusion, projected to a latent.
**LCD-SP** trains a future autoencoder, then an epsilon-prediction DDPM in that latent space conditioned on
the TSS latent; sampling runs the full reverse chain and decodes. Layer detail in `training/TSS-LCD/README.md`.

- **Parameters, their released code at config defaults** `[M]`:

| Component | Parameters |
|---|---|
| `ContextTransformerAE` | 62,300,524 |
| `FutureAutoencoder` | 2,676,750 |
| `DiffusionModel` (noise net) | 527,168 |
| **Total** | **65,504,442** |

- Parameters, paper: not reported `[?]`. No parameter count, FLOPs or model-size table appears.
- **Capacity ratio: 65.5M parameters / 3,710 training windows = ~17,700 parameters per training sample.**
  95% of the model is the context autoencoder, and `freq_pool = nn.Linear(50*250, 750)` alone is 9.4M
  parameters. The training set is 3,760 rows of 750 columns = 2.8M scalars, so the model has **23x more
  parameters than there are numbers in its training data** `[M]`.

**Dead code.** `Context2CondNew.py` defines `ContextTransformerAE` **twice**: a fully commented-out stub at
line 14 and the live implementation at line 224. The live one wins at import. A reader auditing the first
definition is reading code that never runs.

## 6. Hyperparameters: declared vs actual

Paper Table II `[P:10]` against the released code. **Nine of fifteen rows disagree.**

| Parameter | Paper Table II | Their code | We use | Note |
|---|---|---|---|---|
| Historical length T1 | 50 | 50 `[C:config.py]` | 50 | agrees |
| Prediction horizon T2 | 10 | 10 | 10 | agrees |
| Diffusion steps N | 1000 | 1000 `[C:config.py]` | 1000 | `DataSetPrepare.CFG` says 2000 — a third value |
| Number of nodes L | 3 | hardcoded `//3` | configurable | agrees in value, not in flexibility |
| Frequency bins F | 250 | D = 750 = 3 x 250 | configurable | agrees |
| Transformer layers | 2 | 2 | 2 | agrees |
| **Attention heads** | **4** | **5** `[C:Context2CondNew.py:236]` | 5 | **4 is impossible**: `assert dim % num_heads == 0` `[C:...:82]` and 750 % 4 ≠ 0. The published value cannot have produced the published results |
| Transformer hidden d | 1024 | 1024 | 1024 | agrees |
| Dropout | 0.1 | 0.1 | 0.1 | agrees |
| **Latent dimension** | **16** | **32** `[C:config.py]` (class default 16) | 32 | config and class default disagree with each other |
| LSD/LSE hidden dim | 1024 | 256 `[C:F2Cnet.py]` | 256 | |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | agrees |
| **Batch size** | **32** | **256** `[C:config.py]` | 256 | `CFG` says 128 — a third value |
| **Training epochs** | **500** | **300 + 300 + 2000** | same | three staged runs, not one 500-epoch run |
| Optimizer | Adam | Adam | Adam | agrees |
| **Missing rate** | **25%** `[P:10]` | **0.1** `[C:DataSetPrepare.py:68]`, comment on line 84 says 25% | — | code, comment and paper form a three-way disagreement |
| **Early stopping** | "employed based on validation loss" `[P:10]` | **absent** `[M]` | — | no `val`, `early` or `patience` token anywhere in `train.py` |
| **Validation set** | implied by early stopping | **absent** | — | `train.py` has no split at all; it trains on the entire CSV it is given |
| **7:3 train/test split** | stated `[P:10]` | **absent** | — | no split code; train and test are separate files |

## 7. Claims, and how they are tested

### Claim 1: "consistently outperforms all baselines across all metrics" `[P:12]`
- **Evidence**: Table III `[P:12]` — TSS-LCD MSE 0.1541 / RMSE 0.3926 / MAE 0.2486 / MAPE 0.2075%, versus
  STS-PredNet 0.9227, CSMA 0.6441, ConvLSTM 1.4065, DSwinLSTM 1.4416, 3D-SwinSTB 3.9084 (MSE).
- **Baselines**: ConvLSTM [39], CSMA [41], STS-PredNet [43], DSwinLSTM [40], 3D-SwinSTB [26].
- **Trivial reference included?** **No.** No persistence, no lookback mean, no seasonal naive, no ARIMA
  run (ARIMA is discussed in related work but not evaluated). The word "persistence" does not occur in the paper `[M]`.
- **Metrics**: MSE/RMSE/MAE/MAPE in dBm, t+1 only.
- **Seeds / repeats / error bars**: none. Every table is a single run with no dispersion `[M]`.
- **Apples-to-apples?** Unverifiable. Baseline hyperparameters, tuning budget and whether baselines received
  the same savgol-smoothed target are not stated. If baselines were trained on unsmoothed data, the
  comparison is not like-for-like.
- **Our assessment**: the ranking within their chosen set is plausible, but the claim is **not informative**
  — last-value persistence scores MSE 0.0064 on their own released ground truth `[M]`, beating the best row
  of Table III by 24x and the worst by 610x. A baseline set that excludes the trivial predictor cannot
  establish that a model has learned anything beyond signal level.

### Claim 2: robustness to incomplete observations — "lowest MSE under all levels of data incompleteness" `[P:13]`
- **Evidence**: Figs. 11–12, missing rates 15%→35%, random and contiguous patterns.
- **Trivial reference?** No.
- **Our assessment**: the comparison is internally consistent, but the released code masks **10%**, not the
  15–35% swept in the figures, so the figures cannot be regenerated from the repo as shipped. Masking is
  also applied at test time, so the reported numbers are on degraded inputs for every method.

### Claim 3: each component is necessary (ablation) `[P:13]`
- **Evidence**: Table IV — removing TemFE 0.1541→0.3027, SpeFE →0.2910, SpaFE →0.2861, FFM →0.3139,
  latent diffusion →0.4544; diffusion in original space 0.3831.
- **Our assessment**: the ordering is coherent and the margins are large relative to each other. Single runs,
  no seeds. Note every ablated variant still beats three of the five baselines in Table III, and **all of them
  lose to persistence**, so the ablation establishes relative component value, not absolute merit.

### Claim 4: long-horizon advantage — horizons 10→50 steps `[P:13]`
- **Evidence**: text references Fig. 12 for the horizon sweep, but Fig. 12 is captioned as the
  continuous-missing MSE figure. The horizon-sweep figure is either mislabelled or missing `[P:13]`.
- **Our assessment**: **cannot be checked** as published. Also in tension with `test.py`, which reports t+1 only.

## 8. Errors found in the paper or code

| # | Where | What | Severity | Evidence |
|---|---|---|---|---|
| 1 | `DataSetPrepare.py:50` | Centred savgol(51,2) applied before windowing: target is smoothed, and context sees up to 25 steps of future | **invalidating** | `[C]`, and lag-1 ground-truth delta of 0.0497 dB `[M]` |
| 2 | Table III vs `results/` | Released predictions do not reproduce the published table: MAE 0.3951 vs 0.2486 (1.59x), RMSE 0.6218 vs 0.3926 (1.58x), MSE 0.3866 vs 0.1541 (2.51x) | **invalidating** | `[M]`, `tsslcd_table3_check.json` |
| 3 | Table II vs `Context2CondNew.py:236` | 4 attention heads stated; 750 % 4 ≠ 0 and the code asserts divisibility, so the stated config cannot run. Code uses 5 | **substantive** | `[C:82,236]` |
| 4 | §V-A vs `train.py` | "early stopping based on validation loss" and a 7:3 split; the released trainer has no validation set, no early stopping and no split | **substantive** | `[M]` — no `val`/`early`/`patience` token in `train.py` |
| 5 | §V-A vs `DataSetPrepare.py:68` | Missing rate 25% in paper, 0.1 in code, "25%" in the adjacent comment | **substantive** | `[C:68,84-85]` |
| 6 | `test.py` | Scores `[:, 0, :]` only; a 10-step model is reported as 1-step without the paper saying so | **substantive** | `[C]` |
| 7 | `DataSetPrepare.py:47` | Scaler refit per file, so test normalisation uses test statistics | **substantive** | `[C]` |
| 8 | Table II vs `config.py` | Latent 16 vs 32, batch 32 vs 256, 500 epochs vs 300+300+2000 | **inconsistency** | `[C:config.py]` |
| 9 | `DataSetPrepare.CFG` vs `config.py` | A second, contradictory config (latent 16, 2000 steps, batch 128); only `smooth_window`/`smooth_polyorder` are ever read from it | **inconsistency** | `[C]` |
| 10 | `Context2CondNew.py:14` | `ContextTransformerAE` defined twice; the first is a fully commented-out stub | **inconsistency** | `[C]` |
| 11 | §V-A2 `[P:10]` | "the following **three** baseline models" followed by **five** bullets | cosmetic | `[P:10]` |
| 12 | §V-B `[P:13]` | Horizon-sweep text points at Fig. 12, which is the continuous-missing figure | cosmetic | `[P:13]` |
| 13 | Figs. 9–10 captions | "four algorithms" in captions showing six | cosmetic | `[P:11-12]` |
| 14 | `dataset/testDataset.csv` | Malformed cell row 0 col 672 = `-134.20303.1` | cosmetic | `[M]` — masked by the header-consuming read |
| 15 | `results/gt_mat.csv` | Written with a pandas index header `0,…,749`; a naive `header=None` read yields +749 dBm in row 0 | cosmetic | `[M]` |
| 16 | §V-A `[P:10]` vs recovered data | Paper states 10,080 time points per sensor and a 7:3 split; the released data is 6839 rows split chronologically 55:45 | **substantive** | `[M]` train=rows 0-3760, test=rows 3761-6838 |
| 17 | `dataset/data.csv` | A USDCNY foreign-exchange series shipped in a spectrum repo | cosmetic | `[M]` |
| 18 | `trainDataset.csv` + `testDataset.csv` | Both carry the same malformed `-134.20303.1` at row 0 col 672, absent from the source; masked by the header-consuming read | cosmetic | `[M]` |
| 19 | `DataSetPrepare.py` | Window loop is `range(T - C - F)`, dropping the last valid window | cosmetic | `[C:MultiBandDataset]` |
| 20 | Eq. (27) | MAPE computed on dBm, a log quantity with arbitrary zero; the percentage is not a relative error | **substantive** | `[P:10]` |

## 9. Availability and reproducibility

- **Code**: `github.com/Xlab2024/TSS-LCD`, pushed 2025-12-03, 28.7 MB, 8 Python files, all compile `[M]`.
  Spectrum-specific and written by the authors — among our source papers, the most complete release.
- **Data**: upstream AERPAW set is public (aerpaw.org; Dryad `10.5061/dryad.hmgqnk9zn`). The repo ships
  `testDataset.csv` only. The repo's test file is **not** the AERPAW merged CSV we hold — the closest row
  differs by 14.9 dB `[M]` — so their exact preprocessing of the public data is not recoverable.
- **Checkpoints**: none.
- **License**: no LICENSE file `[M]`. Redistribution status unclear; we vendor nothing.
- **Dead links**: none; repo live as of 2026-10-29.

**Verdict: partial — upgraded once the deleted commit was found.** The data blocker is gone:

- ~~`trainDataset.csv` is absent~~ — **recovered** from commit `d7f1e51263`, together with the 6839-row
  source and a byte-exact verification of how the splits were cut.
- ~~Their split is unknown~~ — **determined**: chronological 55:45, train = source rows 0–3760.
- **Their preprocessing chain is confirmed exactly.** Regenerating `gt_mat.csv` from `testDataset.csv`
  (header-consuming read → per-column min-max → savgol(51,2) → window → `s[i+50]` → inverse transform)
  matches the published file to **1.7e-5 dB** `[M]`.

Remaining blockers:
1. Their released predictions score 1.59x worse than their published Table III, so there is no agreed target
   number to reproduce. Without checkpoints this cannot be resolved from the outside.
2. No checkpoints in any of the 14 commits, no seed, no early-stopping criterion, no validation set.
3. Practical: the 2,600-epoch schedule needs a GPU. **Measured** steady-state CPU cost (8 threads,
   forward + backward + Adam step, batch 32): **7.67 s/step, 116 steps/epoch = 14.84 min/epoch**, so
   **stage 1 alone (300 epochs of the context autoencoder) is 74.2 h** `[M]`. The full
   300 + 300 + 2000 schedule is multi-day on CPU and was not attempted.

What **is** now reproducible: architecture, full original data, the exact split, the exact preprocessing,
and an end-to-end training run on their real training file.

## 10. Compute cost

| | Theirs `[P:10]` | Ours `[M]` |
|---|---|---|
| Hardware | NVIDIA RTX 3090 | A30 24 GB (node currently unreachable) |
| Epochs | 500 (paper) / 300 + 300 + 2000 (code) | not yet run |
| Wall-clock to train | **not reported** `[?]` | not run; CPU lower bound measured at 14.84 min/epoch = 74.2 h for stage 1 of 3 `[M]` |
| Inference per sample | **not reported** `[?]` | not yet measured |
| Peak memory | not reported `[?]` | not yet measured |

**Inference is the unusual cost here.** Sampling runs the full 1000-step reverse chain per forecast
`[C:NoiseNet.py]`, so inference is ~1000 network evaluations per sample — three orders of magnitude more
than any one-shot regressor in our benchmark. The paper discusses the step-count/quality trade-off in
footnote 4 `[P:10]` but reports no timing. For a 1-minute-cadence forecasting task this is the figure that
decides deployability, and it is absent.

## 11. Our implementation

- **Representation**: 2D (time x flattened frequency-location), `[O:models/TSSLCD.py]`, 873 lines, 19 classes.
- **Residual variant**: none.
- **Built from**: their released code plus the paper, with our own decomposition into named modules
  (`TemporalFE`, `SpectralFE`, `SpatialFE`, `FeatureFusionModule`, `LatentSpaceEncoder/Decoder`,
  `EnhancedNoiseNet`, `DiffusionModel`, `TSSLCDForecaster`) matching the paper's block names rather than
  their file layout.

| # | Theirs | Ours | Disposition |
|---|---|---|---|
| 1 | `cosine_beta_schedule(s=0.0018)` | was `s=0.008` | **fixed** this session — now 0.0018 `[O]` |
| 2 | min-max [0,1] per frequency + savgol(51,2) | harness per-bin z-score, no smoothing | **deliberate** — adopting their smoothing would import leakage finding #1 into our benchmark. Reproduced only inside the dedicated reproduction run |
| 3 | 10% context zeroing at train and test | none | **unresolved** — to add as an explicit config flag |
| 4 | Future AE: Conv1d k=4 s=2 p=1, hidden 256, 3 blocks, final `Conv1d(hidden, D, k=3, p=1)` | k=3 + MaxPool, init_channels 32, extra location dim | **unresolved** — differs in geometry and parameter count |
| 5 | Context encoder hardcodes 3 locations | `spatial_locations` configurable | **deliberate** — ours must serve 6-site POWDER |
| 6 | Scores t+1 only | scores the full horizon | **deliberate**, but note it when quoting their numbers |
| 7 | `p_sample_loop` from `torch.randn`, no latent rescaling | identical | **faithful** — see below |

**On item 7.** Our earlier smoke test found the forecast path collapsing: the sampler returns latents with
std 1.6574 against an encoder manifold of std 0.3051 (5.43x too wide), and `decoder(cond_z)` scores 0.0084
while `decoder(p_sample_loop(cond_z))` scores 7.04 `[M]`. Their code has the same structure and no
`scale_factor`, `latent_scale` or 0.18215 anywhere. **This is therefore a faithful reproduction of their
method, not a defect in our port.** If we fix it, it must be reported as a declared improvement over the
published method, not as a fidelity correction.

## 12. What we are not sure about

| # | Uncertainty | Why unresolved | Could it change a reported number? | What would settle it |
|---|---|---|---|---|
| 1 | ~~Their train/test split~~ | **RESOLVED** — recovered from commit `d7f1e51263`; chronological 55:45, train = source rows 0–3760 `[M]` | — | — |
| 2 | Why released predictions miss Table III by 1.59x | No checkpoint, no seed, no run log | **Yes** — decides whether Table III is reproducible at all | Authors release checkpoints |
| 3 | Whether baselines saw the same smoothed target | Not stated; no baseline code released | **Yes** — decides whether Table III is like-for-like | Baseline configs |
| 4 | ~~Chronological or shuffled split~~ | **RESOLVED** — chronological; the two files are contiguous slices of the source `[M]`. Note windows are still shuffled *within* the training file (`shuffle=True`), which is normal | — | — |
| 5 | Latent rescaling (item 7 above) | Absent from their code; may be absent from the method or an omission in the release | **Yes** — our forecast path collapses without it | Authors clarify, or we test both arms |
| 6 | Our future-AE geometry (item 4) | Their conv stack differs from ours in kernel, stride and width | Probably minor; changes parameter count | Port their `F2Cnet` verbatim and compare |
| 7 | Effective missing rate | 25% / 10% / "25%" three-way disagreement | **Yes** — robustness figures | Run both; the sweep is cheap |
| 8 | Attention-head count | Table II says 4, which cannot run | Low — 5 is forced | Already settled: 5 |
| 9 | ~~Parameter count~~ | **RESOLVED** — 65,504,442 in their released code `[M]`. Our port's count still pending a config fix (§13) | No | — |
| 11 | Why their released predictions miss their own table | New: now that data and preprocessing are confirmed, the gap cannot be blamed on a preprocessing misreading on our side | **Yes** | Checkpoints, or a reply from the authors |
| 10 | Whether AERPAW nodes are sample-synchronised | Not stated; the 750-column flattening assumes it | Moderate, for the spatial branch | AERPAW documentation |

## 13. Reproduction results

**(a) Their released predictions against their own ground truth**, 3018 x 750, dBm, t+1, computed by us `[M]`:

| Predictor | MSE | RMSE | MAE | MAPE |
|---|---|---|---|---|
| TSS-LCD, paper Table III `[P:12]` | 0.1541 | 0.3926 | 0.2486 | 0.2075% |
| **TSS-LCD, released `pred_matours.csv`** | **0.3866** | **0.6218** | **0.3951** | 0.3296% |
| **Last-value persistence** | **0.0064** | **0.0802** | **0.0497** | — |
| Lookback mean (L=60, n=2958) | 0.5470 | 0.7396 | 0.4571 | — |

Two results. The released artifacts are **1.59x worse in MAE** than the paper's own table. And persistence
beats both by a wide margin — 60x in MSE against the released predictions, 24x against the published table.

Row ordering was verified before trusting persistence: mean |gt[i] − gt[i−k]| = 0.0497 / 0.2926 / 0.7334 dB
at k = 1 / 10 / 60 against 1.6979 dB for random row pairs `[M]` — monotone in lag, so the rows are
time-ordered and persistence is a legitimate reference. The residual is unbiased (−0.0096 dB, std 0.6217).

**(b) Preprocessing chain verified `[M]`.** Regenerating their ground truth from `testDataset.csv` through
their exact pipeline reproduces `results/gt_mat.csv` to **max 1.67e-5 dB, mean 3.50e-6 dB** over
3018 x 750 values. Every preprocessing claim in §3 of this card is therefore confirmed by reconstruction,
not inferred from reading code.

**(c) Training on their real training split**: **not yet run.** The data is now in hand
(`tsslcd_hist/trainDataset.csv`), but all GPU capacity is unavailable: `ssh:chameleon-temp` fails with
`Could not resolve hostname`, and `ssh:chameleon-uc` times out at 192.5.86.172:22. CPU is not a viable substitute: measured
14.84 min/epoch, i.e. 74.2 h for the context autoencoder's 300 epochs alone `[M]`. Commands are staged in `TSSLCD_REPRODUCTION.md` §5, now pointing
at the recovered file rather than the AERPAW substitute.

**(d) Our implementation, smoke build**: **fails.** `KeyError: 'latent_dim'` at
`training/common/model_factory.py:509` `[M]` — the registry anchor omits a key the factory requires. This is
a harness/config gap, not a model defect; the real YAML in `training/configs/` supplies it. Seven of the nine
smoke failures in the A30 sweep share this shape.

## 14. Open questions and next actions

- [ ] Restore GPU capacity, then run the staged training on the **recovered** `trainDataset.csv`
      (smoke 1/1/1, then 300/300/2000) and `test.py`; compare against `pred_matours.csv` and against persistence
- [ ] Fix the `latent_dim` registry anchor so `tss_lcd` builds in the smoke harness; record parameter count
- [ ] Decide and document the latent-rescaling arm (faithful vs fixed) and run both
- [ ] Add the 10% context masking as an explicit flag
- [ ] Email the authors for **checkpoints** — the data and split questions are now answered from history,
      so checkpoints are the only remaining ask, and they alone can explain the Table III gap
- [ ] Quantify the leakage directly: retrain with savgol applied *after* splitting and report the gap
- [ ] Measure sampling wall-clock per forecast (1000-step chain) for the deployability column

## 15. Evidence provenance

- **Paper**: Zotero `3FYBL4KU`, PDF at `Zotero/storage/VUFP7PZM/`, 15 pages, journal pages 7259–7273
  (pages 7268–7271 carry §V). Tables II–IV read from rendered pages 10, 12, 13 — they are not in the text layer.
- **Code**: `github.com/Xlab2024/TSS-LCD`. HEAD = `b156d0a007` (2025-12-03). 14 commits total; full history
  walked 2026-10-30 via the GitHub trees API. **Data recovered from commit `d7f1e51263`** (2025-12-02).
  No `.pth`/`.pt`/`.ckpt` blob exists in any commit, despite `da39384e0f` being titled "Final revise without
  large checkpoint file" `[M]`.
- **Recovered files** (sha256, first 16 hex): `trainDataset.csv` 971a118bd7b28839,
  `merged_power_data_sub6GHz_avg_per_minute.csv` 746ac5560c1fdd3f, `data.csv` 80520635aa86c022.
- **Data**: `results/gt_mat.csv`, `results/pred_matours.csv` (3018 x 750 each), `dataset/testDataset.csv`
  (3079 x 750).
- **Our runs**: `tsslcd_published_audit.json`, `tsslcd_table3_check.json`, `smoke_all_models.json`
  (`tss_lcd` entry), `smoke_tss_lcd2_result.json`, `smoke_tss_lcd3_result.json`.
- **Our code**: `models/TSSLCD.py` at 873 lines, beta schedule aligned to 0.0018 on 2026-10-29.
- **Card author / date**: automated audit, 2026-10-30. Revise after the GPU reproduction run.
