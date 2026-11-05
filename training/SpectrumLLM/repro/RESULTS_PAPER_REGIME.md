# Baselines and metric space for the SpectrumLLM article's own regime

Liu et al., *SpectrumLLM*, IEEE Wireless Communications 33(2), pp. 193-199, 2026. Data:
`github.com/LIUC-000/SP` @ `64b7fa5e96`. Input length T=192 (48 h at 15-min), horizons K=12/24/48/96/192,
chronological 7:3 split, test windows starting at `n - n_test - T`.

## 1. What space is the published MSE in? **Raw PSD units.**

The article never states it. The sentence "the MSE loss is computed between the predicted and actual PSD
values" (p. 198) is in the fine-tuning paragraph and describes the training objective; in Time-LLM's own
code that same `nn.MSELoss()` is applied to standardized tensors, so the wording does not decide it.
Reading the evaluation code is also inconclusive *as evidence about this paper*: the Time-Series-Library
defaults to `--inverse False`, Time-LLM's `run_main.py` never calls `inverse_transform`, and OFA's
`test()` scores raw model output - all of which would imply standardized - but that only binds if the
authors ran those paths unmodified.

Settled empirically instead. A per-channel least-squares linear predictor (192 lags -> K steps, fitted on
the training split) is a tight proxy for DLinear - arguably the best achievable linear map, so it should
sit at or just below the published DLinear row. Fitted and scored in each candidate space:

| Dataset | K | published DLinear | linear probe, **raw** | rel. err | linear probe, standardized | rel. err |
|---|---|---|---|---|---|---|
| 4-ch | 12 | 0.2718 | 0.2741 | 0.8% | 0.3099 | 14.0% |
| 4-ch | 24 | 0.3036 | 0.2920 | 3.8% | 0.3634 | 19.7% |
| 4-ch | 48 | 0.3305 | 0.3014 | 8.8% | 0.3952 | 19.6% |
| 4-ch | 96 | 0.4019 | 0.3193 | 20.6% | 0.4313 | 7.3% |
| 4-ch | 192 | 0.4043 | 0.3697 | 8.6% | 0.5245 | 29.7% |
| 25-ch | 12 | 0.6080 | 0.5779 | 5.0% | 0.3961 | 34.9% |
| 25-ch | 24 | 0.8025 | 0.7596 | 5.3% | 0.5021 | 37.4% |
| 25-ch | 48 | 0.9809 | 0.9690 | 1.2% | 0.6206 | 36.7% |
| 25-ch | 96 | 1.2581 | 1.1808 | 6.1% | 0.7516 | 40.3% |
| 25-ch | 192 | 1.6487 | 1.4628 | 11.3% | 0.9577 | 41.9% |

Mean |relative error|: **raw 8.5% (4-ch) and 5.8% (25-ch)**, against standardized 18.1% and 38.2%. In
standardized space the probe would beat published DLinear by ~40% at every 25-channel horizon, which is
not credible for an equivalent model class. **All comparisons below are therefore in raw PSD units**, and
an earlier version of this file that used standardized units was wrong.

### Recheck with the actual DLinear model (2026-11-04)

The verdict above rested on a least-squares probe. Repeated with the real DLinear
(`models/DLinear.py`), both training spaces, matched protocol (lr 5e-4, 50 epochs,
literal split, seeds 0/1/2), scoring the same test windows in both spaces
(`results_dlinear_raw.json` raw-trained; `results_dlinear.json` std-trained).
Mean |relative error| to the published DLinear row:

| train / score | 4-ch | 25-ch |
|---|---:|---:|
| **raw / raw** | **5.9%** | 24.6% |
| std / std | 12.1% | 38.5% |
| raw / std | 12.5% | 48.8% |
| **std / raw** | **6.5%** | **12.2%** |

Two conclusions, one old and one new:

1. **The reporting space is raw PSD -- confirmed.** Every raw-*scored* column is far
   closer to the published row than its std-scored counterpart. Under the std-scored
   reading a plain linear model would beat the published DLinear by ~38% at every
   25-channel horizon, which is not credible.
2. **The training space was probably standardized.** Raw-loss training overshoots the
   published 25-channel row by 13-40% (it optimizes exactly the loud-channel-weighted
   objective being scored, so it does *better* than the article's own number), while
   std-trained / raw-scored is the only single protocol that fits both scenarios
   closely. That is the Time-Series-Library workflow: scale, train, then report in the
   original units. The reproduction's DLinear campaign is std-trained and therefore
   already the right comparator; the Time-LLM arms are raw-trained but carry RevIN,
   which normalizes per instance and makes the training space a much weaker knob there.

### Structural variants: matching Time-Series-Library's DLinear exactly (2026-11-04)

"Ours beats theirs by 24%" cried out for a configuration check, so our port was diffed
against Time-Series-Library's `models/DLinear.py`. The architecture is identical
(series_decomp moving_avg=25 with replicate padding, two seq_len->pred_len maps,
no internal normalization). Two things differ, and both are now flags
(`--dlinear-init tslib`, `--dlinear-individual`): TSLib initialises both maps to
`(1/seq_len)*ones` (the model starts as the lookback-mean predictor), and TSLib's
`--individual` fits one map pair per channel where our default shares maps across
channels. All variants below: 25 channels, literal split, seeds 0/1/2, scored raw.

| maps | init | trained in | mean rel err vs published 25-ch row |
|---|---|---|---:|
| shared | random | raw | -24.6% |
| shared | TSLib | raw | -24.8% |
| individual | random | raw | -11.8% |
| individual | TSLib | raw | -12.4% |
| shared | random | std (TSLib default workflow) | -12.2% |
| **individual** | **TSLib** | **std** | **-5.6%** |

Three conclusions:

1. **The init is irrelevant** (differences within seed noise) -- rule it out.
2. **Map sharing is the big one.** Shared maps trained on the raw loss reallocate
   capacity toward the loud channels that dominate the raw metric -- that, not a
   scoring error and not model superiority, is where the -24.6% "beat" came from.
   Per-channel maps cannot reallocate, so the overshoot largely vanishes.
3. **The published 25-channel row is consistent with TSLib's `--individual` workflow**
   (-5.6%; the residual sits in the K=96/192 cells, within reach of TSLib's
   lr=0.01/10-epoch schedule and of the table's own internal inconsistencies). At
   4 channels the same configuration undershoots by -11.8% (concentrated in the
   published K=96 cell, 0.4019, which every fit we have -- including the linear probe
   at +20.6% -- says is anomalous), while shared maps match 4 channels closely.
   No single DLinear configuration lands on both published rows, mirroring the
   channel-mode reversal found for Time-LLM.

The "we beat the published DLinear" reading is therefore withdrawn: it was our
configuration, not their number and not our scorer. Raw results archived under `dlinear/` (gitignored).

### Torch verification of the TSLib-faithful configuration (2026-11-04)

`DLINEAR_VARIANTS.md` identified `B_recipe_tslib` as the best of its 45 configurations
(ratios 0.990 / 1.008), using its numpy CPU harness (`dlinear/DLINEAR_VARIANTS.md`). The runner gained
`--lr-decay` (Time-Series-Library's `lr = base * 0.5**epoch` schedule) so the exact
configuration could be verified through the real torch pipeline:

`--split tslib --space std --dlinear-init tslib --lr 0.005 --lr-decay 0.5
--epochs 10 --patience 3`, shared heads, kernel 25, seed 0, 25 channels
(`results_dlinear_tslib_verify.json`):

| K | ours | published | ratio |
|---|---:|---:|---:|
| 12 | 0.607993 | 0.6080 | **1.0000** |
| 192 | 1.674385 | 1.6487 | 1.0156 |

The K=12 cell agrees with the published value to four decimal places. That is a
positive end-to-end reproduction -- of the split borders, the windowing, the
per-channel scaling, the training recipe, and the raw-space reporting -- and it
settles the residual question of this section: under Time-Series-Library's own
protocol the published 25-channel DLinear row is exact, so the earlier shortfalls
were ours to fix, not theirs to explain.

### Direct run of Time-Series-Library itself (2026-11-04)

To remove the last porting doubt, the real library (github.com/thuml/Time-Series-Library,
main) was cloned into a container and its own `run.py` + `Dataset_Custom` +
`exp_long_term_forecasting` executed on the ElectroSense CSVs
(`tslib_direct.sh`, raw log in `tslib_direct_results.txt`). Two adjustments only:
`--label_len 0` (cosmetic for DLinear -- current TSLib's `__len__` ignores it, so
the scored window set is identical either way) and a `sed` of the hardcoded
`fix_seed = 2021` when varying seeds. Everything else is their code verbatim:
`--data custom --features M --target 500 --seq_len 192 --lradj type1
--train_epochs 10 --patience 3 --batch_size 32 --lr 0.005 --inverse`.

**spec4, raw-space MSE (`--inverse`):**

| K | published | seed 2021 | seeds 0/1/2 | 4-seed mean | ratio |
|---|---:|---:|---|---:|---:|
| 12 | 0.2718 | 0.2747 | 0.2778 / 0.2774 / 0.2782 | 0.2770 | 1.019 |
| 24 | 0.3036 | 0.3033 | 0.3062 / 0.3046 / 0.3066 | 0.3052 | 1.005 |
| 48 | 0.3305 | 0.3197 | 0.3233 / 0.3243 / 0.3206 | 0.3220 | 0.974 |
| 96 | 0.4019 | 0.3288 | 0.3329 / 0.3273 / 0.3247 | 0.3284 | **0.817** |
| 192 | 0.4043 | 0.3929 | 0.4003 / 0.3919 / 0.3882 | 0.3933 | 0.973 |

**spec25, full row** (seed 2021): K=12 -> 0.6017 (published 0.6080, ratio
0.990); K=24 -> 0.7776 (0.8025, 0.969); K=48 -> 0.9748 (0.9809, 0.994);
K=96 -> 1.0860 (1.2581, **0.863**); K=192 -> 1.6611 (1.6487, 1.007). The
port's four-decimal K=12 match (0.6080) sits within the same band.

Findings:

1. **The protocol is settled, and its run-to-run noise is large.** Repeating the
   exact protocol across seeds (2021 plus 0-5, both the real library and the
   port) shows the short 10-epoch `type1` schedule swings the test MSE by up to
   6-12% at long horizons (spec4 K=192 spans 0.352-0.400 across seeds; spec25
   K=96 spans 1.086-1.271 in TSLib alone). Any single-seed comparison at this
   schedule is fragile -- including the seed-2021 draws that first looked like
   exact matches. The published tables print one number per cell with no seed
   count, so cells should be judged against the band, not a point.
2. **Judged against the band, 8-9 of the 10 published DLinear cells are
   consistent with the identified protocol**: 4/K24, 25/K12, 25/K48, 25/K192
   sit inside both implementations' seed ranges; 25/K96 sits inside TSLib's own
   range (1.086-1.271 around the published 1.2581 -- the seed-2021 draw of
   1.086 that first looked anomalous was itself the outlier); 4/K12, 4/K48 and
   25/K24 sit a few percent above the ranges, within reach of their own draw.
3. **The 4-channel K=96 cell is the one genuine anomaly**: published 0.4019
   against 0.323-0.343 over every seed of both implementations -- 17-24% above
   the entire band, unreached by ~500 fits across four implementations and two
   learning-rate brackets. That cell is a printing or provenance error in the
   article.
4. **Without `--inverse` TSLib reports standardized-space MSE** (0.3093 at
   spec4 K=12, not 0.2718) -- the authors must have passed the flag or scored
   outside the library.
5. **TSLib's argparse default lr (1e-4) badly underfits at 10 epochs**
   (spec4 K=12 -> 0.3599): their row requires the script lr.
6. **`--individual` is dead code in TSLib main**: the runs with it are
   bit-identical to without it, because `exp_basic` calls `Model(args)` and
   DLinear's `__init__(self, configs, individual=False)` never reads
   `configs.individual` -- the wiring from LTSF-Linear was lost. Their DLinear
   is always shared-head; the per-channel variant exists only in our port.

The canonical DLinear results are now produced by `training/SpectrumLLM/run_dlinear.py`
(exact TSLib protocol, seed 2021) via `repro/run_dlinear.sh` ->
`repro/results_dlinear_settled.json`. All diagnostic campaigns, the protocol
search, and the direct TSLib run logs are archived under `repro/dlinear/`
(gitignored; the seed-sweep records `settled_seed_*.json` live there).

**The final table** (`results_dlinear_settled.json`, seed 2021, raw PSD MSE):

| scenario | K | ours | published | ratio | epochs run |
|---|---|---:|---:|---:|---:|
| 4-ch | 12 | 0.2760 | 0.2718 | 1.016 | 8 |
| 4-ch | 24 | 0.3084 | 0.3036 | 1.016 | 8 |
| 4-ch | 48 | 0.3214 | 0.3305 | 0.972 | 9 |
| 4-ch | 96 | 0.3261 | 0.4019 | **0.811** | 10 |
| 4-ch | 192 | 0.3524 | 0.4043 | 0.872 | 6 |
| 25-ch | 12 | 0.6109 | 0.6080 | 1.005 | 8 |
| 25-ch | 24 | 0.7388 | 0.8025 | 0.921 | 4 |
| 25-ch | 48 | 0.9522 | 0.9809 | 0.971 | 7 |
| 25-ch | 96 | 1.2385 | 1.2581 | 0.984 | 10 |
| 25-ch | 192 | 1.5768 | 1.6487 | 0.956 | 6 |

Read this table together with the seed band of finding 1: every cell except
4-ch K=96 lies within (or a few percent from) the protocol's own run-to-run
spread, as does the real library's own output.


## 2. 4-channel (article Table I), raw PSD^2

| Predictor | K=12 | K=24 | K=48 | K=96 | K=192 | Average |
|---|---|---|---|---|---|---|
| DLinear | 0.2718 | 0.3036 | 0.3305 | 0.4019 | 0.4043 | 0.3424 |
| LSTM | 0.6822 | 0.6751 | 0.6726 | 0.5516 | 0.5420 | 0.6247 |
| LightTS | 0.3177 | 0.3280 | 0.3439 | 0.3667 | 0.4449 | 0.3602 |
| PatchTST | 0.2707 | 0.3044 | 0.3268 | 0.3061 | 0.3771 | 0.3170 |
| Informer | 0.7377 | 0.5264 | 0.5837 | 0.5363 | 0.6409 | 0.6050 |
| Reformer | 0.3577 | 0.4819 | 0.4078 | 0.4948 | 0.5807 | 0.4646 |
| Autoformer | 0.4619 | 0.6480 | 0.5879 | 0.6026 | 0.4813 | 0.5563 |
| TimesNet | 0.3211 | 0.3640 | 0.4530 | 0.6515 | 0.8517 | 0.5283 |
| Time-LLM (BERT) | 0.3045 | 0.3068 | 0.2911 | 0.3081 | 0.3438 | 0.3109 |
| Time-LLM (GPT2) | 0.2621 | 0.2982 | 0.3154 | 0.3157 | 0.3460 | 0.3075 |
| OFA (GPT2) | 0.2686 | 0.2892 | 0.2970 | 0.2958 | 0.3427 | 0.2987 |
| | | | | | | |
| **Last-value persistence** | 0.5638 | 0.9272 | 1.3976 | 1.3807 | 1.4142 | 1.1367 |
| **Lookback mean (L=192)** | 0.7076 | 0.7130 | 0.7177 | 0.7179 | 0.7373 | 0.7187 |
| **Seasonal naive (24 h)** | 0.4474 | 0.4472 | 0.4466 | 0.4473 | 0.4503 | 0.4478 |

- K=12: strongest reference is seasonal naive 24h at 0.4474, which beats **3 of 11** published methods (LSTM, Informer, Autoformer).
- K=24: strongest reference is seasonal naive 24h at 0.4472, which beats **4 of 11** published methods (LSTM, Informer, Reformer, Autoformer).
- K=48: strongest reference is seasonal naive 24h at 0.4466, which beats **4 of 11** published methods (LSTM, Informer, Autoformer, TimesNet).
- K=96: strongest reference is seasonal naive 24h at 0.4473, which beats **5 of 11** published methods (LSTM, Informer, Reformer, Autoformer, TimesNet).
- K=192: strongest reference is seasonal naive 24h at 0.4503, which beats **5 of 11** published methods (LSTM, Informer, Reformer, Autoformer, TimesNet).


## 3. 25-channel (article Table II), raw PSD^2

| Predictor | K=12 | K=24 | K=48 | K=96 | K=192 | Average |
|---|---|---|---|---|---|---|
| DLinear | 0.6080 | 0.8025 | 0.9809 | 1.2581 | 1.6487 | 1.0597 |
| LSTM | 1.5689 | 2.1121 | 2.0390 | 2.1126 | 2.4215 | 2.0508 |
| LightTS | 0.7006 | 0.9350 | 1.1590 | 1.3474 | 1.6807 | 1.1645 |
| PatchTST | 0.6104 | 0.7526 | 0.9043 | 1.0161 | 1.0796 | 0.8726 |
| Informer | 3.9209 | 3.0819 | 3.2069 | 2.7101 | 2.2602 | 3.0360 |
| Reformer | 1.4271 | 2.4233 | 2.4788 | 2.3109 | 2.2178 | 2.1716 |
| Autoformer | 1.3279 | 1.3601 | 1.4001 | 1.3995 | 1.5653 | 1.4106 |
| TimesNet | 1.1265 | 1.0902 | 1.1679 | 1.2690 | 1.3063 | 1.1920 |
| Time-LLM (BERT) | 0.6321 | 0.7564 | 0.8666 | 0.9720 | 1.0311 | 0.8516 |
| Time-LLM (GPT2) | 0.6166 | 0.7592 | 0.8564 | 0.9686 | 1.0429 | 0.8488 |
| OFA (GPT2) | 0.5846 | 0.7593 | 0.8698 | 0.9540 | 1.0634 | *blank* (0.8462) |
| | | | | | | |
| **Last-value persistence** | 0.8076 | 1.2663 | 1.8538 | 1.9612 | 2.1640 | 1.6106 |
| **Lookback mean (L=192)** | 1.2096 | 1.2306 | 1.2495 | 1.2850 | 1.3847 | 1.2719 |
| **Seasonal naive (24 h)** | 1.2732 | 1.2675 | 1.2506 | 1.2364 | 1.2071 | 1.2470 |

- K=12: strongest reference is persistence at 0.8076, which beats **5 of 11** published methods (LSTM, Informer, Reformer, Autoformer, TimesNet).
- K=24: strongest reference is lookback mean at 1.2306, which beats **4 of 11** published methods (LSTM, Informer, Reformer, Autoformer).
- K=48: strongest reference is lookback mean at 1.2495, which beats **4 of 11** published methods (LSTM, Informer, Reformer, Autoformer).
- K=96: strongest reference is seasonal naive 24h at 1.2364, which beats **7 of 11** published methods (DLinear, LSTM, LightTS, Informer, Reformer, Autoformer, TimesNet).
- K=192: strongest reference is seasonal naive 24h at 1.2071, which beats **7 of 11** published methods (DLinear, LSTM, LightTS, Informer, Reformer, Autoformer, TimesNet).

## 4. Our implementation, 4-channel, all horizons, raw PSD units

Merged build (article-first, Time-LLM where the article is silent), trained **on raw PSD** with RevIN
normalising per instance inside the model. 15 runs = 5 horizons x 3 seeds. Hyperparameters are
Time-LLM's argparse defaults (lr 1e-4, 10 epochs, batch 32); the article states none. Every run has a
checkpoint; raw rows in `results_spec4_all.json`, tidy exports in `spectrumllm_{results,summary,history}.csv`.

| K | **Ours** mean ± sd (3 seeds) | rel. sd | Time-LLM (GPT2) published | ours / theirs | seasonal naive 24 h | persistence |
|---|---|---|---|---|---|---|
| 12 | **0.2718** ± 0.0052 | 1.9% | 0.2621 | 1.037x | 0.4474 | 0.5638 |
| 24 | **0.2795** ± 0.0030 | 1.1% | 0.2982 | 0.937x | 0.4472 | 0.9272 |
| 48 | **0.3088** ± 0.0101 | 3.3% | 0.3154 | 0.979x | 0.4466 | 1.3976 |
| 96 | **0.3375** ± 0.0327 | 9.7% | 0.3157 | 1.069x | 0.4473 | 1.3807 |
| 192 | **0.3368** ± 0.0301 | 8.9% | 0.3460 | 0.973x | 0.4503 | 1.4142 |
| **Average** | **0.3069** | | 0.3075 | 0.998x | 0.4478 | 1.1367 |

**The port tracks Time-LLM (GPT-2) across the whole horizon range.** Average 0.3069 against 0.3075
— a difference of 0.2%. Per horizon the ratio stays within 0.937x-1.069x, with ours lower at
K = 24, 48 and 192 and higher at K = 12 and 96. There is **no systematic drift with horizon**, which is
the signature a geometry error in patching, head shape or normalisation would leave; a constant-factor
offset would indicate a scaling error. Neither is present. Our port clears the strongest trivial
reference (24-h seasonal naive, ~0.447) at every horizon by 1.3x-1.6x.

**Seed variance, measured properly.** Relative standard deviation across 3 seeds runs from **1.1% at
K=24 to 9.7% at K=96** (mean 5.0%). This supersedes the 1.19% two-run figure quoted earlier, which
understated it by roughly 4x. Consequences worth stating:

- At **n=5 seeds**, gap-to-spread against the published Time-LLM (GPT2) cell is **+2.5 sd at K=12**
  (ours higher) and **under 1 sd at K=24/48/96/192** (-0.9, -0.6, +1.0, -0.5). Mean ratio 1.012.
- This REPLACES the three-seed reading (which gave 6.3 sd at K=24 on sd=0.0030). Adding seeds 3 and 4
  widened that sd to 0.0125 and moved the mean from 0.2795 to 0.2869; the separation disappeared. At
  n=3 the spread was simply not resolved. The article gives one number per cell with no seed count and
  no variance, so its own spread remains unknown and every such comparison is one-sided.
- A single run of this architecture cannot resolve differences below ~5%. That is a property of our
  build on this data; it is not measured for the published models and must not be transferred to them.

| Cost (mean over 15 runs) | |
|---|---|
| Train wall-clock | 277 s per run (shared A30) |
| Inference, batch 1 | 33.9 ms/sample |
| Trainable parameters | 52,685,760 (K=12) to 53,239,440 (K=192) |
| Checkpoints | 15 files, 3.0 GB total, `repro/checkpoints/` (gitignored; paths and sha256 prefixes in `spectrumllm_results.csv`) |

## Coverage against the article's published tables

The article publishes **110 MSE cells**: 2 scenarios (4-channel Table I, 25-channel Table II)
x 11 models x 5 horizons. We have reproduced **20 of them (18%)** -- the Time-LLM (GPT2) and
Time-LLM (BERT) rows on both scenarios. The remaining **90 are not assessed**: OFA (GPT2),
DLinear, LSTM, LightTS, PatchTST, Informer, Reformer, Autoformer and TimesNet. Every cell is
listed in `comparison_vs_paper.csv` with a `status` column of `reproduced`, `not assessed`, or
`ours only`; nothing is left blank.

Mean ratio of our value to the published cell (1.0 = exact, >1 = we are worse):

| scenario | published row | our arm | mean ratio | seeds |
|---|---|---|---|---|
| 4 channels  | Time-LLM (GPT2) | channel-independent | 0.937 | 3 |
| 4 channels  | Time-LLM (BERT) | channel-independent | 0.920 | 3 |
| 25 channels | Time-LLM (GPT2) | channel-independent | 0.887 | 3 |
| 25 channels | Time-LLM (BERT) | channel-independent | 0.966 | 1 |

Thirty further rows carry no published counterpart and are marked `ours only`: the
all-channel-token reading of the article's eq. (1), the frozen-random control, and all of
the 151-channel scenario, which the article ships but never reports.

### Every number here was regenerated from the weights

No value in this table was carried over from an earlier summary. All 150 checkpoints were
reloaded, each model rebuilt from the config stored inside its own checkpoint, its test windows
rebuilt from the scenario, horizon, lookback and split it records, and the MSE recomputed.
**All 150 reproduced bit-exactly** (max |recomputed - stored| = 0.00e+00).

Two conditions were required to get there, and both are findings in their own right:

1. **The random-backbone runs are not reloadable without their seed.** The untrained transformer
   is frozen, so it is never written to the checkpoint; it is reconstructed by
   `torch.manual_seed(seed)` before the model is built. Rebuilding without re-seeding gave 0.6454
   instead of 0.2840 on one spec4 cell -- a 2.3x error that looks like a result.
2. **Test MSE depends on the evaluation batch size.** Prompts are tokenised with `padding=True`
   and no attention mask, so each prompt is padded to the longest in its batch and the padding
   participates in attention. On spec151 the same weights score 1.1723 at batch 32 and 1.2174 at
   batch 1, a 3.9% swing; disabling TF32 changes nothing, so it is the padding, not precision.
   Every number here is computed at the batch its run was evaluated under.
