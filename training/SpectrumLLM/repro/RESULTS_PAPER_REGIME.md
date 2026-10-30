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

## 4. Our implementation, 4-channel K=12, raw PSD units

First run of the merged build (article-first, Time-LLM for the gaps; 32/32 verification checks and 10/10
unit tests pass). Trained **on raw PSD**, RevIN normalising per instance inside the model and inverting at
the output, MSE loss in PSD units - the arrangement the article describes.

| | |
|---|---|
| **Ours, spec4 K=12, seed 0** | **0.2749** PSD^2 |
| Time-LLM (GPT2) (published) | 0.2621 | 
| OFA (GPT2) (published) | 0.2686 | 
| PatchTST (published) | 0.2707 | 
| DLinear (published) | 0.2718 | 
| Time-LLM (BERT) (published) | 0.3045 | 
| LightTS (published) | 0.3177 | 
| TimesNet (published) | 0.3211 | 
| Reformer (published) | 0.3577 | 
| Autoformer (published) | 0.4619 | 
| LSTM (published) | 0.6822 | 
| Informer (published) | 0.7377 | 

| Our run | |
|---|---|
| space trained/scored in | raw |
| epochs | 10 |
| batch size | 32 |
| learning rate | 0.0001 |
| trainable parameters | 52,685,760 |
| train windows | 1,626 |
| test windows | 851 |
| train wall-clock (s, shared A30) | 274.1 |
| inference s/sample (batch 1, shared A30) | 0.03262 |

**Verdict: the port reaches the published range.** 0.2749 against a best published cell of
0.2621 (Time-LLM GPT2) is **1.049x** - 4.9% worse than the best row, 2.4% worse
than OFA (0.2686), 1.2% worse than DLinear (0.2718), and better than Time-LLM (BERT) 0.3045, LightTS
0.3177 and TimesNet 0.3211. It also clears all three trivial references (best of which is the 24-h
seasonal naive at 0.4474).

Caveats on this number, in order of size:

1. **One seed, one horizon, one dataset.** No dispersion. The article reports single runs too, but that is
   not a defence.
2. **Hyperparameters are Time-LLM's argparse defaults** (lr 1e-4, 10 epochs, batch 32), not tuned and not
   the article's - it states none. Time-LLM's own shipped script disagrees with its argparse by 100x on
   learning rate, so there is no single "their setting" to adopt.
3. **Validation loss was still falling** in places (0.4355 at epoch 3, 0.4405 at epoch 9, non-monotone);
   the 10-epoch budget is Time-LLM's default, not a converged run.
4. The GPU was shared with another job throughout, so the 274 s wall-clock is not a clean timing.

What this does **not** show: that our build is Time-LLM. It is an implementation of the article's
described paradigm with Time-LLM's values where the article is silent, and it differs from Time-LLM in
channel handling and patch embedding by deliberate choice (see `TIMELLM_OFA_REFERENCE.md`).

Raw rows: `results_k12_raw.json`. Runner: `run_paper_regime.py --space raw`.
