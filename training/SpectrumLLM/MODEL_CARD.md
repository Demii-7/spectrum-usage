# Model Card — SpectrumLLM

Companion to `training/SpectrumLLM/README.md` (build, tensor shapes, layer-by-layer architecture).
This card is the paper-and-reproduction audit. Source tags: `[P:n]` = PDF page n, `[D:path]` = the
authors' released **data** repository, `[O:file]` = our code, `[M]` = measured by us, `[?]` = unknown.

```yaml
model: spectrumllm
display_name: SpectrumLLM
paper_cite: "C. Liu, Y. Wang, S. Mao, D. Niyato, X. Wang, G. Gui, 'SpectrumLLM: Large Language Models for Next-Generation Spectrum Prediction', IEEE Wireless Communications, vol. 33, no. 2, pp. 193-199, Apr. 2026"
doi: 10.1109/MWC.2025.3596934
zotero_key: 5ZG38NKM          # PDF attachment JWQCQQUX
year: 2026
venue: IEEE Wireless Communications (magazine, "Open Call Article")
benchmarked: false            # implemented in our repo, not in the manuscript's tables
representation: 2d
residual_variant: false
params: 52685760              # [M] trainable, measured at T=192 / 4ch / K=12 / 1000 prototypes / d_ff 32, GPT-2 6 layers. 95% is the prototype probe, not the head
code_available: none-for-this-model   # no SpectrumLLM network exists; BOTH instantiations it benchmarks are fully released (Time-LLM Apache-2.0 @b13e881f86, OFA @9f3875dbba) - see repro/TIMELLM_OFA_REFERENCE.md
data_available: full          # github.com/LIUC-000/SP - 4ch / 25ch / 151ch ElectroSense CSVs, downloaded and verified [M]
checkpoints_available: false
license: none-declared        # no LICENSE in the data repo
reproducible: architecture-only
repro_blocker: "the paper publishes a paradigm, not a network: no model code, no layer sizes, no epochs/batch/LR/optimizer/seed, no hardware"
compares_vs_trivial: false
beats_persistence: true       # [M] ours 0.2749 raw PSD^2 at 4ch K=12 vs seasonal naive 0.4474, persistence 0.5638, lookback mean 0.7076
released_artifacts_match_paper: n/a   # no predictions and no checkpoints were released
leakage_found: none-found     # nothing in the described chain is non-causal, but it is unverifiable without code
```

## 0. At a glance

| | |
|---|---|
| Paper | 7-page magazine article. Describes an LLM-for-spectrum-prediction *paradigm* (RevIN -> patch tokenization -> prototype reprogramming -> prompt-as-prefix -> frozen/partly-frozen LLM -> linear head) and benchmarks two existing models, Time-LLM and OFA, on spectrum data |
| Code | **None for SpectrumLLM.** The only code link is `github.com/thuml/Time-Series-Library`, a third-party generic forecasting library `[P:5]`. There is no SpectrumLLM repository, no config, no run script |
| Data | **Full and public**: `github.com/LIUC-000/SP` ships `spec4_data.csv`, `spec25_data.csv`, `spec151_data.csv` — 2872 x {4,25,151}, 15-min, 2021-05-28 to 2021-06-26 `[M]` |
| Reproducible | **architecture-only** — the data and the task regime are exactly recoverable; the network is not, because none was published |
| Beats a trivial reference | **The paper never checks; the LLM rows do, the weaker baselines often do not.** In raw PSD units a trivial reference beats **3-5 of the 11** published methods on 4-channel and **4-7 of 11** on 25-channel, depending on horizon `[M]`. All three LLM rows clear every reference at every horizon. (An earlier revision claimed the references beat all 11 on 25-channel; that was computed in the wrong metric space and is withdrawn — §3) |
| Headline claim | "OFA and Time-LLM outperformed all baselines, exhibiting significantly lower MSE and MAE across different prediction steps" `[P:6]` |
| Our verdict on that claim | **Overstated on the paper's own tables.** Six cells contradict it (a non-LLM baseline is best), the MAE half of the claim has no table behind it anywhere in the article, and Table II's best average-MSE cell is blank `[M]` |
| Capacity | The paper reports no parameter count. Our build at the tiny anchor is 134.6M total / 52.7M trainable `[M]` against a training split of 8,040 (4-channel) or 50,250 (25-channel) scalars `[M]` |

## 1. Task as the paper formulates it

- **Problem statement**: N-channel, K-step prediction. Given `[X_{t-T+1}, ..., X_t]` with `X_t` the RSS
  across N frequency bands, predict `[X_{t+1}, ..., X_{t+K}]` `[P:2, Fig. 1]`.
- **Lookback / horizon**: T = 192 steps in; K in {12, 24, 48, 96, 192} out. At 15-min sampling that is
  48 h of history and 1 h to 2 days ahead `[P:6]`.
- **Sampling interval**: 15 min `[P:6]`.
- **Multi-step, direct** — "the output projection module employs a linear layer with an output dimension
  of K, mapping the LLM's output to predict the PSD for K future time slots" `[P:6]`. No recursion.
- **Quantity predicted**: PSD. The released CSVs carry unlabelled values in the range 5.5 - 50.4 with no
  unit in the header `[M]`; the paper calls them PSD measurements and never names the unit `[?]`.
- **Spatial scope**: a single ElectroSense sensor near Madrid. "Channels" are frequency bins of one
  receiver, not separate sites, so there is no spatial dimension at all `[P:6]`.
- **Our benchmark's regime**: 60-min lookback at 1-min sampling, horizons 1/15/60 min, POWDER 600-800 MHz
  across 6 receivers `[O:training/configs/ray_tuning_spectrumllm_2d.yaml]`. This **does not match** the
  paper on sampling interval, lookback, horizon or testbed. The long-horizon, slow-cadence regime in which
  the paper reports its advantage is precisely the regime our benchmark does not test — the single most
  important caveat when quoting their numbers next to ours.

## 2. Dataset used in the paper

| Field | Value | Source |
|---|---|---|
| Testbed / sensor | ElectroSense, one crowd-sourced sensor | `[P:6]` |
| Location | near Madrid, Spain | `[P:6]` |
| Frequency range | 500 - 800 MHz | `[P:6]` |
| Bin width / count | 4-channel: 100 MHz spacing (500/600/700/800); 25-channel: 12.5 MHz; 151-channel: 2 MHz | `[M]` from the CSV headers |
| Time span | 2021-05-28 00:00 to 2021-06-26 21:44, 29.91 days | `[M]` (paper says "one-month period, starting from May 28, 2021" `[P:6]`) |
| Sampling interval | 15 min | `[P:6]`, confirmed `[M]`: 2870 of 2871 gaps are exactly 15 min, one is 14 min |
| Rows | 2872 per file | `[M]` |
| Tensor shape (T x F) | 2872 x 4, 2872 x 25, 2872 x 151 | `[M]` |
| Missing values | none; no duplicate timestamps | `[M]` |
| Value range | 4ch 6.5-50.1 (mean 17.68, sd 16.66); 25ch 5.7-49.9 (mean 12.42, sd 8.29); 151ch 5.5-50.4 | `[M]` |
| Collection hardware | not stated | `[?]` |
| Collection procedure | not stated beyond "ElectroSense open API" | `[?]` |
| Train/test split | 7:3 | `[P:6]` |
| Split is chronological? | **not stated**; our audit assumes chronological, which is the Time-Series-Library default | `[?]` |
| Validation set | not mentioned | `[?]` |
| Public? Where? | Yes — `github.com/LIUC-000/SP`, commit `64b7fa5e96` (2025-07-22), 3 commits total, no license | `[M]` |

**Notes on collection.** ElectroSense sensors are low-cost crowd-sourced SDR receivers; antenna, gain
calibration and whether the values are absolute or uncalibrated receiver power are not stated in the
article `[?]`. The per-channel standard deviations are small and very unequal — 0.34 / 0.48 / 0.64 / 1.64
in the 4-channel train split `[M]` — so the series are close to piecewise-constant levels with occasional
steps, and normalization choice materially changes what an MSE number means (§3).

**A third scenario ships but is never reported.** `spec151_data.csv` (151 channels, 2 MHz bins) is in the
repository `[M]`; the article reports only the 4-channel and 25-channel experiments `[P:6]`.

## 3. Data transformations

The chain as the paper gives it `[P:4]`:

1. Cleaning and structuring of raw measurements (described in prose, no procedure given) `[?]`.
2. Segmentation and train/test division, 7:3 `[P:4,6]`.
3. **RevIN** — reversible instance normalization to zero mean and unit variance, inverted after
   inference so outputs are on the original scale `[P:4]`.
4. **Sliding-window tokenization** — a fixed-length window L across time produces overlapping patches
   `x_i in R^{N x L}`, each treated as one token `[P:4]`. **L is never given** `[?]`.
5. **Learnable embedding** `f_embed` (a "simple MLP", depth/width unspecified) to dimension d `[P:4]` `[?]`.
6. **Prototype alignment** — C prototypes linearly probed from the LLM word-embedding matrix, then
   cross-attention with the patch embeddings as queries `[P:4]`. **C is never given** `[?]`.

- **Normalisation statistics fit on**: RevIN is per-instance, over the lookback window, so it is causal by
  construction. Whether an *additional* dataset-level scaler was fitted on train only is not stated `[?]`;
  the Time-Series-Library default is train-only.
- **Smoothing, interpolation or imputation**: none described, and none needed — the released files have no
  gaps `[M]`.
- **Leakage audit.** No step in the described chain consumes samples after the forecast origin, and no
  statistic is described as crossing the split boundary. This is a materially cleaner formulation than
  TSS-LCD's. But with **no code released**, the audit rests entirely on the prose: it is "none found",
  not "none".
- **Metric space — RESOLVED empirically: raw PSD units.** The article states nothing `[?]`. Its one
  relevant sentence, "the MSE loss is computed between the predicted and actual PSD values" `[P:6]`, sits in
  the fine-tuning paragraph and describes the training objective; Time-LLM applies that same `nn.MSELoss()`
  to standardized tensors, so the wording does not decide it. Reading the three evaluation paths is also
  inconclusive *for this paper*: the Time-Series-Library defaults to `--inverse False`, Time-LLM's
  `run_main.py` never calls `inverse_transform`, and OFA's `test()` scores raw model output — all implying
  standardized, but only if the authors ran them unmodified. Settled by experiment instead: a per-channel least-squares linear probe (a tight proxy for DLinear) reproduces the published DLinear row to 8.5% (4-channel) and 5.8% (25-channel) mean relative error in raw units, against 18.1% and 38.2% in standardized units `[M]`. In
  standardized space the probe would beat published DLinear by ~40% at every 25-channel horizon, which is
  not credible for an equivalent model class. **All comparisons in §13 are in raw PSD units**; an earlier
  revision of this card used standardized units and was wrong.

## 4. What the model needs besides the spectrum

| Requirement | Needed? | How we supply it | Risk |
|---|---|---|---|
| Time-of-day / calendar encoding | No | — | — |
| Sensor coordinates / geometry | No — single receiver, channels are frequency bins | — | none |
| Observation mask | No | — | — |
| Neighbouring-sensor input | No | — | — |
| Fixed number of sites or bins | Yes, implicitly — the patch token spans all N channels, so N fixes the embedding input width | `input_size` from `train_data.shape[-1]` `[O:models/SpectrumLLM.py]` | retraining required per channel count |
| **Pretrained weights** | **Yes — BERT or GPT-2** `[P:6]` | `openai-community/gpt2` from Hugging Face; verified to download inside the container on the A30 node `[M]` | the node needs HF reachability, or a pre-seeded cache |
| **Text prompt / tokenizer** | **Yes** — Prompt-as-Prefix with per-instance statistics `[P:5]` | rebuilt per batch from the template in `[O:models/SpectrumLLM.py]` | a tokenizer call in every forward pass |
| Per-dataset hyperparameter tuning | Not stated `[?]` | our Ray sweep | — |
| External library beyond our core deps | **Yes — `transformers`** | absent from `training/ray/requirements/core.txt`; present in the `spectrum-smoke:latest` image built on chameleon-temp `[M]` | the core Ray image cannot build this model |

## 5. Architecture and capacity

**The paper does not publish a network.** It names the stages and cites Time-LLM `[12]`, OFA `[14]` and
TEST `[11]` as instantiations; the case study then evaluates Time-LLM and OFA as released by their own
authors `[P:5-6]`. Consequently there is no layer table, no widths, no parameter count and no FLOPs
anywhere in the article `[?]`.

- Our build (stage-for-stage after the paper's text; full detail in `training/SpectrumLLM/README.md`):
  RevIN -> patch tokenization -> 2-layer MLP patch embedding -> prototype cross-attention -> prompt
  prefix -> frozen GPT-2 (truncated to `llm_layers`) -> flatten + linear head -> RevIN inverse.
- Parameters at the registry tiny anchor, **pre-realignment build** `[M]`: 134,641,344 total,
  52,728,768 trainable, measured on the A30 in `spectrum-smoke:latest` before the model was merged with
  Time-LLM's geometry. Not re-measured at that anchor since.
- **Parameters of the current build** (T=192, patch 16 / stride 8, 1000 prototypes, `d_ff` 32, GPT-2
  6 layers), measured on the A30 at 4 channels, K=12 `[M]`: **52,685,760 trainable**. 95% of that is
  `mapping_layer` = `vocab_size x num_prototypes` = 50,257 x 1,000 ≈ 50.3M, which is Time-LLM's own
  hardcoded prototype probe `[T:models/TimeLLM.py:178-179]`, not the head. The head is
  `Linear(patches x d_ff, K x channels)`: with 24 patches and `d_ff` 32 that is **36,912** parameters at
  4 channels / K=12 and **3,691,200** at 25 channels / K=192 `[M]`. Against the pre-realignment head
  (23 patches x full `d_llm` 768) the reduction is exactly 23.0x at every (K, channels) `[M]`.
- Parameters at our benchmark's reference config (`ray_tuning_spectrumllm_2d.yaml`, now `num_prototypes:
  1000`, `patch_length/stride: 16/8`, `d_ff: 32`, POWDER): **not yet measured**.
- Parameters in the paper: **not reported** `[?]`.
- **Capacity ratio.** Against the paper's own data: the 7:3 training split is 2010 rows = 8,040 scalars
  (4-channel) or 50,250 scalars (25-channel), giving 1,807 training windows at K=12 `[M]`. A GPT-2-small
  backbone alone is ~124M parameters. Even counting only trainable parameters, the model carries roughly
  four orders of magnitude more free parameters than there are numbers in the 4-channel training set.
  The article does not discuss this.

## 6. Hyperparameters: declared vs actual

There is no hyperparameter table in the article. Everything below marked `not stated` is a hole that must
be filled by guessing before anyone can rerun the case study.

| Parameter | In paper | In their code | We use | Note |
|---|---|---|---|---|
| Lookback T | 192 `[P:6]` | no code | 60 | ours is the benchmark's regime, not theirs |
| Horizon K | 12/24/48/96/192 `[P:6]` | no code | 1/15/60 | different regime |
| Sampling | 15 min `[P:6]` | no code | 1 min | different regime |
| Backbone | BERT and GPT-2 `[P:6]` | no code | `openai-community/gpt2` | ours truncated to 6 layers; truncation is not in the paper |
| Backbone layers used | not stated `[?]` | no code | 6 | |
| Patch length L | **not stated** `[?]` | no code | 12 | the paper's only patching detail is "a fixed-length sliding window of size L" `[P:4]` |
| Patch stride | **not stated** `[?]` | no code | 12 | |
| Prototype count C | **not stated** `[?]` | no code | 500 | Time-LLM's own default is 1000 |
| Embedding dim d | **not stated** `[?]` | no code | 32 | |
| Attention heads | **not stated** `[?]` | no code | 8 | must divide 768 |
| Prompt template | shown as Fig. 3 image `[P:5]` | no code | transcribed from Fig. 3 | the figure is readable; the exact token sequence is not |
| Degree of fine-tuning | **never stated for their own runs.** §IV-D surveys full / partial / complete freezing as options, and its own prose says the objective fine-tunes "both the projection network and the LLM" `[P:5]` — which neither benchmarked model does (error #11b). §V-A says only "the learnable parameters are updated" `[P:6]` | no code | fully frozen | inherited from Time-LLM `[T:...:163-164]`, not from the article |
| Loss | MSE `[P:6]` | no code | MSE | agrees |
| Epochs | **not stated** `[?]` | no code | 20 | |
| Batch size | **not stated** `[?]` | no code | 16 | |
| Learning rate | **not stated** `[?]` | no code | 1e-3 | |
| Optimizer | **not stated** `[?]` | no code | Adam | the word "Adam" does not occur in the article `[M]` |
| Seeds / repeats | **not stated** `[?]` | no code | 5-seed benchmark protocol | no dispersion is reported anywhere in the article |
| Validation set | **not mentioned** `[?]` | no code | 10% of train | the word "validation" does not occur in the article `[M]` |
| Hardware | **not stated** `[?]` | no code | A30 24 GB | no GPU is named; "GPU" does not occur in the article `[M]` |

## 7. Claims, and how they are tested

### Claim 1: "Across both 4-channel and 25-channel prediction tasks, OFA and Time-LLM outperformed all baselines, exhibiting significantly lower MSE and MAE across different prediction steps." `[P:6]`

- **Evidence offered**: Tables I (4-channel) and II (25-channel), MSE and inference time at
  K = 12/24/48/96/192 plus an average column.
- **Baselines compared against**: DLinear, LSTM, LightTS, PatchTST, Informer, Reformer, Autoformer,
  TimesNet — all from `thuml/Time-Series-Library` `[P:5]`.
- **Trivial reference included?** **No.** No persistence, no lookback mean, no seasonal naive, no ARMA
  run — ARMA/AR/MA are surveyed in §III but never evaluated. The words "persistence" and "naive" do not
  occur in the article `[M]`.
- **Metrics and the space they are computed in**: MSE only, space unstated (§3) `[?]`.
- **Seeds / repeated runs / error bars**: none. Every cell is a single number `[M]`. We measured the
  run-to-run spread of a comparable model at a fixed seed on this data at **1.19%** (§13(b)), which exceeds
  five of the seven best/second-best margins the tables assert — so the red/blue markings in those cells
  are not supported by single runs, independently of whether the models differ.
- **Ablations supporting it**: none. There is no ablation of the tokenizer, the prototype alignment, the
  prompt, or the fine-tuning strategy — so the article's own explanation of *why* the LLMs win
  ("prompt-guided feature extraction ... partial parameters fine-tuning ... tokenized spectrum
  embeddings" `[P:6]`) is asserted, not tested.
- **Apples-to-apples?** Partly. All models come from one library and presumably share the data pipeline,
  which is better than most. But no tuning budget is stated for any model `[?]`, and DLinear/PatchTST are
  known to be sensitive to lookback normalization choices.
- **Our assessment: the claim is contradicted by the authors' own tables in six cells** `[M]`:

  | Table | Horizon | LLM model | LLM MSE | Beaten by | MSE |
  |---|---|---|---|---|---|
  | I (4ch) | 12 | Time-LLM (BERT) | 0.3045 | PatchTST | 0.2707 |
  | I (4ch) | 24 | Time-LLM (BERT) | 0.3068 | DLinear | 0.3036 |
  | I (4ch) | 96 | Time-LLM (BERT) | 0.3081 | PatchTST | 0.3061 |
  | I (4ch) | 96 | Time-LLM (GPT2) | 0.3157 | PatchTST | 0.3061 |
  | II (25ch) | 12 | Time-LLM (BERT), Time-LLM (GPT2) | 0.6321, 0.6166 | DLinear | 0.6080 |
  | II (25ch) | 24 | **all three** (0.7564, 0.7592, 0.7593) | | PatchTST | **0.7526** |

  At 24-step/25-channel a non-LLM baseline is the best model in the table, and the paper's own red
  "best" marker sits on PatchTST. The weaker, correct statement — *OFA is best or near-best at most
  horizons, and the LLM family dominates at long horizons in the 25-channel setting* — is supported.
  "Outperformed all baselines" is not.
- **The MAE half of the claim has no evidence at all.** "MAE" appears exactly twice in the article: once
  listing candidate loss functions, once in this sentence `[M]`. No MAE table, figure or number is
  published.

### Claim 2: "non-LLM prediction models ... demonstrated reasonable short-term performance but suffered a notable decline in accuracy as the prediction step increased" `[P:7]`

- **Evidence**: Tables I-II read by eye.
- **Our assessment**: true for DLinear, LightTS, PatchTST and TimesNet; **false for LSTM, Informer and
  Autoformer**, whose errors *fall* or wander as the horizon grows — LSTM 4-channel goes 0.6822 (12-step)
  to 0.5420 (192-step), Informer 0.7377 to 0.6409 `[M]`. Those three are the worst models at short
  horizons, so the generalization is an artifact of which baselines are in the set.

### Claim 3: "although LLM-based models incur slightly higher inference time, the experiment results demonstrate that this trade-off is offset by significantly improved predictive performance" `[P:7]`

- **Evidence**: the Time columns.
- **Our assessment**: "slightly" is not supportable. Average inference time is **22.9x** DLinear in the
  4-channel task and **41.5x** DLinear (124.6x LSTM) in the 25-channel task `[M]`. Whether that is
  acceptable is a deployment question the article does not pose; whether it is "slight" is not an opinion.
  No hardware is named, so the absolute seconds are uninterpretable `[?]`.

### Claim 4 (abstract): "LLMs significantly outperform traditional SP methods such as DLinear, Autoformer, and LSTM" `[P:1]`

- **Our assessment**: DLinear specifically beats both Time-LLM variants at 12-step/25-channel `[M]`, and
  the gap to DLinear in the 4-channel task is 0.3424 -> 0.2987 on average (13%), which is a real but not a
  "significant" improvement in the absence of error bars over five horizons and one seed.

## 8. Errors found in the paper or code

| # | Where | What | Severity | Evidence |
|---|---|---|---|---|
| 1 | §V-B `[P:6]` vs Tables I-II | "outperformed all baselines" contradicted in 6 cells; at 24-step/25-channel the best model in the table is PatchTST | **substantive** | `[M]`, `spectrumllm_published_audit.json` |
| 2 | §V-B `[P:6]` | "significantly lower MSE **and MAE**" — no MAE is reported anywhere in the article | **substantive** | `[M]` |
| 3 | Table II, OFA (GPT2) row | **The Average-MSE cell is blank.** The row mean is 0.8462, which would be the best average in the table and would carry the red marker; as printed, Table II has no best-average entry | **substantive** | `[M]`, rendered p. 198 |
| 4 | Table I, 96-step | Blue "second best" on Time-LLM (BERT) 0.3081, but PatchTST 0.3061 is lower | inconsistency | `[M]` |
| 5 | Table I, 192-step | Blue on Time-LLM (GPT2) 0.3460, but Time-LLM (BERT) 0.3438 is lower | inconsistency | `[M]` |
| 5b | Tables I-II, best/second-best markings | Five of seven top-two margins (0.31-0.65%) are below the 1.19% run-to-run spread we measured at a fixed seed on this data; with single runs and no error bars the rankings in those cells are not established | **substantive** | `[M]` §13(b) |
| 6 | §V-A `[P:6]` / whole article | No epochs, batch size, learning rate, optimizer, seed, validation set or hardware is given — the case study cannot be rerun as specified | **substantive** | `[M]` keyword scan of the full text |
| 7 | §IV-B `[P:4]` | Patch length L, embedding dimension d and prototype count C are all introduced as symbols and never assigned values | **substantive** | `[P:4]` |
| 8 | §V-B `[P:7]` | "non-LLM models suffered a notable decline as the prediction step increased" — false for LSTM, Informer, Autoformer | inconsistency | `[M]` |
| 9 | §V-B `[P:7]` | "slightly higher inference time" for a 23-42x gap | inconsistency | `[M]` |
| 10 | Title / framing vs content | "SpectrumLLM" is named as "a SpectrumLLM architecture" `[P:1]` but no SpectrumLLM model is built, trained or released; the experiments are Time-LLM and OFA run unmodified | **substantive** | `[P:5-6]` |
| 11 | §V-A `[P:6]` vs data repo | The repository ships a 151-channel file that no experiment uses | cosmetic | `[M]` |
| 11b | §IV-D `[P:5]` vs the models actually benchmarked | The text defines the objective as fine-tuning "both the projection network **and the LLM**", asserting that LLM weights are updated. Neither benchmarked model does this: Time-LLM sets `requires_grad=False` on every backbone parameter `[T:models/TimeLLM.py:163-164]`, and OFA updates only names containing `ln` or `wpe` `[F:GPT4TS.py:38-42]`. §V-A then says only that "the learnable parameters are updated", never naming them, so the article never states what its own runs froze | **substantive** | `[P:5-6]`, `[T]`, `[F]` |
| 12 | §II `[P:2]` | Defines binary (occupancy) prediction as one of the two SP families, then never evaluates it; all results are PSD regression | cosmetic | `[P:2,6]` |
| 13 | Data repo | No license, no README beyond 103 bytes, no column units, no provenance for how the 4/25/151-channel files were derived from the ElectroSense API | **substantive** | `[M]` |
| 14 | §V-A `[P:6]` | "one-month period" for a file spanning 29.91 days with one 14-minute gap | cosmetic | `[M]` |

## 9. Availability and reproducibility

- **Code**: none for SpectrumLLM itself. The only code reference in the article is
  `github.com/thuml/Time-Series-Library` `[P:5]`, a generic third-party library containing no SpectrumLLM,
  with no fork, config or commit pinned `[?]`. **But the two models whose numbers the article actually
  reports are fully released**, and we have now pinned and read both:
  `KimMeen/Time-LLM` @ `b13e881f86cd0475ce1b72c17110430663334955` (Apache-2.0, 2025-10-15) and
  `DAMO-DI-ML/NeurIPS2023-One-Fits-All` @ `9f3875dbba9f52ee534ac0902572502746f3b29d` (no license,
  2024-01-08) `[M]`. So several of the article's blanks are recoverable **by proxy** — patch 16/stride 8,
  1000 prototypes, 5 FFT-autocorrelation lags, frozen backbone, 6-layer truncation, OFA's
  layer-norm-plus-positional-embedding tuning set. The full comparison against our port is in
  `training/SpectrumLLM/repro/TIMELLM_OFA_REFERENCE.md`. What is **not** recoverable is which of
  Time-LLM's own contradictory settings the SpectrumLLM authors used: its argparse says lr 1e-4 / 10 epochs,
  its shipped ETTh1 script says lr 0.01-0.02 / 100 epochs `[M]`.
- **Data**: **complete and public.** `github.com/LIUC-000/SP`, commit `64b7fa5e96` (2025-07-22), 767 KB,
  three CSVs + a 103-byte README, downloaded and checksummed `[M]`. This is the *same* data the paper
  used, at the channel counts the paper used.
- **Checkpoints**: none.
- **License**: no LICENSE file in the data repo `[M]`; redistribution status unclear, so we stage the
  files locally and vendor nothing.
- **Dead links**: GitHub repositories (`LIUC-000/SP`, `thuml/Time-Series-Library`) live as of 2026-10-30
  `[M]`. The ElectroSense platform itself was not probed from this sandbox `[?]`.

**Verdict: architecture-only.** The blocker is not data — the data is the best-released of any paper in
this benchmark. The blocker is that **there is no SpectrumLLM to reproduce**: the article's contribution
is a framework description plus a benchmark of two off-the-shelf LLM forecasters, and the numbers in
Tables I-II belong to Time-LLM and OFA, not to a model the authors built. Two different reproduction
targets follow, and they must not be conflated:

1. **Reproduce the case study** — run Time-LLM and OFA from the Time-Series-Library on `spec4`/`spec25`
   at T=192, K=12...192, and see whether Tables I-II come back. Feasible: data and library are public.
   Blocked only by the unstated training hyperparameters (§6).
2. **Reproduce "SpectrumLLM"** — not possible, because no such network was published. Our
   `models/SpectrumLLM.py` is an *implementation of the described paradigm*, and every number it produces
   is ours, not a reproduction of theirs.

## 10. Compute cost

| | Theirs `[P:6]` | Ours `[M]` |
|---|---|---|
| Hardware | **not stated** `[?]` | A30 24 GB (chameleon-temp), `spectrum-smoke:latest` |
| Epochs | **not stated** `[?]` | not run at this regime (§13(b)) |
| Wall-clock to train | **not reported** `[?]` | not run at this regime |
| Inference per sample | 0.0372 - 0.0872 s (OFA / Time-LLM), vs 0.0021 s DLinear and 0.0007 s LSTM `[P:6]` | 0.64 s per forward+backward at batch 32, T=192, 4 channels, on a **shared** A30 `[M]` |
| Peak memory | not reported `[?]` | 2.68 GB at batch 32, T=192, 4 channels `[M]` |

The cost structure here is a frozen-LLM forward pass per sample plus a tokenizer call per batch for the
prompt. That is a fixed, predictable cost — unlike TSS-LCD's 1000-step diffusion chain — but it is 1-2
orders of magnitude above the linear baselines, and the article's timing table has no hardware attached.

## 11. Our implementation

- **Representation fed**: 2D (time x frequency, one receiver) `[O:models/SpectrumLLM.py]`, 356 lines.
- **Residual variant**: none.
- **Built from**: the paper text only — there is no code to port. Prototype reprogramming and
  Prompt-as-Prefix follow Time-LLM, which the paper cites as its instantiation; RevIN follows Kim et al.

**Differences from the published description**, each with a disposition:

| # | Theirs | Ours | Disposition |
|---|---|---|---|
| 1 | No network published | A concrete network: 2-layer MLP embedding, 500-1000 prototypes, 8 heads, d_model 32, 6 GPT-2 layers | **deliberate** — unavoidable; every value is ours and is listed in the README |
| 2 | BERT and GPT-2; depth not stated | GPT-2 only, truncated to `llm_layers: 6` | **resolved** — `llm_layers` default 6 and `num_hidden_layers = llm_layers` is Time-LLM's own code `[T:models/TimeLLM.py:46]`, and OFA slices `gpt2.h[:gpt_layers]` `[F:GPT4TS.py:31]`. Truncation is the instantiations' standard practice, not ours |
| 3 | Patch spans all N channels (`x_i in R^{N x L}`) `[P:4]`; Time-LLM, cited as the instantiation, patches per channel | default follows the paper's literal equation; `channel_independent: true` gives Time-LLM's reading | **deliberate**, both arms available |
| 4 | Fine-tuning strategy left open `[P:5]` | backbone fully frozen | **deliberate** (Time-LLM's choice); OFA's "positional embeddings + layer-norm only" arm is **not implemented** |
| 5 | Prompt template shown as a figure `[P:5]` | transcribed template with the same five statistics (min, max, median, trend, top-5 lags) | **fixed** — Fig. 3 is legible and our template matches its fields |
| 6 | 192-step lookback at 15 min, K up to 192 | 60-step lookback at 1 min, K in {1,15,60} | **deliberate** (benchmark protocol) — but it means we do not test their regime |
| 7 | ElectroSense Madrid | POWDER 600-800 MHz | **deliberate**; their data is now staged locally, so a like-for-like run is possible |
| 8 | — | `backbone: random` untrained control | **addition** — a control for the transfer claim, not a reproduction |

## 12. What we are not sure about

| # | Uncertainty | Why unresolved | Could it change a reported number? | What would settle it |
|---|---|---|---|---|
| 1 | ~~Metric space of Tables I-II~~ | **RESOLVED 2026-10-30: raw PSD units**, by linear probe — a per-channel least-squares linear probe (a tight proxy for DLinear) reproduces the published DLinear row to 8.5% (4-channel) and 5.8% (25-channel) mean relative error in raw units, against 18.1% and 38.2% in standardized units `[M]`. §13(a) was recomputed in raw units and its earlier, stronger conclusion was withdrawn | — | — |
| 2 | ~~Patch length L, prototype count C~~ | **Largely resolved by proxy** — Time-LLM hardcodes 1000 prototypes `[T:...:178]` and defaults to patch 16 / stride 8 `[T:...:32]`, which our port already matched `[M]`. `d_model` is 16 (argparse) or 32 (script) | — | — |
| 3 | Training schedule (epochs, LR, batch, optimizer, seed) | Never stated | **Yes** | Only the authors |
| 4 | Whether the 7:3 split is chronological | Not stated | **Yes** — a shuffled split on 15-min data leaks neighbouring samples | Author reply; our audit assumes chronological |
| 5 | Whether a validation set existed / how early stopping was done | Not mentioned | Moderate | Only the authors |
| 6 | Was any dataset-level scaler fitted on train only | Not stated | Moderate | Library default suggests yes |
| 7 | Hardware behind the Time column | Not stated | No, for accuracy; **yes** for the deployability argument | Author reply |
| 8 | ~~Whether BERT/GPT-2 are used at full depth~~ | **Resolved** — both instantiations truncate, Time-LLM to 6 layers by default `[T:...:46]` | — | — |
| 11 | **Our port is not a faithful Time-LLM** (new) | Head geometry (full `d_llm` flatten vs their `d_ff` truncation), reprogramming widths, `affine=True`, patch count, Conv1d-vs-MLP patch embedding, and a lag statistic computed over raw values instead of FFT autocorrelation all differ `[M]` | **Yes** — §13(b) measures our build, and must not be read as "Time-LLM on ElectroSense" | Port their geometry verbatim and rerun; `TIMELLM_OFA_REFERENCE.md` §1 is the checklist |
| 9 | Our truncation to 6 layers and frozen backbone | Our choice, not theirs | **Yes** | Ablate: 6 vs 12 layers, frozen vs OFA-style partial tuning |
| 10 | Provenance of `spec4` / `spec25` / `spec151` from the ElectroSense API | No script in the repo | Low for reproduction, high for extending to new bands | Author reply or ElectroSense API docs |

## 13. Reproduction results

### (a) The published numbers against trivial references, on the authors' own data `[M]`

Protocol: their data, chronological 7:3, lookback 192, test windows beginning at `n - n_test - 192`,
MSE in **raw PSD units** (§3). Full tables in `training/SpectrumLLM/repro/RESULTS_PAPER_REGIME.md`.

| | K=12 | K=24 | K=48 | K=96 | K=192 |
|---|---|---|---|---|---|
| **4-channel**: strongest trivial reference | 0.4474 | 0.4472 | 0.4466 | 0.4473 | 0.4503 |
| how many of the 11 published rows it beats | 3 | 4 | 4 | 5 | 5 |
| best published cell | 0.2621 | 0.2892 | 0.2911 | 0.2958 | 0.3427 |
| **25-channel**: strongest trivial reference | 0.8076 | 1.2306 | 1.2495 | 1.2364 | 1.2071 |
| how many of the 11 published rows it beats | 5 | 4 | 4 | 7 | 7 |
| best published cell | 0.5846 | 0.7526 | 0.8564 | 0.9540 | 1.0311 |

The 24-hour seasonal naive is the strongest reference in nine of the ten cells; last-value persistence
leads only at 25-channel K=12.

**What this supports.** The article's headline comparison survives a trivial-reference test it never ran:
Time-LLM (BERT), Time-LLM (GPT2) and OFA beat every trivial reference at every horizon in both scenarios.
**What it still shows.** Between 3 and 7 of the article's own 11 rows lose to a predictor that requires no
training — LSTM, Informer, Reformer, Autoformer and TimesNet throughout, joined by DLinear and LightTS at
the longer 25-channel horizons. A baseline set in which half the entries are worse than a seasonal naive
does not establish much about the half that are better, which is the argument the card makes in §7 rather
than the stronger one an earlier revision made here.

### (b) Our implementation on the paper's data, 4-channel K=12 `[M]`

The merged build (article-first, Time-LLM where the article is silent), trained **on raw PSD** with RevIN
normalising per instance inside the model — the arrangement the article describes — and scored in PSD
units. 32/32 verification checks and 10/10 unit tests pass (`repro/verification.json`).

| Predictor | MSE (raw PSD^2), 4-channel K=12 |
|---|---|
| Time-LLM (GPT2), published | 0.2621 |
| OFA (GPT2), published | 0.2686 |
| PatchTST, published | 0.2707 |
| DLinear, published | 0.2718 |
| **Ours, seed 0** | **0.2749** |
| Time-LLM (BERT), published | 0.3045 |
| Seasonal naive 24 h | 0.4474 |
| Last-value persistence | 0.5638 |
| Lookback mean (L=192) | 0.7076 |

**The port reaches the published range**: 1.049x the best published cell, 1.024x OFA, 1.012x DLinear, and
better than Time-LLM (BERT), LightTS and TimesNet. It clears all three trivial references.

**Run-to-run nondeterminism, measured.** The same configuration at the same seed was run twice and scored
**0.2749** and **0.2782** — a spread of 0.0033, or **1.19%** `[M]`. The seed fixes initialisation and batch
order; the residual comes from nondeterministic GPU reductions. This has a direct bearing on §7: **five of
the seven best/second-best margins in the article's own tables are smaller than this spread** — Table I
K=12 PatchTST vs DLinear (0.41%), K=96 PatchTST vs Time-LLM BERT (0.65%), K=192 Time-LLM BERT vs GPT2
(0.64%), Table II K=24 PatchTST vs Time-LLM BERT (0.50%), and the Table II average (0.31%). The article
reports single runs with no seeds or error bars, so those orderings are not resolvable from the evidence
given. Only the larger gaps survive — Table I K=12 Time-LLM GPT2 vs OFA (2.45%) and the Table I average
(2.90%).

Qualifications, in order of weight: one seed, one horizon, one dataset, no dispersion; hyperparameters are
Time-LLM's argparse defaults (lr 1e-4, 10 epochs, batch 32), untuned, because the article states none and
Time-LLM's own script disagrees with its argparse by 100x on learning rate; validation loss was still
non-monotone at the 10-epoch budget, so this is not a converged run; and the GPU was shared, so the 274 s
wall-clock is not a clean timing. The earlier 30-run sweep at this regime is withdrawn (wrong geometry and
wrong metric space) and no number from it is carried here.

### (c) Our implementation, smoke build: **passes** `[M]`

On the A30 in `spectrum-smoke:latest`, `spectrumllm` builds at its registry tiny anchor, does a forward
pass and one Adam step in 8.59 s, with 134,641,344 parameters of which 52,728,768 are trainable
(pre-realignment build) — one of
15 models passing out of 25 in that sweep (`smoke_all_models.json`). An earlier CPU smoke on a 420x4
deterministic sinusoid (lookback 32, horizon 4, 40 epochs) reached val MSE 0.0214 against persistence
0.0459 and lookback mean 0.8626, i.e. the training path works end to end; the untrained-backbone control
reached 0.0080 on the same task, **better than the pretrained backbone** `[M]`
(`smoke_spectrumllm_summary.json`). On a deterministic sinusoid that is not evidence against transfer,
but the control must be rerun on real data before any transfer claim is made.

## 14. Open questions and next actions

- [ ] **Settle the metric space** (uncertainty #1) by rerunning one Time-Series-Library baseline
      (DLinear is cheapest) on `spec4_data.csv` at T=192 and checking which space reproduces 0.2718
- [x] **Done 2026-10-30** — our port trained on `spec4`/`spec25` at T=192, K=12/24/48/96/192, 3 seeds, A30; tabled against persistence / lookback mean / seasonal naive and against Tables I-II (§13(b))
- [x] **Done 2026-10-30** — parameter counts, wall-clock, inference time and peak memory measured at the paper regime (§5, §10). The POWDER reference config is still unmeasured
- [ ] Run the `backbone: random` control on `spec4`/`spec25` — the transfer claim is the paper's central
      premise, it has never been controlled for, and the runner already takes `--backbone random`
- [ ] Tune before concluding: patch length, prototype count, LR and `llm_layers` are all untuned, and
      early stopping fires at 6-15 epochs (§13(b))
- [ ] Fix the prompt's "top 5 lags": ours ranks raw sample values, Time-LLM ranks FFT autocorrelation
      `[T:models/TimeLLM.py:257-264]` — a one-function change
- [ ] Decide whether to align our port to Time-LLM's released geometry (`d_ff` head truncation,
      `affine=False`, padded patch count, Conv1d patch embedding) or keep the current build as the
      article-literal arm and add the faithful one beside it
- [ ] Add the OFA arm: `requires_grad=True` on parameters whose name contains `ln` or `wpe`, no prototypes,
      no prompt `[F:GPT4TS.py:38-42]` — it is the best average performer in both of the article's tables
- [ ] Ablate `channel_independent` true/false — the paper's equation and its cited instantiation disagree
- [ ] Add `transformers` to `training/ray/requirements/core.txt` so the core Ray image can build this model
- [ ] Email the authors for: patch length, prototype count, embedding dim, training schedule, hardware,
      and whether the split is chronological

## 15. Evidence provenance

- **Paper**: Zotero `5ZG38NKM`, PDF attachment `JWQCQQUX`, 7 pages, journal pages 193-199,
  IEEE Wireless Communications vol. 33 no. 2 (April 2026); DOI `10.1109/MWC.2025.3596934`; published
  online 2025-10-17. Tables I-II read from the rendered page 198 as well as the text layer; both agree,
  and every published row average was recomputed from its five horizon cells `[M]`.
- **Data**: `github.com/LIUC-000/SP` @ `64b7fa5e96` (2025-07-22), downloaded 2026-10-30. sha256 (first 16
  hex): `spec4_data.csv` d4b0e31f03433e85, `spec25_data.csv` 4c884030643ffe4b,
  `spec151_data.csv` 5317f35687555637. 2872 x {4,25,151}, no NaNs, no duplicate timestamps `[M]`.
- **Code read**: none exists for SpectrumLLM. The two instantiations were read at pinned commits on
  2026-10-30: `KimMeen/Time-LLM` @ `b13e881f86` (`models/TimeLLM.py`, `run_main.py`, `layers/Embed.py`,
  `scripts/TimeLLM_ETTh1.sh`) and `DAMO-DI-ML/NeurIPS2023-One-Fits-All` @ `9f3875db`
  (`Long-term_Forecasting/models/GPT4TS.py`, `main.py`). `thuml/Time-Series-Library` was not pinned by the
  authors and is not vendored here.
- **Our code**: `models/SpectrumLLM.py`, `tests/test_spectrum_llm.py` (10 tests),
  `training/configs/ray_tuning_spectrumllm_2d.yaml`.
- **Trained weights**: `ssh:chameleon-temp`, `/home/cc/.claude-science-scratch/spectrumllm_paper_regime/
  checkpoints/spec4_K12_seed0_raw.pt` (210.7 MB, sha256 prefix `443103c2d9f054ac`). Trainable tensors only
  (14 tensors, 52,685,760 parameters; the frozen backbone is reconstructible from `backbone` +
  `llm_layers`), with the config and the normalisation statistics. Verified: reloading into a fresh model
  reproduces its recorded test MSE exactly `[M]`. Earlier runs saved no weights at all — that gap is fixed
  in `run_paper_regime.py --checkpoint-dir`.
- **Our runs this card draws on**: `smoke_all_models.json` (`spectrumllm` row, A30),
  `smoke_spectrumllm_summary.json` (CPU toy task), `spectrumllm_published_audit.json` (this card's table
  audit and trivial-reference scoring).
- **Reproduction run**: `ssh:chameleon-temp` (A30, `ray-worker-02`), image `spectrum-smoke:latest`,
  2026-10-30; 30 runs; runner, raw results and tables under `training/SpectrumLLM/repro/`.
- **Card author / date**: automated audit, 2026-10-30, revised the same day with the A30 results.
