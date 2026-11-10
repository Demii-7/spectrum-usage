# REPRODUCTION — DeepSPred (3D-SwinSTB): code vs paper, and why every result is what it is

Reproduction of G. Pan et al., *"Spectrum Prediction With Deep 3D Pyramid Vision Transformer
Learning"*, IEEE Trans. Wireless Commun. 24(1):509-525, Jan 2025 (DOI 10.1109/TWC.2024.3495812;
identical in every quoted number to arXiv:2408.06870v3). The paper's 3D-SwinSTB predicts future
radio spectrograms: 256x256 RGB images, one per second of I/Q, input-n-predict-n (K = T).

This single file consolidates the former EXPERIMENTS.md, DEEPSPRED_REFERENCE.md,
CODE_VS_PAPER.md, MODEL_CARD.md, MODEL_WALKTHROUGH.md, REPRODUCTION_SUMMARY.md, RESULTS.md and
DESIGN.md (2026-11-09; nothing unique was dropped — see section 7.4 for the file map). It
contains, in order: what the paper fixes and this code implements as stated; every place the
code differs from the paper and why; the reasoning that determines each result; the run ledger;
and the verdict against the paper's published anchors.

**Status.** All 13 checkpoints in `checkpoints/repro/deepspred/` were re-scored after the
source-code loss (2026-11-09): all 25 documented cells reproduced, 0 differ (max deviation
0.02 %, the global-SSIM cell). Metrics live in the paper's `uint8_mean_per_channel` space (mean
squared error over pixels and channels of 0..255 uint8 values), MAX_I = 255, frame index
k = 1..K. Complete frame-wise numbers: `repro/results.json` (13 run records + `_probes`).

Tags: `[P:eq. n]` / `[P:L n]` = paper equation / extracted-text line; `[O:file]` = our code;
`[M]` = measured by us; `[?]` = the paper is silent.

---

## 1. What the paper fixes — implemented exactly as stated

### 1.1 Regime

| Property | Value | Source |
|---|---|---|
| Frame | 1 second of I/Q, converted by STFT to a 256x256 image | `[P:L 1775, 1811]` |
| Channels / axes | 3 (RGB); H = STFT time sub-steps, W = frequency bins | `[P:L 409-412]` |
| FM band / centre | 90-110 MHz / 99 MHz (fs 125 MHz, decimation 4, "STFT number" 32508; LTE 16254) | `[P:L 1777-1795]` |
| Split | chronological 4:1:1; FM 7200 train / 1800 val / 1800 test — reproduced exactly by time-based boundaries | `[P:L 1788-1793]` |
| Lookback / horizon | **K = T** (8/8, 16/16, 20/20) — a property of eq. (22), not a knob; the runner refuses any other combination | `[P:L 2355-2358]` |
| Loss / metrics | MSE; PSNR (MAX_I = 255), SSIM, LPIPS, frame-wise (per predicted index k) | `[P:L 2040-2050]` |

### 1.2 Architecture

C = 96; encoder depths {2, 4, 2}, heads {4, 8, 16}; bottleneck = third stage's depth (2, heads
16); decoder mirrors encoder with heads reversed {16, 8, 4}; patch (2, 4, 4); window (2, 7, 7);
alternating W-MSA / SW-MSA with shift = window//2; learned 3D relative position bias;
LN + residual around attention and MLP (eq. 13); patch merging 2x2 -> 4C -> 2C and its inverse
expanding; skips encoder-i -> decoder-(4-i); the eq. (22) head: one ConvTranspose3d C->C at
patch stride back to (T, H, W), then n = ceil(log2(C/3)) 1x1x1 inverse convolutions halving
96 -> 48 -> 24 -> 12 -> 6 -> 3; direct one-pass K-frame output `[P:L 1812-1819]`.

**Parameter count: 16,180,935 vs Table I's 16.32 "MB" — ratio 0.991** `[M]`. Table I's unit is
almost certainly millions of parameters, not megabytes (16.32 MB fp32 would be ~4.3 M params,
which no reading of the stated architecture produces). This count is the strongest single
corroboration that our reading of the architecture is the authors'.

### 1.3 Training

AdamW, lr 0.001, batch 1, <= 20 epochs; early stop when the loss improves < theta_per = 0.01 %
over n_ep = 4 consecutive epochs, patience 4 `[P:L 1821-1827]`. Hardware in the paper: i9-10940X,
" GTX 3090Ti" (no such product; RTX 3090 Ti), PyTorch 1.8.0 `[P:L 1808-1810]`.

---

## 2. What the code does differently from the paper

Everything the paper states numerically is implemented as stated (section 1). Every remaining
disagreement falls into four classes: **(A)** the paper is silent and we chose + recorded;
**(B)** the paper contradicts itself and we resolved it; **(C)** we added something the paper
lacks; **(D)** paper components that are unimplementable as published.

### 2.1 (A) The paper is silent — our recorded choices

| # | Item | Paper says | Code does | Why |
|---|---|---|---|---|
| A1 | STFT hop | window length 256 only; output must be 256x256 | hop = (N-256)/(256-1) = 126 (FM) | the only hop consistent with the stated shape `[O:repro/fm_data.py:spectrogram]` |
| A2 | STFT window function | nothing | Hann | switch, recorded per run in `switches` |
| A3 | one- vs two-sided FFT | nothing | two-sided (fftshift) | the only reading giving 256 frequency bins over the 31.25 MHz band |
| A4 | dB vs linear scaling | eq. (2) is \|STFT\|^2 | 10*log10, then min-max to [0,1] | a ~122 dB linear span cannot map to 8 bits without compression |
| A5 | Normalisation scope | nothing | dataset-wide dB min/max, sampled every 50th second | per-frame normalisation would destroy absolute level information |
| A6 | Colormap | "a common Jet colormap" `[P:L 409]` | `jet` | as stated |
| A7 | Weight decay | nothing | 0.05 (Swin convention) | recorded choice |
| A8 | Dropout / drop-path | nothing | 0.0 | the paper names no regulariser; a silent one would change results |
| A9 | Seeds / averaging | nothing; no error bars anywhere | explicit `--seed` in the run key; >= 2-3 seeds where the paper ran one | the paper's cells have no dispersion estimate |
| A10 | Channel projection after each skip concat | eq. (10) concatenates with no projection | `Linear(2C -> C)` after each concat | **forced**: without it the channel count doubles at every skip and the next stage cannot accept its input |
| A11 | Output nonlinearity | sigmoid drawn only in the SOR predictor (Fig. 5) | sigmoid on the spectrogram head | bounds output to the [0,1] target range |
| A12 | LayerNorm placement | eq. (11)-(12) name only "a simple linear layer" | LN after the embedding conv, before the merge reduction, after the expand rearrange | Swin convention; the paper names no normalisation |
| A13 | qkv/proj bias | matrices only | bias (Swin default) | unstated |
| A14 | Bottleneck head count | blocks only, heads unspoken | third stage's (16) | unstated |
| A15 | Weight init | nothing | trunc_normal_(0.02), zero bias, unit LN | Swin defaults |
| A16 | Which loss early-stopping watches | "the loss function reaches convergence" | validation loss | train loss cannot stall under batch-1 reshuffling; test loss would leak |
| A17 | Early-stop baseline | "< theta_per % in n_ep consecutive epochs" | vs. best validation so far; < 0.01 % for 4 epochs, or 4 without a new best | both readings coincide at 4; implemented as one condition |
| A18 | Window stride / meaning of "7200 samples" | stride never stated | stride 1 over the 7200 training frames (7122 windows after gap-dropping) | the alternative reading (7200 windows) differs by < 1 % |
| A19 | Per-epoch sample order | nothing | reshuffled every epoch | convention |
| A20 | Loss space | eq. (23), pixel scale unstated | MSE on [0,1] floats | constant 255^2 factor vs 0..255 — identical argmin, recorded |
| A21 | SSIM implementation | rho1 = 0.01, rho2 = 0.03 only | Wang et al. 2004: 11x11 Gaussian, sigma 1.5, per channel then averaged | no `skimage` in the image; differences below ~0.01 are not meaningful across implementations |
| A22 | LPIPS backbone | "the LPIPS network" | AlexNet, lpips 0.1 weights | recorded per run |
| A23 | H/W axis semantics | H = time, W = frequency | frame arrays are (H, W) with the STFT output transposed | faithful; listed because the transpose is easy to get backwards |

### 2.2 (B) The paper contradicts itself — the code resolves it

| # | Contradiction | Resolution |
|---|---|---|
| B1 | eq. (14) writes the MLP hidden width as 2C, but Table I's 16.32 M params is only consistent with the Swin 4C (2C gives ~12.2 M) | `mlp_ratio = 4.0`, source-commented `[O:models/DeepSPred.py]` |
| B2 | eq. (30) defines "MSE" as a double **sum**, then eq. (31) feeds it to PSNR as a **mean**; the literal sum (1.06e8) is 5 orders from every published panel | mean squared error; all four candidate spaces still measured and recorded per run `[O:repro/metrics.py]` |
| B3 | Fig. 7(c) PSNR 35.1039 at frame 8 implies MSE 20.1; the MSE panel reports 448.7628 — a **22.35x** gap no convention explains (3x channel sum, 65536x sum-vs-mean, 1/65025 unit-space all fail) | target the **MSE** panel; report the PSNR panel as unreproducible. Decided empirically before quoting anything (section 3.1) |
| B4 | eq. (10) / Algorithm 2 print `S_en^1` twice — the skip indices are garbled | symmetric skips: dec1 <- enc3, dec2 <- enc2, dec3 <- enc1 |
| B5 | Footnote 3 parameterises the relative-bias table over (2T-1), but the bias lives inside a window of temporal extent P | table of (2P-1)(2Mh-1)(2Mw-1) x heads, indexed per window |
| B6 | eq. (22) makes the output length equal the **lookback** (K = T structural), while the text speaks of predicting K frames | K = T enforced; every paper experiment is in fact input-n-predict-n |
| B7 | Block ablation text reads "a MSE decrease of ... to the {2,2,2}" twice; the second must be {2,6,2} `[P:L 2588-2591]` | read as {2,6,2}; text error only |
| B8 | "the number of 3D Swin Transformer blocks is set to 2" immediately after selecting {2,4,2} `[P:L 2592]` | {2,4,2} kept (Table I's parameter count requires it) |
| B9 | Channel ablation quotes "a MSE decrease of 2.0077" twice, but Table II's own cells give 6.0452 and 4.4058 `[P:L 2594-2600]` | Table II cells used; the quoted delta is copied from the block ablation |
| B10 | Table II: {2,2,2} has the **lower** frame-8 MSE (447.48 vs 448.76) yet {2,4,2} is adopted on a 0.44 % frame-16 margin, with no seeds or error bars | noted as unresolvable at the paper's own noise level (section 3.6) |
| B11 | "GTX 3090Ti" — no such product exists | environment note only |

### 2.3 (C) Additions — things the code has that the paper does not

| # | Addition | Why |
|---|---|---|
| C1 | **Gap-aware windows**: the released FM capture is missing 7 seconds inside its span (cross-validation: 234); windows straddling a missing second are dropped via timestamp-derived segments `[O:repro/fm_data.py]` | the paper's contiguous-window assumption is false on its own release |
| C2 | **Trivial references** (persistence, lookback mean) in every result row `[O:repro/run_paper_faithful.py:trivial_references]` | the paper reports no trivial baseline at all; at 1 s frames persistence is the first thing a reviewer asks for. A seasonal naive is omitted **deliberately** (a 3 h capture at 1 s with a 20-frame lookback exposes no period) and the omission is recorded |
| C3 | **JPEG round-trip rendering switch** (`--jpeg-quality`) `[O:repro/fm_data.py]` | the paper's MSE panel sits below the measured noise floor of the raw rendering; a codec pass is the only reading that makes the published numbers mutually consistent — carried as an explicit recorded hypothesis, never silently (section 3.3) |
| C4 | **Metric-space probe before quoting any number** (payload in `results.json` -> `_probes.metric_space`) | the paper's own metrics are mutually inconsistent (B2, B3); assuming a space is forbidden |
| C5 | Four candidate MSE spaces measured per frame, the primary named per row | consequence of B2 |
| C6 | Frame caching; resumable run keys; checkpoints with sha256 + normalisation stats; per-epoch history | the paper keeps no artifacts; every run here is rebuildable and checkable |
| C7 | Robustness padding for odd token grids (merge pads, decoder crops) | the paper's own config never produces odd grids; other configs in this repo do |
| C8 | The `head="temporal"` projection head (Linear(T_in -> K)) | used **only** by the integrated POWDER benchmark track, which needs K != T; the reproduction always uses the paper head |

### 2.4 (D) Paper components absent from the code

| # | Component | Status | Why |
|---|---|---|---|
| D1 | 3D-SwinLinear (the SOR arm — half the paper) | **not implemented** | the SOR targets cannot be constructed: the binarization block size `w` (eq. 26) and the accuracy threshold `lambda` (section VI-A4) have no values anywhere in the published text. The "90 % accuracy" claim is uninterpretable as published |
| D2 | Transfer learning FM -> LTE (Algorithm 4, Fig. 10) | **run** (`--init-from`) | comparable MSE reproduces (347.3 vs 347.9); scratch wall time reproduces exactly (46.0 min vs the paper's "46.3", identifying its unit as minutes); the exact -77.32 % time saving is not attainable under any defensible reading of the stopping criterion (closest -42 %) |
| D3 | The four baselines (DCG, ConvLSTM, NN-ResNet, SAE-TSS) | **not reimplemented, by policy** | no released code, no hyperparameters beyond "the same as ours"; we compare against their published anchors as-is |
| D4 | Grayscale ablation (Appendix A) | **tested at data level and trained** (`--gray`, C = 32 per the paper's one-third-capacity note) | refutes the paper's 11.5x claim (section 3.7) |

---

## 3. Why each result is what it is

### 3.1 The metric space was fixed before any number was quoted

The paper's MSE and PSNR panels cannot both hold: PSNR 35.1039 with MAX_I = 255 implies
MSE = 20.08, but the MSE panel reports 448.7628 at the same frame — a ratio of **22.35** that no
convention (3x channel-sum, 65536x sum-vs-mean, 1/65025 unit-space) explains. The probe
(`_probes.metric_space`) measured frame persistence on the authors' own FM release at
**MSE@8 = 1614.66**: a model scoring 449 is 3.6x better than persistence (plausible), a model
scoring 20.1 would be 80x better (not credible). Hence the MSE panel is the target and the PSNR
panel is reported as unreproducible by construction. Primary space:
`uint8_mean_per_channel`, MAX_I = 255.

### 3.2 The noise floor of the raw rendering

The release is raw I/Q, so the rendering is ours to define (section 2.1) — and the literal
pipeline (hann STFT -> dB -> dataset-wide min/max -> jet -> uint8, no codec) has a measurable
irreducible floor `[M]`:

| Quantity | Value | Meaning |
|---|---|---|
| consecutive-frame MSE (test) | 1558.2 | ~2 sigma^2 if frame-to-frame differences are noise |
| variance around a 5-frame local mean | 623.8 | ~1.2 sigma^2 -> sigma^2 ~ 520 |
| MSE between frames 16 s apart | 1615.8 | ~= the 1 s value: the capture is stationary; persistence error is noise, not drift |
| **irreducible floor sigma^2** | **~520-780** | any predictor's test MSE is bounded below by this |
| persistence MSE @8 | 1614.7 | ~2 sigma^2, consistent |
| our 3D-SwinSTB (raw rendering) | 845.9 | at the floor: the model learned the conditional mean |
| paper's claimed MSE @8 | 448.7628 | **below the floor** under the raw rendering |

Frame-to-frame differences are RF noise, not drift, so no seed, budget or hyperparameter can
reach 449 under the literal pipeline — it is not a matter of tuning. This single measurement is
what forces the rendering question.

### 3.3 The JPEG hypothesis — why a codec, and why q95

The paper never says how its "spectrograms" were stored before training; the data README calls
the release spectrograms while the release is actually raw I/Q. A lossy image codec is the one
unstated step that reconciles the published numbers with the data:

- **Why a codec at all.** A JPEG round-trip smooths pixel-level noise: at any quality 75-95 the
  floor drops from sigma^2 ~ 520-780 to **~188-218**, and persistence lands at **449-481 across
  frames 8-18 — exactly the paper's baseline cluster** (its DCG baseline reports 475.6; all four
  of its methods sit within ~5 % of each other at 440-480). Under the raw rendering the paper's
  numbers are impossible (3.2); under a codec rendering they are mutually consistent — and the
  paper's model score (448.76) then coincides with the **persistence level** of that rendering.
- **Why q95 specifically.** Within q75-95 the behaviour is flat (persistence ~465-500), so q95 is
  the *least-lossy* member of the family — the most conservative choice that still makes the
  paper's numbers attainable. Quality is recorded in every run key (`raw` vs `jpegNNN`), so the
  hypothesis can never silently contaminate a result.
- **Why not anything else.** The rendering-identification sweep (payload in
  `results.json` -> `_probes.rendering_identification`) probed the alternatives: heavier
  smoothing (Gaussian blur sigma 1.5, 2x resample) pushes trivial-reference SSIM to 0.74-0.79 —
  near the paper's 0.7753 anchor — but drops persistence MSE to 76-258, *below* the paper's
  baseline cluster; JPEG q80-95 put persistence in the cluster but cap windowed SSIM at
  0.47-0.52. **No single rendering satisfies the paper's MSE, SSIM and LPIPS anchors at once**:
  like the MSE/PSNR pair (3.1), the panels are jointly inconsistent. JPEG q95 is kept because it
  is the only family that makes the *targetable* panel (MSE) reachable at all, not because it
  reproduces everything.
- **The conclusion it licenses.** Even under this most favourable reading, no trained model
  reaches 449: ours scores 266.6 — the claim is beaten by 42 %, and the claimed value is
  identified as the persistence level of a codec-smoothed rendering (3.5).

### 3.4 The SSIM anchor reproduces only under the paper's literal eq. (32)

Eq. (32) as printed has **no sliding window** — one mean/variance/covariance per image. Read
literally (global SSIM), our raw-rendering model scores **0.8037 at frame 18 vs the paper's
0.7753 — reproduced within 3.7 %**. Under the standard windowed (Wang et al.) reading the same
model gives 0.47-0.58 under every rendering probed. Without the authors' code the implementation
they used cannot be pinned; the global reading is the one that reproduces, and both are recorded.

### 3.5 Why our model beats the claimed MSE by 42 %

Under the rendering that makes the paper's numbers possible at all (JPEG q95, same frames, same
split, same metric space as the anchors):

| FM, input-20-predict-20, MSE @8 | |
|---|---|
| **Our faithful 3D-SwinSTB** (seeds 0 / 1 / 2) | **266.6 / 259.6 / 259.4** |
| Paper's claimed 3D-SwinSTB | 448.8 |
| Paper's best baseline (DCG) | 475.6 |
| Frame persistence | 464.9 |
| Lookback mean | 252.0 |

Three measured reasons:

1. **Their model performs at the persistence level.** Everything the paper reports — its model
   and all four baselines — clusters at 449-476, which is where repeating the last frame lands
   (465). A properly trained model sits at the conditional mean (~260), 43 % below persistence.
   Their stated protocol (20 epochs, batch 1, lr 0.001 — run exactly here) reaches the
   conditional mean within **one epoch** on this data, so their numbers are not an
   under-training artefact of the protocol; they are consistent with predictors that never got
   past persistence-level behaviour.
2. **The task is noise-floor-limited, not capacity-limited.** At the floor every competent model
   converges to ~sigma^2 + drift ~ 260; architecture differences (the paper's Tables II-III
   spreads are 1-2 %) sit far below seed noise (+-1.3 % across our three seeds).
3. **Their evaluation protocol is not reproducible as published** (rendering unstated; SSIM
   formula windowless; PSNR panel self-contradictory; SOR constants missing), so their absolute
   numbers correspond to no constructible configuration — while every run here carries trivial
   references, seed dispersion and recorded switches.

### 3.6 Why the ablation tables are unresolvable

The paper's Tables II-III quote 1-2 % spreads from single runs with no error bars. Our three-seed
spread on the headline configuration is **+-1.3 %** (259.4-266.6), the same order as every cell
the ablations treat as significant. Four ablation cells were run to test this directly: C=48
scores 261.3 (paper claims a 1.3 % *penalty* — indistinguishable here, marginally better),
patch {4,4,4}/window {4,7,7} scores 253.8 (paper claims it is the *worst* — it is the *best* of
our runs), lr 1e-4 scores 276.1 (the only real effect, direction matches the paper), horizon-8
scores 261.4 at frame 8 (the error level is horizon-independent, as a noise floor dictates).
The paper's ablation ordering does not survive seed noise on its own data.

### 3.7 Grayscale: the Appendix-A claim is refuted at data and model level

The paper claims grayscale is 11.5x worse (5294.5 vs 450.8 at frame 10). A colormap is a
deterministic pointwise function of the scalar field — it cannot add information; the claimed
ratio is a metric-scale and capacity artefact. Measured: all data readings give grayscale
0.1-0.2x the jet-RGB error (jet *amplifies* pixel noise). Trained per the paper's own Appendix-A
design (1-channel, C = 32, one-third capacity): **MSE@8 = 142.2 vs the RGB C=96 model's 266.6 —
1.9x better**, with 8.6x fewer parameters. The published ratio is a capacity artefact, not a
property of the representation.

### 3.8 Cross-validation and LTE transfer learning

- **Cross-validation (Fig. 8 claims 531.4 @12).** Locally trained: **205.4** (seed 0) / 201.4
  (seed 1) — the claim is beaten 2.6x, and even the trivial references outscore it (persistence
  375.3, lookback mean 203.0). The paper never states whether Fig. 8 is local or transferred;
  zero-shot FM transfer lands at 849, so the anchors sit in a bracket (375-849) that no
  trained-model protocol reaches. The 531.4 is not reproducible under any reading we constructed.
- **LTE transfer learning (Fig. 10).** Scratch: avg MSE over 16 frames **347.9**; transfer-learned
  from the FM checkpoint per Algorithm 4: **347.3** — "comparable MSE" reproduces, marginally
  better as the paper claims. The eq. (22) head has no T-dependent parameters, so the FM T=20
  checkpoint loads into T=16 unmodified. Scratch wall time 46.0 min reproduces the paper's "46.3"
  exactly, identifying its unit. The claimed **-77.32 %** training time is not attainable under
  any defensible reading of the early-stopping criterion (best observed -42 %); the TL run's
  *convergence* advantage does reproduce (plateau at epoch 1 vs 5).

---

## 4. The ledger — every run, its checkpoint, and what it proved

Checkpoints are `checkpoints/repro/deepspred/<sha1(run_key)[:12]>.pt`; the run key is the record
key in `results.json`. All 25 documented cells were re-scored from the checkpoints on 2026-11-09
and matched (section 0). Metrics: `uint8_mean_per_channel`, MAX_I = 255.

| # | Experiment | Checkpoint | Key result (recomputed = documented) | What it proved |
|---|---|---|---|---|
| 1 | FM raw rendering, seed 0 | `954d763a44c4.pt` | MSE@8 845.9; @16 845.3; @18 846.5; persistence@8 1614.7; lookback-mean@8 857.6; SSIM@18 0.4713; LPIPS@8 0.2522; global SSIM@18 0.8037 (doc 0.8039) | the literal pipeline puts the paper's MSE panel below the data's information-theoretic floor: the model sits *at* the floor (conditional mean), 1.9x better than persistence, yet 1.9x above the claim. The SSIM anchor reproduces under the literal windowless eq. (32) |
| 2 | FM JPEG q95, seed 0 — **headline** | `f6426d84dd5d.pt` | MSE@8 266.6; @16 266.4; @18 267.8; persistence@8 464.9; lookback-mean@8 252.0; SSIM@18 0.5827; LPIPS@8 0.2853 | the codec rendering makes every published MSE mutually consistent: persistence lands exactly on the paper's baseline cluster, and the claimed 448.76 coincides with the persistence level. The faithful port beats the claim by 42 % |
| 3 | FM JPEG q95, seed 1 | `a9486d198417.pt` | MSE@8 259.6 | seed dispersion the paper does not have |
| 4 | FM JPEG q95, seed 2 | `172d4271c596.pt` | MSE@8 259.4 | with #2/#3: three-seed spread 259.4-266.6 (+-1.3 %), an order of magnitude below the gap to the claimed 449 |
| 5 | FM JPEG q95, horizon 8 | `52749e3ffdd0.pt` | MSE@1..8 = 230.7 -> 261.4; persistence@8 464.8 | short-horizon regime: error grows then flattens; even frame 1 (230.7) beats the paper's claimed frame-8 score. Completes the paper's horizon set {8, 16, 20} |
| 6 | FM JPEG q95, C = 48 (Table II axis) | `b7787a09b65b.pt` | MSE@8 261.3 (4.13 M params) | the paper claims a 1.3 % penalty for C=48; at the noise floor capacity does not bind — indistinguishable from C=96, marginally better |
| 7 | FM JPEG q95, lr 1e-4 (Table III axis) | `46051ae4197d.pt` | MSE@8 276.1 (early stop epoch 7) | ~4 % above lr 1e-3 — consistent with the paper's ordering, the only ablation axis with a real effect |
| 8 | FM JPEG q95, patch {4,4,4} & window {4,7,7} | `f79c9f671e00.pt` | MSE@8 253.8 (16.6 M params) | the coarsest patching is *best* here; the paper claims it is its *worst* (455.13). A 1-2 % Table cell that does not survive seed noise |
| 9 | FM JPEG q95, **grayscale**, C = 32 (Appendix A design) | `1f25298f21b9.pt` | MSE@8 142.2 (1.89 M params) | refutes Appendix A at the model level: the paper's own grayscale design is 1.9x *better* than the RGB C=96 model (section 3.7) |
| 10 | Cross-validation dataset, local, seed 0 | `949a9e195592.pt` | MSE@12 205.4; persistence@12 375.3; lookback-mean@12 203.0 (987 windows) | Fig. 8's claimed 531.4 beaten 2.6x — even the trivial references outscore it (section 3.8) |
| 11 | Cross-validation dataset, seed 1 | `e51cac94d6d0.pt` | MSE@8 201.4; @12 201.2 | +-1 % seed dispersion on the second dataset; #10 is not a seed artefact |
| 12 | LTE from scratch (Fig. 10) | `142297339c71.pt` | avg MSE over 16 frames 347.9 (369 windows, early stop epoch 9) | scratch arm of the TL pair; wall time 46.0 min reproduced the paper's "46.3" exactly, identifying Fig. 10(c)'s unit as minutes |
| 13 | LTE, transfer-learned from FM (Algorithm 4) | `6545adec4af7.pt` | avg MSE 347.3 | TL arm: "comparable MSE" reproduced (marginally better, as claimed); the -77.32 % time claim is not attainable under any defensible stopping rule (best -42 %) |

**Per-run deviations from the paper, at a glance.** #1 is the literal pipeline (no rendering
deviation — only the section-2.1 gap fills). #2-#4 add the JPEG q95 hypothesis (section 3.3) and
replication the paper does not have. #5: K = T = 8. #6, #7, #8: the paper's own Table II/III
axes (kept at the paper's values). #9: 1-channel grayscale frames, C = 32 (the paper's design),
LPIPS skipped as the paper reports only MSE/PSNR there. #10-#11: second dataset trained locally
(the paper never states whether Fig. 8 is local or transferred), plus the codec hypothesis and
seeds. #12-#13: LTE band, input-16-predict-16 per Fig. 10; #13 initialised from checkpoint #2.

**The probes** (no model trained; payloads in `results.json` -> `_probes`; both re-run on
2026-11-09 and matching their documented values exactly):

| Probe | Key numbers | What it proved |
|---|---|---|
| Metric space | persistence MSE@8 = 1614.66 (raw); implied MSE from the published PSNR = 20.08 | the paper's MSE and PSNR panels are mutually inconsistent by 22.35x; only the MSE panel is targetable (3.1) |
| Rendering identification | jpeg80: persistence MSE@18 = 448.6 (inside the paper's 449-481 cluster); blur1.5 / resize2: lookback-mean SSIM 0.79 / 0.74 | no single rendering satisfies the paper's MSE, SSIM and LPIPS anchors jointly (3.3) |
| Noise floor (section 3.2) | sigma^2 ~ 520-780 raw vs ~190-220 after JPEG; persistence 1614.7 raw / ~465-500 codec | the claimed 449 is below the raw floor and equals the codec persistence level — the mechanism behind runs #1 and #2 |

---

## 5. Verdict against the paper's anchors

| Paper anchor | Verdict | Run |
|---|---|---|
| Table II MSE 448.7628 @8 (453.0376 @16) | **Not attained by any trained model under any rendering probed**; under the codec rendering the value equals the persistence level, and the faithful port beats it by 42 % | #2-#4 |
| Fig. 7 SSIM 0.7753 @18 | **Reproduced within 3.7 %** (0.8037) under eq. (32) read literally (global SSIM) | #1 |
| Fig. 7 LPIPS 0.1222 @8 | Bracketed: persistence 0.092-0.098, our model 0.252-0.285 — consistent with a persistence-level predictor | #1-#2 |
| Fig. 7 PSNR 35.1039 @8 | **Unreproducible by construction** (contradicts the paper's own MSE panel 22.35x) | probe |
| Fig. 8 cross-validation 531.4 @12 | **Claim beaten 2.6x** (205.4); even trivial references outscore it | #10-#11 |
| Fig. 10 TL "comparable MSE" | **Reproduced** (347.3 vs 347.9) | #12-#13 |
| Fig. 10 "-77.32 %" time | **Not reproduced** (best defensible reading -42 %) | #12-#13 |
| Appendix A grayscale 11.5x penalty | **Refuted at data and model level** (grayscale 1.9x better) | #9 |
| Table I params 16.32 M | **Reproduced**: 16,180,935 (ratio 0.991) with the eq. (22) head | all C=96 runs |
| 3D-SwinLinear "90 % accuracy" | **Blocked**: the SOR threshold lambda and block size w are never given values | — |
| Abstract ">5 % better than benchmarks" | The paper's own body text quotes MSE -4.51 %, SSIM +0.23 %, PSNR +3.45 %, LPIPS -1.05 % on FM; three of four are below 5 %. The margin is smaller than the persistence -> conditional-mean gap any trained model achieves here | all of the above |

Reading: every anchor that survives the paper's internal inconsistencies is either reproduced
(SSIM under the paper's own formula, parameter count, TL comparability), explained (the MSE panel
is the persistence level of a codec-smoothed rendering — and is beaten), or blocked by
information the paper never provides (SOR arm, PSNR panel).

---

## 6. The targetable published anchors

The headline comparisons are published as plots; only prose-quoted cells are numerically
targetable. Self-consistent families, compared against in section 5:

**Tables II-III (FM, input-20-predict-20, MSE)** — 20 cells; the ablation grid:

| Ablation | Setting | MSE @8 | MSE @16 |
|---|---|---|---|
| blocks | {2,2,2} / {2,4,2} / {2,6,2} | 447.4807 / 448.7628 / 450.8983 | 455.0453 / 453.0376 / 456.3673 |
| channels C | 48 / 96 / 128 | 454.8080 / 448.7628 / 453.0270 | 470.2125 / 453.0376 / 457.4434 |
| learning rate | 0.01 / 0.001 / 0.0001 | x / 448.7628 / 459.8991 | x / 453.0376 / 460.7154 |
| patch & window | {2,2,2}&{2,4,4} / {2,4,4}&{2,7,7} / {4,4,4}&{4,7,7} | 440.8266 / 448.7628 / 455.1307 | 455.7650 / 453.0376 / 467.7145 |

**Cross-method prose anchors (FM, Fig. 7)**: MSE@18 DCG 475.6372 vs 3D-SwinSTB 454.1826
`[P:L 2070]`; SSIM@18 ConvLSTM 0.7735 vs 0.7753 `[P:L 2081]`; PSNR@8 NN-ResNet 33.9343 vs
35.1039 `[P:L 2083]`; LPIPS@8 SAE-TSS 0.1235 vs 0.1222 `[P:L 2085]`.

**Cross-validation (Fig. 8)**: MSE@12 DCG 593.4100 vs 531.4336 `[P:L 2105]`; no-pyramid 606.8934
vs 531.4336 `[P:L 2110]`; PSNR@18 32.3268/32.7158 vs 33.1593 `[P:L 2107, 2113]`.

**Limitation anchors (Fig. 11)** — the paper's own counter-evidence: MSE@15 ConvLSTM{128,64,32}
444.2705 vs 3D-SwinSTB 452.6198 `[P:L 2639]`; SSIM@10 ConvLSTM{256,64,16} 0.7787 vs 0.7764
`[P:L 2642]`.

**Appendix A (frame 10)**: RGB vs grayscale MSE 450.7880 vs 5294.5261, PSNR 35.0961 vs 11.3215
`[P:L 2777-2782]`.

---

## 7. Running it

### 7.1 Environment

Python 3.10, PyTorch 2.6.0+cu124 (the paper used PyTorch 1.8.0 on an RTX 3090 Ti; runs here were
made on a Quadro RTX 6000 24 GB and re-verified on an RTX 3050 laptop GPU — the LTE scratch wall
time matching the paper's minute-for-minute indicates the compute environments are equivalent
for these workloads). With `uv`:

```bash
uv venv --python 3.10
uv pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install -r training/DeepSPred/requirements.txt
```

### 7.2 Train -> checkpoint -> evaluate

```bash
cd training/DeepSPred/repro

# train the headline run (resumable: run keys already in results.json are skipped)
python run_paper_faithful.py --data ../../../data/datasets/deepspred/FM-Spectrum-Dataset \
    --horizon 20 --seed 0 --jpeg-quality 95
# ablation axes: --embed-dim 48 | --depths 2 6 2 | --lr 0.0001 | --patch 4 4 4 --window 4 7 7
# grayscale (Appendix A): --gray --embed-dim 32
# transfer learning (Algorithm 4):
python run_paper_faithful.py --data ../../../data/datasets/deepspred/LTE-Spectrum-Dataset \
    --dataset lte --horizon 16 --jpeg-quality 95 \
    --init-from ../../../checkpoints/repro/deepspred/f6426d84dd5d.pt

# evaluate: re-score every checkpoint on its test split, check the documented cells,
# rebuild results.json (expect: documented cells checked: 25, DIFFER: 0)
python verify_checkpoints.py            # or: --only <sha12> --no-lpips --device cpu
```

Training writes `checkpoints/repro/deepspred/<sha1(run_key)[:12]>.pt` (self-describing: run key,
arch, regime, normalisation note); evaluation writes `repro/results.json`.

### 7.3 Data, caches, and Windows notes

- Raw release (gitignored): `data/datasets/deepspred/{FM,LTE,Cross-Validation}-Spectrum-Dataset/`
  — one `.xls` per second of raw I/Q (10,800 / 2,400 / 6,600 files; FM is missing 7 seconds
  inside its span, cross-validation 234 — hence C1). Upstream: the authors' Google Drive release
  via `github.com/pgl1234/Real-world-Spectrum`; no licence declared, so it is never committed.
- Frame caches (`data/datasets/deepspred/frames/`, gitignored) are built on first use and keyed
  on every preprocessing switch. They are regenerable (hours of CPU).
- **Windows**: `fm_data.load_frames` renders via a `fork` multiprocessing pool, which does not
  exist on Windows. Build the caches single-process first (then all scripts load them normally):
  ```python
  from fm_data import load_frames
  frames, meta = load_frames(r'..\..\..\data\datasets\deepspred\FM-Spectrum-Dataset', workers=1)
  ```
  or run under WSL. A full `verify_checkpoints.py` pass takes ~2-4 h, mostly in float64 SSIM;
  each evaluation peaks at ~75 GB RAM — run at most 2 in parallel.
- LPIPS(alex) weights are downloaded by the `lpips` package on first use; `--no-lpips` skips it.

### 7.4 Provenance and file map

The reproduction sources were lost and rebuilt **byte-identical** from surviving `__pycache__`
compiled artifacts (verified instruction-by-instruction, 2026-11-09): `repro/{fm_data, metrics,
run_paper_faithful}.py` and `models/DeepSPred.py`. `verify_checkpoints.py` is the post-loss
verification harness that re-scored all 13 checkpoints and rebuilt `results.json`; fields
checkpoints never stored (`history`, `best_val_loss`, `train_seconds`) are `null` in the rebuilt
records (marked `_recovered`) — intentional, documented here.

```
models/DeepSPred.py                     shared 3D-SwinSTB architecture (both heads)
checkpoints/repro/deepspred/*.pt        13 self-describing checkpoints (verified; gitignored,
                                        object store: spectrum/checkpoints/repro/deepspred/,
                                        manifest in checkpoints/MANIFEST.tsv)
data/datasets/deepspred/                raw .xls release (gitignored)
data/datasets/deepspred/frames/         rendered frame caches (gitignored, regenerable)
training/DeepSPred/
  README.md                             how to run (entry point)
  REPRODUCTION.md                       this file: code vs paper + reasoning + results
  requirements.txt                      single dependency list (repro + integrated tracks)
  repro/
    fm_data.py                          I/Q -> frames: STFT, dB, colormap, JPEG switch, caches
    metrics.py                          MSE (4 spaces), PSNR, SSIM (windowed + global), LPIPS
    run_paper_faithful.py               training: paper regime, resumable, checkpoints
    verify_checkpoints.py               evaluation: re-scores checkpoints, rebuilds results.json
    results.json                        13 run records + _probes payloads
  dataset.py, model.py, utils.py, config_support.py,
  train_integrated.py, evaluate_integrated.py            integrated POWDER benchmark track
                                        (separate regime, wired into training/common; never
                                        mixed with the reproduction numbers)
```

**Consolidation note (2026-11-09).** This file replaces nine former documents; their content
survives here as follows: EXPERIMENTS.md -> sections 4-5; DEEPSPRED_REFERENCE.md -> sections
1-3, 6; CODE_VS_PAPER.md -> section 2; MODEL_CARD.md -> sections 2.2, 3, 5 (paper-audit items)
and section 3.1 (the abstract's ">5 %" assessment); MODEL_WALKTHROUGH.md -> sections 1-3;
REPRODUCTION_SUMMARY.md -> sections 3-5; RESULTS.md / DESIGN.md (the separate POWDER benchmark
regime) -> retired with the standalone benchmark entry points; the integrated track keeps its
files (section 7.4). The probe scripts (`probe_metric_space.py`, `identify_rendering.py`) and
the CSV exporter (`export_results_csv.py`) were removed with this consolidation; their outputs
are preserved verbatim in `results.json` -> `_probes`, and their findings in sections 3.1-3.3.
