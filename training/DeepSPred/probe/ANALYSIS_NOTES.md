# DeepSPred (3D-SwinSTB) reproduction: analysis notes for the paper's reproduction section

Written 2026-11-10 from a working session. Everything here is either traceable to a file in this
repository, to a job on the UC cluster, or to the paper, and every statement carries a status tag.

Status tags used throughout:

- **[V]** verified by me directly: from the paper's text, from the data, or from a result file in `results/`.
- **[C]** taken from the collaborator's writeup (`training/DeepSPred/REPRODUCTION.md`, commit `45dc184`). I read it; I did not re-run it.
- **[D]** derived by hand or by an approximation (the approximation is stated).
- **[H]** hypothesis, not tested.

All MSE values are in the paper's own space: mean squared error over pixels and the three colour channels of
uint8 (0-255) values, `MAX_I = 255` (the collaborator's `uint8_mean_per_channel`).

Companion files: `results/TABLES.md` (every table below, generated from the JSONs by `make_tables.py`),
`results/*.json` (raw probe outputs, sha256 in section 9), `rendering_probe.py`, `acf_probe.py`, `range_probe.py`.

---

## 0. One-paragraph summary

The paper reports MSE 448.76 at frame 8 on FM spectrograms (Table II) and 531.43 at frame 12 on the
cross-validation set (Fig. 8). The collaborator implemented the architecture and training recipe with every stated
value and a parameter count within 1% of Table I [C], but the paper does not say how the raw I/Q was turned into
images, and the result depends on that more than on anything else. With the most literal rendering (Hann STFT, dB,
dataset-wide min-max, jet) the trained model scores 846 [C], 1.9x the paper. We then characterised what any model
could reach under 20+ candidate renderings with a linear-predictor ceiling, and found that (a) the absolute MSE level
is close to a free parameter, because it scales roughly as 1/(dB span)^k with k between 1.3 and 2.5 and the paper does
not state the span; (b) no rendering we tried reproduces the paper's FM level, FM-to-cross-validation ratio and
horizon shape together; (c) the one rendering that matched the FM level (rectangular window, 433) fails the
cross-validation ratio (2.1 vs the paper's about 1.2), and its detrended variant (78) is an artifact of a 1e-12 floor in the dB
conversion. The honest conclusion available today is "not reproducible as published from the released data", with one
specific, testable lead (full-dataset min-max under scipy-default STFT settings) that has not been closed.

---

## 1. The paper and what it specifies

Pan, Wu, Zhou, Li, Wang, Ding, Yau, *Spectrum Prediction With Deep 3D Pyramid Vision Transformer Learning*,
IEEE Trans. Wireless Commun. 24(1):509-525, Jan 2025, DOI 10.1109/TWC.2024.3495812. The copy I read is the IEEE PDF
(`papers/DeepSPred.pdf`, downloaded from IEEE Xplore on 2026-06-24). The paper's code/data pointer is
`https://github.com/pgl1234/Real-world-Spectrum` (Google Drive links; raw I/Q only, no preprocessing code) [C].

### 1.1 Stated in the paper [V]

| Item | Value | Where |
|---|---|---|
| Task | input-n-predict-n (K = T) spectrogram prediction; frames are H x W x 3 RGB, "a common Jet colormap" | Sec. III-A, eq. (3) |
| Spectrogram | eq. (2): `SPEC = |STFT|^2`; STFT window length 256 | eq. (2), Sec. VI-A1 |
| FM data | 90-110 MHz, centre 99 MHz, fs 125 MHz, "descending sampling coefficient" 4, "STFT number" 32508 | Sec. VI-A1 |
| FM split | 4:1:1 chronological on 2022-09-23: 7200 train (17:20-19:20), 1800 validation (19:20-19:50), 1800 test (19:50-20:20) | Sec. VI-A1 |
| Cross-validation data | another node, 2024-07-10 22:14 to 2024-07-11 00:04, same settings otherwise | Sec. VI-A1 |
| LTE data | 690-710 MHz, centre 700 MHz, STFT number 16254, 2023-05-02 17:52-18:32 | Sec. VI-A1 |
| Architecture | C=96; encoder blocks {2,4,2}, heads {4,8,16}; bottleneck 2 layers; predictor {2,4,2}, heads {16,8,4}; patch (2,4,4); window (2,7,7) | Sec. VI-A2 |
| Training | MSE, AdamW, lr 0.001, batch 1, 20 epochs, early stop (theta_per = 0.01, n_ep = 4), patience 4 | Sec. VI-A2 |
| Hardware / software | i9-10940X, "GTX 3090Ti" (sic), PyTorch 1.8.0 | Sec. VI-A2 |
| Metrics | MSE (eq. 30, a double sum), PSNR (eq. 31, `MAX_I = 2^8-1 = 255`), SSIM (eq. 32, no window stated), LPIPS | Sec. VI-A4 |
| Baselines | DCG, ConvLSTM, NN-ResNet, SAE-TSS: "same hyperparameters, except the network structure" | Sec. VI-A3, VI-B |

### 1.2 Not stated in the paper [V]

The paper never says: the STFT window function; the hop size or overlap (only that the result is 256 x 256); one- or
two-sided FFT; whether power is converted to dB; how values are scaled to [0,1] or 0-255 (min-max? per image or per
dataset? which frames? fixed limits?); whether the band was cropped from the recorded 31.25 MHz to the stated 20 MHz;
how the image was stored (PNG/JPEG/float) or resized; weight decay; seeds; the SSIM window; the LPIPS backbone; the
binarisation block size `w` (eq. 26) or the accuracy threshold lambda (Sec. VI-A4) needed for the SOR arm.
`MAX_I = 255` implies 8-bit images; nothing else about pixel scale is implied.

### 1.3 The numerical anchors we can target [V unless tagged]

Only values printed in the text are exactly targetable; figure curves are read off plots (approximate, roughly +-1%).

| Anchor | Value | Source |
|---|---|---|
| FM, 3D-SwinSTB MSE at frame 8 / 16 | 448.7628 / 453.0376 | Table II |
| FM, MSE at frame 18: DCG vs ours | 475.6372 vs 454.1826 (-4.51%) | Sec. VI-B |
| FM, SSIM at frame 18: ConvLSTM vs ours | 0.7735 vs 0.7753 | Sec. VI-B |
| FM, PSNR at frame 8: NN-ResNet vs ours | 33.9343 vs 35.1039 | Sec. VI-B |
| FM, LPIPS at frame 8: SAE-TSS vs ours | 0.1235 vs 0.1222 | Sec. VI-B |
| Table II, block numbers {2,2,2}/{2,4,2}/{2,6,2} | 447.4807/448.7628/450.8983 (frame 8); 455.0453/453.0376/456.3673 (frame 16) | Table II |
| Table II, channels C = 48/96/128 | 454.8080/448.7628/453.0270 (8); 470.2125/453.0376/457.4434 (16) | Table II |
| Table III: lr 0.01/0.001/0.0001 | x/448.7628/459.8991 (8); x/453.0376/460.7154 (16) | Table III |
| Table III: patch,window {2,2,2},{2,4,4} / {2,4,4},{2,7,7} / {4,4,4},{4,7,7} | 440.8266/448.7628/455.1307 (8); 455.7650/453.0376/467.7145 (16) | Table III |
| Cross-validation, MSE at frame 12: DCG, no-pyramid vs ours | 593.4100, 606.8934 vs 531.4336 | Sec. VI-B |
| Cross-validation, PSNR at frame 18: NN-ResNet, no-flow vs ours | 32.3268, 32.7158 vs 33.1593 | Sec. VI-B |
| Appendix A, frame 10, RGB vs grayscale: MSE | 5185.2540 -> 450.7880 (-91.31%) | Appendix A text (see 1.5) |
| Appendix A, frame 10, RGB vs grayscale: PSNR | 11.3390 -> 35.0961 (+209.52%) | Appendix A text (see 1.5) |
| Table I params of 3D-SwinSTB | 16.32 "MB" (almost certainly millions of parameters) | Table I |
| Transfer learning FM -> LTE | training time -77.32% (46.3 -> 10.5 min), comparable MSE | Fig. 10, Sec. VI-C |
| 3D-SwinLinear accuracy | about 0.90 at 8 and 16 frames | Fig. 9 |

Curve shapes read from Fig. 7/8 (approximate) [V]: 3D-SwinSTB MSE rises about 440 -> 455 over 20 frames (+3.4%);
ConvLSTM starts near 418 and rises to about 460 (the text says it beats 3D-SwinSTB below 4 frames); DCG,
NN-ResNet and SAE-TSS sit roughly flat at 460-485. Cross-validation: models 526-538, baselines 575-612.

### 1.4 The paper contradicts itself [V, arithmetic checked here]

- **PSNR vs MSE.** PSNR 35.1039 with `MAX_I = 255` implies MSE = 20.08, but the MSE panel says 448.76 at the same
  frame: a factor of **22.35**. The Appendix A RGB pair (35.0961 -> implied 20.11; MSE 450.788) gives 22.41. For the grayscale
  pair the implied MSE (4777 from PSNR 11.339) is within 1.09x of the printed 5185. For cross-validation, PSNR 33.1593 implies 31.4
  against 531.4 (a ratio of 16.9, at different frames, so indicative only). The ratio is not a single convention.
- **Grayscale is 11.5x worse than RGB in the paper** (5185 vs 451). In our data a single-channel rendering has *lower* MSE
  than RGB (see `results/TABLES.md` section 1: gray rows about 220-250 vs RGB about 420-1610), because jet amplifies pixel
  differences across three channels. The paper's gray/RGB ratio therefore implies different scalings between the arms or an error.
- Eq. (14) writes the MLP width as 2C, but Table I's parameter count only fits the Swin 4C [C]. Eq. (30) is a sum, eq. (31) uses it as a mean [C].
  The block-ablation prose quotes the same delta twice [C]. Full list: REPRODUCTION.md section 2.2.

### 1.5 A version discrepancy to resolve before citing Appendix A

The IEEE text I read gives the grayscale/RGB pair as MSE 5185.2540 -> 450.7880 ("decreased by 91.31%") and PSNR
11.3390 -> 35.0961 ("increased by 209.52%"). I recomputed both percentages and they match these numbers exactly
(0.91306 and 2.09517). The collaborator's REPRODUCTION.md section 6 instead quotes 5294.5261 and 11.3215 (line refs
`[P:L 2777-2782]`) and says its numbers are identical to arXiv:2408.06870v3. With those values the percentages would be
91.49% and 209.99%, which do not match the paper's printed percentages. Either the arXiv version differs from the IEEE
version, or the figure's curve values were used. Pick one version, and say which, in the report.

---

## 2. The data

| Item | Value |
|---|---|
| Source | The authors' Google Drive folder `1w2kTaRKVRmsO5jUAovZc78W_eTQ63TDm` (given by the project owner; three zips: FM, Cross-Validation, LTE) |
| Downloaded | FM (1,610,418,457-byte zip) and Cross-Validation (~769 MB zip), extracted 3.8 GB and 2.0 GB. **LTE not downloaded.** |
| Location | `ray-uc-1:/home/cc/spectrum-usage/data/deepspred/{FM-Spectrum-Dataset,Cross-Validation-Spectrum-Dataset}` (this host only; not on the other seven UC hosts or any other cluster) |
| File counts [V] | FM 10,800 `.xls` (names `172.19.220.14-2022-0923-hhmmss.xls`); Cross-Validation 6,600 (`172.19.220.12-2024-0710-...`); all names match the timestamp pattern the code expects |
| Format [V] | Tab-separated text; row 0 = `I, Q, hhmmss, longitude, latitude, altitude`; remaining rows = `I, Q`. One file = one second of capture |
| Samples per file [V] | 32,508 for FM (equals the paper's "STFT number"). At fs = 125 MHz / 4 = 31.25 MHz that is **about 1.04 ms of signal per file**, taken once per second [D] |
| Consequence [D] | Noise in consecutive frames is independent by construction (non-overlapping 1 ms snapshots, 1 s apart). Any frame-to-frame correlation is real signal |
| Gaps [C] | FM is missing 7 seconds inside its span; Cross-Validation is missing 234. Windows that straddle a gap are dropped in every analysis here |
| Band [V, derived] | The complex baseband spans 31.25 MHz around 99 MHz (83.4-114.6 MHz); the paper states 90-110 MHz (20 MHz) |

---

## 3. What the collaborator did (commit `45dc184`, 2026-11-10, author LaveshM)

Files: `training/DeepSPred/REPRODUCTION.md` (465 lines), `repro/{fm_data,metrics,run_paper_faithful,verify_checkpoints}.py`,
`repro/results.json` (13 run records + probes), `models/DeepSPred.py` (649 lines), `test_deepspred.py`. The code is on
`origin/integrate`; it was not in my local checkout, so **I did not read the model code or run it**. Checkpoints are in the
object store at `spectrum/checkpoints/repro/deepspred/` [C].

### 3.1 Fidelity (my assessment of the writeup against the paper)

Faithful to everything the paper states [V on the values I spot-checked, C on the code]: C=96, depths/heads, patch and
window sizes, AdamW at 1e-3, batch 1, 20 epochs, early stopping rule, 4:1:1 split reproduced exactly by time boundaries
(7200/1800/1800), K = T, MSE loss, parameter count 16,180,935 vs Table I's 16.32 (ratio 0.991), after setting the MLP ratio to 4.

Judgement calls that are defensible but not in the paper [C]: weight decay 0.05 (PyTorch's AdamW default is 0.01);
output sigmoid; LayerNorm placement; loss on [0,1] floats; early stopping watches validation loss; a Linear(2C -> C)
after each skip concatenation (forced: otherwise channel counts double).

Rendering choices the paper is silent on, which the collaborator fixed and recorded as switches [C]: Hann window, hop 126
(the only hop that yields exactly 256 columns from 32,508 samples), two-sided fftshifted FFT, `10*log10(power + 1e-12)`,
dataset-wide min-max from every 50th second, jet, uint8. **Added by the collaborator, not in the paper:** a JPEG round-trip
switch (`--jpeg-quality`, headline runs at q95) to bring the paper's MSE above the measured noise floor; persistence and
lookback-mean reference predictors; gap-aware windows.

### 3.2 Their key results [C], metric space as above

| Run | Rendering | Model MSE @ frame 8 | Persistence @8 | Lookback-mean @8 |
|---|---|---|---|---|
| #1 (the literal pipeline) | raw (no codec) | **845.9** | 1614.7 | 857.6 |
| #2 (headline) | JPEG q95 | 266.6 (seeds 266.6 / 259.6 / 259.4) | 464.9 | 252.0 |
| #5 (K=8) | JPEG q95 | 261.4 (frame 1: 230.7) | 464.8 | - |
| #9 grayscale, C=32 | JPEG q95 | 142.2 | - | - |
| #10 cross-validation, local | JPEG q95 | 205.4 @12 (seed 1: 201.4) | 375.3 | 203.0 |

Also: SSIM@18 0.4713 windowed (standard) vs 0.8037 under eq. (32) read literally (paper 0.7753); LPIPS@8 0.252 (paper
0.1222); LTE transfer learning "comparable MSE" reproduced (347.3 vs 347.9); the -77.32% time saving is not (best -42%).
The collaborator's verdict table is REPRODUCTION.md section 5; their ablation conclusions (Tables II-III differences are
1-2% and smaller than the +-1.3% seed spread) are in section 3.6.

### 3.3 Where I would be careful citing them

- "The paper's 448.8 is the persistence level of a JPEG-smoothed rendering, and our model beats it by 42%" is an inference
  from consistency, not a measured fact: it depends on JPEG being the right unstated step, and the collaborator's own probes
  say no rendering satisfies the paper's MSE, SSIM and LPIPS together (section 3.3 of their doc).
- "At the noise floor" for the raw run: the floor was estimated as 520-780 from crude statistics, and the model's 846 is above
  that range. The ceiling analysis below (section 5) gives a better-defined version: 837 for the best linear predictor on that rendering.
- Run histories (per-epoch loss, best validation loss, wall time) were lost with the source code and are null in `results.json`;
  "reaches the conditional mean within one epoch" cannot be checked from the stored records.
- Most of their results are for the JPEG q95 rendering, which is not paper-faithful. The paper-faithful one is #1 (raw).

---

## 4. What we added: a rendering study with no training

Rationale: the absolute MSE depends on the unstated rendering, so instead of training on each candidate, we measure what
any predictor could achieve on it. All runs are CPU-only jobs on the UC cluster, pinned to `ray-uc-1` (the only host with the
data), 16 cores each. My probes are an independent reimplementation (`rendering_probe.py`), not the collaborator's `fm_data.py`
(which was not in my checkout); sanity check [V]: it reproduces the collaborator's persistence numbers (raw 1555 / 1610 vs
their 1559 / 1615; JPEG q95 419 / 464 vs their 420 / 465).

### 4.1 Rendering definitions (all in `rendering_probe.py`, `acf_probe.py`)

- STFT: complex I/Q, nperseg 256, two-sided, `fftshift`. Window names encode shape and hop: `hann126` = Hann, hop 126 (derived so
  exactly 256 columns fit; this is the collaborator's rendering); `hamm128` = Hamming, hop 128 (MATLAB `spectrogram` default
  50% overlap, 252 columns resized to 256); `rect126`; `tukey126`; `tukey224` = hop 224 = scipy's `noverlap = nperseg // 8`, ~145
  columns resized to 256 by bilinear interpolation along time (`scipy.ndimage.zoom`, order 1); a trailing `d` subtracts each
  segment's complex mean before windowing (scipy `detrend='constant'`). Tukey alpha is 0.25 (scipy's default). Windows are
  `scipy.signal.get_window` (periodic).
- `10*log10(|S|^2 + 1e-12)`; then scaling: `glob` = min-max with the dataset-wide dB range from every 50th second (the
  collaborator's); `frame` = per-frame min-max; `p1p99` = global 1-99 percentile clip; `lin` = linear power min-max.
- Post-processing: `raw`, `jpegQ` (PIL JPEG at quality Q), `blurS` (Gaussian sigma S on the [0,1] field), `avgM` (box average of
  M adjacent STFT rows in linear power, before dB). Colormap jet (matplotlib), uint8 rounding.
- Cross-validation is rendered with its own dB range unless labelled `crossval_fmnorm` (FM's range).

### 4.2 Probe A: trivial predictors (`rendering_probe.py`; job `claude-uc-deepspred-render-probe-4`)

Persistence (repeat the last input frame) and lookback mean (mean of the 20 inputs) at k = 1..20 on the test split, stride-1
windows inside contiguous segments. Results: `results/TABLES.md` section 1. Findings [V]:
raw Hann persistence 1555 -> 1610; JPEG q95 419 -> 464 (matches the paper's ConvLSTM curve in level and shape); no rendering
puts persistence at the paper's cross-validation level (all < FM, the paper's is above). **I later judged persistence to be the
wrong yardstick** (a trained model is not a copy of the last frame), so it is reported for completeness and for sanity-checking
against the collaborator, not as evidence about the paper.

### 4.3 Probe B: temporal structure and the linear ceiling (`acf_probe.py`)

For a fixed random subset of 6,000 of the 196,608 pixel positions (seed 0), over the paper's validation block (frames at 2/3-5/6
of the capture span; "fit") and test block (the last 1/6):

- **mean-image MSE** (`clim`): predict every test frame with the fit-block mean image (zero-skill reference).
- **rho(k)**: lag-k autocorrelation of deviations from that mean image, pooled over pixels, contiguous segments only.
- **ceiling(k)**: out-of-sample MSE at horizon k of the best *linear* predictor of the deviation at t+k from the deviations at
  t, t-1, ..., t-4, with one coefficient set per k shared across all pixels, solved from fit-block autocovariances with ridge
  `lambda = 1e-3 * c(0)`.
- **skill(k)** = 1 - ceiling(k) / clim. **rise** = ceiling(20) / ceiling(1) - 1.

Limits [D]: a linear, pixel-shared predictor on 5 lags is a *lower-effort* ceiling; a better model could go lower. It is
independent of any specific trained model. Calibration check [V]: on the raw Hann rendering the collaborator's trained Swin model
(846) is within 1.1% of this ceiling (837) and within 0.4% of `clim` (843); on JPEG q95 it is about 6% above the ceiling at k=8
(267 vs 252) and about flat over the horizon.

### 4.4 Probe C: the dB range (`range_probe.py`; job `claude-uc-deepspred-range-probe-1`)

For each window, min and max dB over (i) all frames, (ii) every 50th frame (the collaborator's), (iii) the first 2/3 of the
capture (train only), plus a robust 0.1-99.9 percentile span and the median per-frame span.

---

## 5. Findings

### 5.1 Temporal structure is real and short-lived [V]

(`results/TABLES.md` sections 2-5.) FM lag-1 autocorrelation of the frame deviations: raw Hann 0.08; JPEG q95 0.19; Tukey(0.25)
at hop 224 0.15-0.16; heavy in-frame averaging (avg16) 0.76 decaying to 0.17 by lag 8. With in-frame noise reduction the ceiling's
skill at k=1 is large and is gone by k of about 4-8. **Short-term-better-than-long-term skill therefore appears from in-frame noise
reduction alone; it does not require smoothing across frames**, which the data's 1 ms-per-second structure would not support anyway.
The measured rise of the ceiling from k=1 to 20 (FM, raw): Hann +1.1%, Tukey hop 126 +1.8%, rectangular +2.9%, Hamming +2.9%,
Tukey hop 224 +3.7%, **scipy default (tukey224d) +3.9%**; JPEG +5-12%; row-averaged +15-180%. The paper's rise is about +3.4%
(read off a plot, +-1%), so the shape excludes heavy smoothing and does not strongly discriminate among the raw windows.

### 5.2 The absolute level is nearly a free parameter because of normalisation [V for the data, D for the scaling]

With min-max scaling a dB error maps to a pixel error proportional to 1/span, so MSE scales as span^-k. Measured exponents
from our own results [D]: rectangular -> rectangular+detrend (span 117.9 -> 239.4 dB, ceiling 438 -> 80): k = 2.4;
Hann global -> per-frame (span 146.9 -> 116.6 median, 843 -> 1515): k = 2.5; Hann global -> 1-99% clip
(146.9 -> 79.3, 843 -> 1922): k = 1.3 (clipping softens it). So "about 1/span^2" is a rough guide only, and any
estimate built on it carries a factor-of-a-few-tens-of-percent uncertainty.

Consequence: matching the paper's 449 or 531 by itself is weak evidence for a rendering, because each dataset's span is an unstated
choice. The scale-free quantities (rise over horizon, k=1 vs k=20, autocorrelation, the share of predictable variance) are the
better discriminators. This is a correction of how I read the rectangular-window result earlier (section 7).

### 5.3 No rendering reproduces level, ratio and shape together [V]

FM ceiling at k=8 (paper 448.8) and cross-validation / FM ratio (paper about 1.18-1.25), sampled-range normalisation:

| Rendering (raw unless stated) | FM k=8 | CV own range k=8 | CV/FM own | CV, FM's range k=8 | CV/FM (FM range) | FM rise |
|---|---|---|---|---|---|---|
| Hann hop 126 (collaborator's) | 837 | 863 | 1.03 | 693 | 0.83 | +1.1% |
| Hamming hop 128 | 598 | 983 | 1.64 | 621 | 1.04 | +2.9% |
| Rectangular hop 126 | 433 | 899 | 2.08 | 1051 | 2.43 | +2.9% |
| Tukey(0.25) hop 126 | 786 | 1077 | 1.37 | 678 | 0.86 | +1.8% |
| Tukey(0.25) hop 224 | 709 | 1010 | 1.42 | 700 | 0.99 | +3.7% |
| **scipy defaults (Tukey, hop 224, detrend)** | 725.5 | 1059 | 1.46 | 744 | 1.03 | +3.9% |
| Hann, JPEG q95 | 252 | 305 | 1.21 | - | - | +5.5% |

The JPEG q95 cross-validation/FM ratio (1.21) matches the paper's, but its level (about 250-300) is about half the paper's;
and the level is the adjustable part (5.2). Rectangular reaches the FM level (433) but is 1.7-2x too high on cross-validation.
All numbers: `results/TABLES.md`.

### 5.4 Artifact: rectangular + detrend [V]

Rectangular window with per-segment mean subtraction gives a ceiling of 78 on FM (vs 433 without detrend). The range probe shows
why: with a rectangular window and the mean removed, the DC bin is exactly zero, `10*log10(0 + 1e-12) = -120.0 dB`, and the
dataset minimum becomes exactly -120.0 (FM full range -120.0..119.4, span 239.4 vs 132.0 without detrend). The same exact -120.0
floor appears for rectangular *without* detrend on the full cross-validation set (span 225.2 vs 117.6 sampled), so a few frames
contain an exact-zero bin there, which the 1-in-50 sample missed. These are artifacts of our epsilon, not properties of the data.
In the authors' pipeline exact zeros would give -inf dB, so this reading cannot be what they did without additional handling.

### 5.5 What the range probe says about the "full dataset min-max" lead [V for spans, D for the estimates]

(`results/TABLES.md` section 6.) Sampled vs full dB spans for the scipy-default window (tukey224d): FM 130.7 vs 156.2; cross-validation
104.0 vs 117.0 (train-only: 146.5 and 117.0). Under a span^-2 scaling [D] (uncertain, section 5.2), the full-dataset range would move
the ceilings at k=8 from 725.5 -> about 508 (FM; paper 448.8) and from 1059 -> about 837 (cross-validation; paper 531.4). To reach the
paper's numbers exactly, the span would have to be about **166 dB (FM) and 147 dB (cross-validation)**; measured full spans are 156 and 117.
FM then lands within about 13% of the paper; cross-validation remains 1.6x too high. **These are scaling estimates, not measurements.**
The measurement would be to re-run the ceiling with full-range (or train-only) normalisation, which has not been done.
(An earlier message of mine quoted needed spans of 187 and 177 dB; that used the Hann spans (147/125) as the baseline instead of the
tukey224d sampled spans (130.7/104.0), and is wrong. The values above replace it.)

---

## 6. What can and cannot be claimed

Safe to state, with the evidence [V]:
1. The paper specifies none of: window, hop, dB conversion, normalisation scope or range, band crop, image storage. The FM error
   level (linear-predictor ceiling or persistence) spans about 230 to 3,500 across the plausible choices tried, a factor of
   about 15, excluding the collapsed linear scaling and the detrend artifact (`TABLES.md` sections 1-5).
2. With the most literal rendering, the architecture and recipe stated in the paper give MSE 846 on FM at frame 8 [C], 1.9x the
   paper's 448.8, and that model is within about 1% of what a linear predictor achieves on that rendering [V for the ceiling].
3. The paper's PSNR and MSE panels are mutually inconsistent by a factor of 22.4 on FM, and its grayscale/RGB comparison contradicts what
   the data gives (a single channel has lower, not 11.5x higher, error).
4. The absolute MSE level depends on the unstated dB span, roughly as span^-1.3 to span^-2.5 [D], so a level match alone cannot
   identify the rendering.
5. Short-horizon skill that decays within about 4-8 s exists in the data and appears under in-frame noise reduction, without
   cross-frame smoothing.

Do not claim:
- That the authors used JPEG, a rectangular window, the scipy defaults, or any specific rendering. None is established.
- That "the paper's models perform at persistence level" or that "our model beats the paper by 42%". Both depend on an unverified rendering.
- That the paper's numbers are impossible. We showed they are impossible *under the renderings we tried*; full-range normalisation of the
  scipy-default rendering is still open (5.5).
- Any trained-model result for a rendering other than raw Hann and JPEG q95: all other entries are linear-predictor ceilings.
- The dB equivalents of MSE as exact; they are approximate (section 8).

---

## 7. Corrections to statements I made earlier in this session

Included so the report does not inherit them.

| Earlier statement | Status |
|---|---|
| The probe would take 1-1.5 h | Wrong; it took about 20 min (no effect on results) |
| JPEG q95 fails the cross-validation check (persistence ratio 0.8 vs 1.25) | Retracted: persistence is not the right comparison; the ceiling ratio under JPEG q95 is 1.21 |
| The rectangular window matches the paper's level (433) | Weakened: level is nearly free via the dB span (5.2); the FM-to-cross-validation ratio (2.1) fails |
| Rect + detrend should land near 460-500 | Wrong: 78, an epsilon artifact (5.4) |
| The scipy default "needs spans of 187 / 177 dB" | Wrong baseline; correct values 166 / 147 dB (5.5) |
| "MSE is about 1/span^2" | Rough: measured exponents 1.3-2.5 (5.2) |
| "scipy's default hop and detrend don't help" (before the exact default was run) | Premature: the exact default (tukey224d) has been run since; FM 725.5 |
| The collaborator's grayscale anchors are 5294.5 / 11.32 | Disputed: the IEEE text gives 5185.25 / 11.339 and its percentages match those (1.5) |

---

## 8. Converting MSE back to physical units (for interpretation only) [D]

The pipeline is power -> 10 log10 -> min-max to [0,1] -> jet -> three uint8 channels. MSE averages over the three channels and the
colormap is nonlinear, so no exact inverse of an MSE exists. An approximation: the jet colormap's squared slope averaged over its
three channels is about 6.2 per unit^2 of the normalised value (from its piecewise-linear definition), i.e. about 4.0e5 uint8^2 per
unit^2; therefore `RMSE_dB ~ RMSE_pixels * span_dB / 632`, which for FM (span 147 dB) is `RMSE_pixels * 0.23`. By this, the paper's 449 is about
4.9 dB RMS, and the raw-rendering persistence of 1610 corresponds to about 9.3 dB between two frames, or about 6.6 dB per frame
(single-periodogram theory predicts about 5.6 dB, which is consistent). Treat as +-40%: it assumes pixels spread evenly across the colormap.
MAE needs the error distribution; MAPE is meaningless in pixel or dB space and must be computed on linear power. Exact conversion requires
inverting the images (`invert_colormap` in `training/DeepSPred/utils.py`, 1024-step lookup; `evaluate_integrated.py` already does it)
and is limited by JPEG damage and clipping at the range limits.

---

## 9. Reproducibility record

Cluster: UC (8x RTX 6000), CPU-only jobs on `ray-uc-1`, submitted with `ray job submit` from the head container, 16 CPUs each.
Code synced with `training/ray/scripts/sync-cluster-code.py --cluster uc` (committed tree). **Nothing pushed.**

| Job | Script | Result file (sha256) |
|---|---|---|
| `claude-uc-deepspred-render-probe-4` (-1 to -3 failed on setup bugs, fixed) | `rendering_probe.py` | `rendering_probe_20261110T172330.json`  |
| `claude-uc-deepspred-acf-probe-1` | `acf_probe.py` | `acf_probe_20261110T172942.json`  |
| `claude-uc-deepspred-tukey-probe-1` | `acf_probe.py --set tukey` | `acf_probe_tukey_20261110T180408.json`  |
| `claude-uc-deepspred-tukey-probe-2` | `acf_probe.py --set tukey` (adds tukey224d) | `acf_probe_tukey_20261110T183333.json`  |
| `claude-uc-deepspred-rect-probe-1` | `acf_probe.py --set rect` | `acf_probe_rect_20261110T183943.json`  |
| `claude-uc-deepspred-range-probe-1` | `range_probe.py` | `range_probe_20261110T184251.json`  |

Commits (local branch `integrate`): `d39e018`, `fd1f42d`, `7fde749`, `831b6d8`, `63f5eca`, `38befbd`, `3d77d4f`, `86cca82`, plus this
notes/tables commit. Full sha256 sums: `results/SHA256SUMS`. Original files live on `ray-uc-1:/home/cc/spectrum-usage/data/deepspred/probe/`; copies in `results/`.
Collaborator's reproduction: commit `45dc184` on `origin/integrate`.

Known limitations of the probes: fixed pixel subset (6,000 of 196,608, seed 0); linear shared-coefficient ceiling with 5 lags; stride-1 windows
within contiguous segments; fit and test blocks follow the paper's chronological split; float32 arithmetic; bilinear resize only; frame, percentile
and FM-range normalisation were each tried once; no band crop; no full-range normalisation run of the ceiling; no trained model on any alternative rendering.

---

## 10. Open items ranked by value per cost

1. **Re-run the ceiling with full-range and train-only normalisation** for the raw windows (minutes, CPU). This closes or kills the one live lead (5.5).
2. **Band crop** to the stated 90-110 MHz (central 164 of 256 bins, resized to 256), with and without full-range normalisation (minutes, CPU).
3. **Resize method** for the hop-224 case (bilinear vs bicubic vs area) and image-save path (matplotlib `savefig` vs direct array).
4. **FM-trained model evaluated on cross-validation** (a domain-shift test of the paper's Fig. 8); only the trivial-predictor version is cheap.
5. **One trained model** on the best candidate rendering, only if 1-3 yield a rendering matching level, rise and ratio together (hours of GPU).
6. **Overfit/sanity run** of the collaborator's Swin model on a tiny set, and a loss curve for the raw run (their histories are lost); would show the 846 is not an
   optimisation failure. The ceiling already argues it is not (846 vs 837).
