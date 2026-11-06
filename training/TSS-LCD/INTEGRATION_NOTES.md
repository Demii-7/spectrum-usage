# TSS-LCD integration into spectrum-usage — pre-integration notes

Status: NOTES ONLY. No integration work has started. Written 2026-11-05,
based on a full read-only survey of this repo plus the completed first-commit
reproduction campaign at `/home/cc/TSS-LCD/` (see its `CODE_REVIEW.md`,
`EXPERIMENTS.md`, `repro/README.md`).

## 1. What exists where (four codebases, three architectures)

| location | what it is | architecture |
|---|---|---|
| `/home/cc/TSS-LCD/` (root .py files) | the authors' FIRST commit `d2d4f29`, untouched | context-**autoencoder** (time branch dim-750 + serial freq→spatial branch + 2-input cross-attn fusion + 32-dim bottleneck), Conv1d-along-time future AE, cosine s=0.0018, their analytic p_sample, batch 256, 300+300+2000 epochs, 10% masking, overlapping split, savgol(51) + eval-time averaging |
| `/home/cc/TSS-LCD/repro/` + `results/first_commit/` | our first-commit campaign (this repo's sibling) | same architecture, instrumented; 8 runs incl. ablations and wide-condition variant |
| `training/TSS-LCD/repro/` (here) | the HEAD `b156d0a` campaign | wraps the authors' HEAD code (`upstream/`); 19 arms; source of `EXPERIMENTS.md` |
| `training/TSS-LCD/model.py` (here) | **paper-faithful clean-room rewrite** — the LIVE integrated model | 3 parallel branches with out_proj, 3-input FFM (Q=temp, K=spec, V=spat), flatten→latent projection, Conv2d future AE, **no context-AE**, stage-2 regresses the future latent, cosine **s=0.008**, DDIM-50 eval |
| `models/TSSLCD.py` (here) | a SECOND, divergent implementation — **dead code in the live path** | 2-input FFM (Q=spectral), mean-pooled condition, cosine **s=0.0018**, 1000-step sampler, `TSSLCDForecaster` adapter; 60.55M params vs authors' 65.50M |

Key fact: the two in-repo copies share class names but have
**non-interoperable state dicts** and different math (FFM wiring, pooling,
schedule). The live benchmark path (`training/common/specialized_models.py`
→ `training/TSS-LCD/train_integrated.py`) uses `training/TSS-LCD/model.py`;
`models/TSSLCD.py` is reachable only through the **dead** factory branch
(`training/common/model_factory.py:494-697`) and the **orphaned**
`training/common/tss_lcd_training.py` (nothing imports it). This violates
the repo's own policy (`REPRODUCTION_VS_BENCHMARK.md` §1: reproduction and
benchmark share exactly `models/<Model>.py`).

## 2. What the first-commit campaign established (decisions this forces)

From `/home/cc/TSS-LCD/CODE_REVIEW.md` and `EXPERIMENTS.md`:

1. The authors' headline (MSE 0.1541) reproduces **only under their leaking
   evaluation** (overlapping split + full-series scaler + centered savgol +
   eval-time averaging + hindsight in the scored series). Leak-free: 0.453.
   Truly held-out windows: 0.289. Persistence: 0.0064 (26x better than the
   model, everywhere).
2. Table IV ablations do not reproduce (−9.9%..+32.1% vs paper's
   +85.7%..+103.7%); removing SpeFE *improves* the model.
3. The 32-dim condition bottleneck is load-bearing: the paper's wide
   conditioning through their pipeline is *worse* (0.200 vs 0.164).
4. Their sampler contradicts their own Algorithm 1; the paper's sampler
   diverges on their schedule (ᾱ₉₉₉ ≈ 2.5e-9).

Consequence for integration: **there is no single "TSS-LCD model" to
benchmark.** The released architecture only performs under a leaking
evaluation; the paper-described architecture is a different (non-functioning
as-written) system; the in-repo rewrite is already a third thing. The
benchmark entry must declare which one it is and why. Given this repo's
leak-free benchmark protocol, the released code's evaluation-specific
behaviors (savgol, eval averaging, overlapping split, t+1-only scoring)
must NOT be ported — the existing rewrite's choices (no smoothing,
chronological splits, train-fit normalization) are the right baseline.

## 3. Concrete incompatibilities to resolve (framework contract)

1. **Data.** Framework loader (`training/common/data.py::load_chunk`) needs
   `timestamp_utc`-headered per-site dBm CSVs; authors' CSVs are headerless,
   timestamp-less, 750 cols = 3 sites x 250 bins. `data/tsslcd/PROVENANCE.md`
   warns the local `training/data/merged_power...csv` is NOT the authors'
   data (−2.16 dB mean offset). Options: (a) benchmark on framework POWDER
   data (L=1, current behavior), (b) write an AERPAW converter with
   synthesized timestamps and per-site files.
2. **Locations axis.** Integrated path hard-codes `L=1`
   (`train_integrated.py:112`); the model's SpaFE branch is then a 1-token
   no-op. Multi-site would need the framework's segment machinery to expose
   site as a tensor axis, or keep L=1 and document that the spatial branch
   is inert in benchmark configuration.
3. **Horizon contract.** Framework requires `prediction_horizon ∈ {1,
   max(horizons)}`; TSS-LCD uses direct t_out=60 from ONE diffusion sample
   (legal, but per-horizon rows slice a single rollout — different failure
   profile than AR models; must be stated in the MODEL_CARD).
4. **Checkpoint format.** Specialized `component_states` dict vs generic
   `model_state_dict` (`model_factory.py:780` raises). Either unify, or
   formally exempt specialized models from the generic loader and remove
   the dead factory branch.
5. **Config schema.** Flat `tss_lcd:` section in `training/common/config.yaml`
   lacks keys `build_model` requires (recorded smoke `KeyError` in
   MODEL_CARD §13d); nested form exists in
   `training/configs/ray_tuning_tss_lcd_2d.yaml`. Pick one schema, fix the
   other, keep the `config_sections()` shim only as long as needed.
6. **Eval-time masking.** `evaluate_integrated.py` re-applies 25% zero
   masking to TEST inputs (faithful to the paper's scenario). Every other
   benchmark model gets clean test inputs. For a comparable TABLE.md row,
   masking should default OFF at eval (or be reported as a separate
   robustness table).
7. **Diffusion stochasticity + cost.** Evaluation pins a seeded generator
   (good — seed must travel in checkpoint metadata, it does). DDIM-50 makes
   TSS-LCD an inference-cost outlier; DEFERRED_WORK.md's idle-GPU cost
   benchmark will surface this. Full 1000-step sampling is ~20x worse.
8. **Schedule inconsistency.** `training/TSS-LCD/model.py` uses cosine
   s=0.008; `models/TSSLCD.py` uses the authors' s=0.0018 (whose degenerate
   tail we showed breaks the paper's own sampler). Whichever copy survives,
   set s explicitly and record the choice; s=0.008 is the defensible one.

## 3b. Weight portability: our first-commit checkpoints vs this repo (measured)

Question raised: can the checkpoints trained in `/home/cc/TSS-LCD/results/first_commit/`
be loaded into the spectrum-usage TSS-LCD implementations? **No.** Measured
comparison of `paper_s0/diffusion.pth` (authors' first-commit `NoiseNet.DiffusionModel`,
96-dim NEN input) against both in-repo diffusion models:

| | keys | shared names w/ ours | shared name+shape |
|---|---|---|---|
| `models/TSSLCD.py::DiffusionModel` | 42 | 6 | 5 (schedule buffers + time-embedding) |
| `training/TSS-LCD/model.py::DiffusionModel` | 42 | 6 | 5 |

- Their NEN is not geometry-compatible: ours flattens the U-Net output to
  `Linear(32, 6144)`; theirs applies `AdaptiveAvgPool1d` then `Linear(32, 64)`
  (the one shared-name layer, `noise_net.final.weight`, has shape (32,6144)
  vs (32,64)).
- Conditioner: 0 of our 89 `ContextTransformerAE` key names exist in their
  `TSSConditionConstructor` (parallel branches + 3-input FFM + no context
  decoder vs serial branches + context autoencoder). The future AE is
  Conv1d-along-time vs their Conv2d-with-pooling.
- Even with a shape-perfect transplant the weights would be semantically
  invalid: ours were trained on savgol-smoothed, per-column min-max, 750-col,
  10%-masked inputs with a 32-dim context-AE condition; the framework feeds
  z-scored (train-fit), unsmoothed, differently-masked inputs through a
  different conditioner.

Consequence: getting the authors' architecture into the benchmark means
porting the first-commit modules (they are self-contained; our
`first_commit_repro.py` already wraps them) as a new entry and **training
from scratch** under the framework pipeline. Our checkpoints' value remains
in the reproduction campaign, not the benchmark.

## 4. Benchmark entry requirements (per repo policy)

- `REPRODUCTION_VS_BENCHMARK.md`: reproduction numbers (authors' data,
  smoothed targets, dB² MSE) never enter TABLE.md. Benchmark rows come from
  `training/configs/*.yaml` + `training/ray/registry.py` anchors, every
  hyperparameter explicit, **5 seeds (40–44)**, mean ± SEM.
- Current campaign: `training/configs/oct2026/ray_tuning_2d.yaml`
  (October POWDER, horizons [1,15,60], lookback 60) — TSS-LCD **not** in its
  model list. A staged June config exists
  (`training/configs/ray_tuning_tss_lcd_2d.yaml`, 6 sites, 20/20/20 epochs)
  and the Ray search space already registers `tss_lcd`
  (`training/ray/registry.py:346`).
- Final numbers additionally need the stride-1 rescoring on the final test
  range (2026-10-27T20:00Z→2026-11-03T20:00Z) per DEFERRED_WORK.md.
- TABLE.md today has no TSS-LCD row; 2D leaders: ResidualLinearAR2D 0.860
  MAE, TemporalConvNet 0.868, LookbackMean2D 0.888 (the trivial reference).

## 5. Decisions needed before starting (for the repo owner)

1. **Which architecture is the benchmark entry?**
   (a) keep the existing paper-faithful rewrite (`training/TSS-LCD/model.py`)
   as `tss_lcd` — recommended: it is already integrated, leak-free, and its
   deviations are documented; or
   (b) additionally port the authors' released architecture (context-AE,
   32-dim bottleneck, their sampler) as a separate entry, e.g.
   `tss_lcd_authors`, purely to answer "how does the released system fare
   under an honest protocol?" (our chrono_s0 predicts: badly — ~0.29 dB² MSE
   on their data, worse than every current 2D table entry is in MAE terms
   after protocol conversion... needs a protocol-matched run to say).
2. **Canonical model copy:** delete or defuse `models/TSSLCD.py`, the
   factory `tss_lcd` branch, and `training/common/tss_lcd_training.py`, or
   make them thin re-exports of `training/TSS-LCD/model.py`.
3. **Eval masking default OFF** for the benchmark row (report the masked
   variant separately), yes/no.
4. **L=1 (framework data, spatial branch inert) vs multi-site AERPAW
   benchmark**, or both.
5. **Campaign:** join oct2026 (add to config + registry anchors) or run the
   staged June config first.
6. **Where the first-commit campaign artifacts live long-term:** keep
   `/home/cc/TSS-LCD/` standalone, or fold its `CODE_REVIEW.md`,
   `EXPERIMENTS.md`, and `repro/` under `training/TSS-LCD/repro/first_commit/`
   (note: `repro/upstream/` there already vendors HEAD; the first-commit
   originals are byte-different for `Context2CondNew.py` and `train.py`).

## 6. Suggested order of work (not started)

1. Resolve §5 decisions 1–3 (architecture, canonical copy, masking).
2. De-duplicate: kill dead paths, fix the config-schema trap, unify
   checkpoint handling for specialized models.
3. Smoke: `config.smoke.yaml` tss_lcd section → end-to-end train+eval.
4. Join oct2026: config entry + registry anchors (explicit hyperparams),
   5-seed run, stride-1 final eval, TABLE.md row + MODEL_CARD §13 benchmark
   subsection.
5. Optional: `tss_lcd_authors` entry (released architecture under the
   honest protocol) + AERPAW data converter if multi-site is wanted.
6. Fold first-commit campaign docs into the repo per §5.6.
