# Model Card Template

Copy to `training/<Model>/MODEL_CARD.md` and fill. One card per benchmarked model.

**Scope.** This card is a *paper-and-reproduction audit*, not a build document. The existing
`training/<Model>/README.md` already covers how to run the model, its tensor shapes and its layer-by-layer
architecture; do not restate those here. Link to them. The card answers a different set of questions: what
the original paper actually did, what it claimed, whether we can check it, and where our version is guessing.

**Rules for filling it in.**
- Every number carries its source: `[P:7]` = paper page 7, `[C:file.py:120]` = code line, `[M]` = measured by us,
  `[?]` = unknown. An unsourced number is a guess and must be marked `[?]`.
- Leave a field `unknown` rather than inferring it. Blank fields are the finding — three of our seven source
  papers have whole sections that can only read `unknown`, and that is the result.
- Never copy a number from the paper's abstract if the body or code contradicts it. Record both.

---

```yaml
# Machine-readable header - keep keys stable, a cross-model table is generated from these.
model: ""                  # registry key, e.g. tsslcd
display_name: ""
paper_cite: ""             # First author, short title, venue, year
doi: ""
zotero_key: ""
year:
venue: ""
benchmarked: true          # false = implemented but not in the paper's tables
representation: ""         # 1d | 2d | 4d  (as fed in our repo)
residual_variant: false    # does a Residual* wrapper exist
params:                    # our build, at reference config
code_available: ""         # none | partial | full          (+ is it spectrum-specific?)
data_available: ""         # none | partial | full
checkpoints_available: false
license: ""
reproducible: ""           # exact | partial | architecture-only | not-reproducible
repro_blocker: ""          # one phrase, if not exact
compares_vs_trivial: false # did the PAPER compare against persistence / lookback mean?
beats_persistence:         # OUR measurement: true | false | not-run
released_artifacts_match_paper:   # true | false | n/a - do the authors' own released
                           # predictions/checkpoints reproduce their own published table?
leakage_found: ""          # none | suspected | confirmed - one phrase
```

## 0. At a glance

| | |
|---|---|
| Paper | |
| Code | |
| Data | |
| Reproducible | |
| Beats a trivial reference | |
| Headline claim | |
| Our verdict on that claim | |

## 1. Task as the paper formulates it

<!-- The formulation is where most incomparability between papers originates. Be literal. -->

- Lookback / horizon, in samples **and** wall-clock:
- Sampling interval:
- Single-step or multi-step; if multi-step, direct or recursive:
- Quantity predicted: raw power dBm / normalised / occupancy binary / spectrogram image /
- Spatial scope: single sensor / N sensors / interpolated map
- **Our benchmark's regime**, and whether it matches:

## 2. Dataset used in the paper

| Field | Value | Source |
|---|---|---|
| Testbed / sensor | | |
| Location | | |
| Frequency range | | |
| Bin width / n bins | | |
| Time span (dates) | | |
| Sampling interval | | |
| Tensor shape (T x F x L) | | |
| Total samples | | |
| Collection hardware | | |
| Collection procedure | | |
| Train/val/test split | | |
| Split is chronological? | | |
| Public? Where? | | |

Notes on collection: <!-- sweep vs simultaneous capture, antenna, gain calibration, dead bins,
outages, whether dBm is absolute or uncalibrated relative power. Affects whether our data is comparable. -->

## 3. Data transformations

<!-- The full chain in order, from raw file to model input. This section has repeatedly been the
place where the real explanation of a paper's results was hiding. -->

1.
2.

- Normalisation statistics fit on: train only / whole file / per-file  `[C:]`
- Scaling is per-bin / global / per-file:
- Smoothing, interpolation or imputation applied, and **whether it is applied before windowing**:
- **Leakage audit.** Does any transform at time *t* consume samples after *t*? Does any statistic cross the
  split boundary? State explicitly, with the line of code:
- **Metric space.** Are the paper's reported errors in physical units (dB) or in normalised [0,1] space?
  Is the transform inverted before scoring?  <!-- Papers reporting MSE in [0,1] are not comparable to
  each other or to us; record the inversion status, it decides whether a number can be quoted at all. -->

## 4. What the model needs besides the spectrum

<!-- Auxiliary inputs are the main reason a model is hard to port to a new testbed. -->

| Requirement | Needed? | How we supply it | Risk |
|---|---|---|---|
| Time-of-day / calendar encoding | | | |
| Sensor coordinates / geometry | | | |
| Observation mask (missing data) | | | |
| Neighbouring-sensor input | | | |
| Fixed number of sites or bins | | | <!-- e.g. a hardcoded `input_dim // 3` -->
| Pretrained weights (which, size) | | | |
| Text prompt / tokenizer | | | |
| Per-dataset hyperparameter tuning | | | |
| External library beyond our core deps | | | |

## 5. Architecture and capacity

- One-paragraph summary; full detail in `README.md`:
- Parameter count at our reference config `[M]`:
- Parameters in the paper, if stated `[P:]`:
- **Capacity ratio**: params / training samples =   <!-- >1 means more parameters than data points;
  record it, it is often the simplest explanation of a reported result. -->

## 6. Hyperparameters: declared vs actual

| Parameter | In paper | In their code | We use | Note |
|---|---|---|---|---|
| | | | | |

<!-- Mismatches between a paper's table, a config file and a second config file in the same repo are
common. List every one found; they are findings for the manuscript. -->

## 7. Claims, and how they are tested

<!-- One numbered block per claim. Quote the claim; do not paraphrase into something stronger or weaker. -->

### Claim 1: ""  `[P:]`
- Evidence offered:
- Baselines compared against:
- **Trivial reference included?** (last-value persistence, lookback mean, seasonal naive):
- Metrics and the space they are computed in:
- Seeds / repeated runs / error bars:
- Ablations supporting it:
- Is the comparison apples-to-apples (same split, horizon, preprocessing, tuning budget)?
- **Our assessment:**

## 8. Errors found in the paper or code

| # | Where | What | Severity | Evidence |
|---|---|---|---|---|
| | | | | |

<!-- Severity: cosmetic (typo, mislabelled axis) | inconsistency (paper vs code disagree, dead code) |
substantive (affects a reported number) | invalidating (the headline result does not follow).
Each row needs a line number or page number someone else can check. -->

## 9. Availability and reproducibility

- Code: URL, commit/date pinned, what is actually in it, is it spectrum-specific or a generic forecaster:
- Data: URL, is it the *same* data the paper used or a successor release:
- Checkpoints:
- License, and whether redistribution is permitted:
- Dead links, retired platforms, repos returning 404:

**Verdict** — pick one and name the blocker:

| Level | Meaning |
|---|---|
| exact | their code + their data + their split reproduces their table |
| partial | code and data exist, but split / seed / hyperparameters are missing |
| architecture-only | we rebuilt the model from the text; no original data to test against |
| not-reproducible | neither code nor data; or the data source no longer exists |

## 10. Compute cost

| | Theirs `[P:]` | Ours `[M]` |
|---|---|---|
| Hardware | | |
| Epochs | | |
| Wall-clock to train | | |
| Inference per sample | | |
| Peak memory | | |

<!-- Also note if training time is dominated by something unusual (a 1000-step diffusion chain at
inference, a frozen LLM forward) - it changes how the model can be used, not just how long it takes. -->

## 11. Our implementation

- Representation fed (1D / 2D / 4D) and why:
- Residual variant exists / is used:
- Built from: their code / the paper text / a third-party reimplementation:

**Differences from their released code**, each with a disposition:

| # | Theirs | Ours | Disposition |
|---|---|---|---|
| | | | fixed / deliberate / unresolved |

<!-- "Deliberate" must give the reason (e.g. we do not adopt a transform that leaks future information).
"Unresolved" belongs in section 12. -->

## 12. What we are not sure about

<!-- The honest-uncertainty register. For the manuscript, this is what separates a reproduction from a
claim. Every guess our implementation makes, with its consequence. -->

| # | Uncertainty | Why unresolved | Could it change a reported number? | What would settle it |
|---|---|---|---|---|
| | | | | |

## 13. Reproduction results

| Predictor | Metric @ h=1 | @ h=15 | @ h=60 |
|---|---|---|---|
| This model (ours) | | | |
| Last-value persistence | | | |
| Lookback mean | | | |
| Paper's reported number | | | |
| **Authors' released predictions, rescored by us** | | | |

<!-- The last row is separate from the paper's number on purpose. Where a repo ships predictions or
checkpoints, score them yourself: a gap between a paper's table and its own released artifacts is a
distinct finding from a gap between the paper and our reimplementation, and the two must not be merged. -->

- Units and whether inverse-transformed:
- Skill vs persistence:
- If the paper's number is not comparable, say why rather than tabling it:

## 14. Open questions and next actions

- [ ]

## 15. Evidence provenance

- Paper PDF read: Zotero key, pages
- Code read: repo, **commit SHA** (pin it - a branch tarball gives no SHA and the upstream can move), download date
- Data file: path, shape, checksum
- Runs this card draws on: run id / artifact id
- Card author and date; revise on any re-run:
