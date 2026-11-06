# The SpectrumLLM repro checkpoints: what they are and how to use them

Every trained weight file in this reproduction, decoded. Companion inventory with
exact SHA-256 hashes: `checkpoints_manifest.tsv` (one row per checkpoint, local
and remote).

## 1. Where the weights live

**Local** (`checkpoints/repro/2d/spectrumllm/seed-N/`, gitignored, ~15 GB):
every current weight file. All of them are `tslib`-split runs.

**Remote** (object storage, container `spectrum` on Chameleon Chi@TACC, via
`checkpoint_storage.py`):

```
checkpoints/repro/spectrumllm/
  literal/seed-0..4/   150 first-campaign weights (2026-11-02): literal split,
                       seeds 0-4, Time-LLM gpt2/bert/random, 4/25/151 channels.
                       Metrics: reeval_runs.json (each was re-evaluated bit-exact
                       before local deletion). Old prompt text.
  tslib/seed-0..2/     the current weights, mirroring the local directory.
```

## 2. Reading a filename

```
electrosense_madrid_500_800_n25_h96_tslib_ci_ofa_spectrumllm.pt
└──────── site ────────┘└band┘└n┘└h┘└─split─┘└ci┘└arm┘└─ suffix ─┘
```

| field | meaning |
|---|---|
| `n25` | channel count (4, 25 or 151 -- the band alone would collide) |
| `h96` | prediction horizon K in steps (12..192; 96 steps = 24 h at 15 min) |
| `tslib` | split convention: 70/10/20 chronological (`Dataset_Custom`). **Files without this tag predate 2026-11-05** and are old-prompt vintage |
| `ci` | channel-independent (each channel folded through the model separately); absent = one token spans all channels (the article's equation) |
| `_ofa` | the OFA arm (GPT4TS: no reprogramming, no prompt, only ln+wpe trained) |
| `_bb<name>` | non-GPT-2 backbone, e.g. `bbbertbaseuncased` |
| `_dm64ff128` | non-default capacity (d_model 64, d_ff 128); absent = 32/32 |

## 3. How to load a checkpoint and get its result

The checkpoint payload is self-describing: `model_name` (spectrumllm/ofa),
`model_state_dict` (**trainable parameters only** -- the frozen backbone is
rebuilt from `config.model.backbone` + `llm_layers`), `normalization`
(train-span mean/std), `frequencies` (channel centre frequencies, MHz),
`training_results`, and `config` -- **the model must be rebuilt from the
payload's own config**, not from current defaults.

The one-command way (re-scores every local checkpoint against its recorded test
MSE):

```bash
docker run --rm --gpus all -v "$PWD:$PWD" -v "$HOME/hf_cache:/root/.cache/huggingface" \
  -w "$PWD" -e PYTHONPATH="$PWD/models" --entrypoint python spectrum-split:latest \
  training/SpectrumLLM/repro/verify_checkpoints.py
```

Rules that matter when re-scoring by hand:

- **Evaluation batch size changes the metric.** Prompts are tokenized with
  `padding=True` and no attention mask, so score at the batch size the run used:
  32 for all-channel, 8 for 4-channel CI, 4 for 25-channel CI.
- **Prompt vintage**: the prompt template lives in `models/SpectrumLLM.py`, not
  in the checkpoint. Old-tag files (no `_tslib` in the name, or any of the 150
  `literal/` files) re-score within ~0.2-1.5% of their recorded values under
  current code -- the drift is the prompt-text change, not corruption.
  `--prompt-stats` in the payload config says which statistics the prompt carries.
- **Random backbones** (`_bbrandom`) need `torch.manual_seed(seed)` before the
  model is built, or they reload as different weights.
- Single-window forecast extraction from one checkpoint: `make_preds.py`.

## 4. What each checkpoint means

All values are test MSE in raw PSD units, seed 0, `tslib` split, recorded in the
named result file. "pub" = the article's published cell; ratio = ours/published.

**Time-LLM (GPT-2), d_model 32 -- the four protocol cells (2026-11-05,
verbatim prompt).** Result records: campaign `results_n*_tslib.json` in
`results_archive.json`:

| files | cell | mean MSE | ratio to pub |
|---|---|---:|---:|
| `n4_h*_tslib_spectrumllm` | 4ch all-channel | 0.3093 | **1.006** |
| `n4_h*_tslib_ci_*` | 4ch channel-indep | 0.2977 | 0.968 |
| `n25_h*_tslib_spectrumllm` | 25ch all-channel | 1.1425 | 1.144 |
| `n25_h*_tslib_ci_*` | 25ch channel-indep | 0.8240 | **0.968** |

The two bolded cells are the reproduction: **4ch published row = all-channel,
25ch published row = channel-independent**, each matched within ~3% under a
single protocol.

**Capacity variants:** `*_tslib_dm64ff128_*` -- d_model 64 / d_ff 128
(campaigns `results_n25_all_d64_tslib.json`, `res4_all_match.json`, with the
d32 side in `res4_all_dm32.json`; all in `results_archive.json`). Finding:
widening does **not** rescue all-channel at 25ch (1.45x vs CI) -- the failure is
architectural, not a width bottleneck.

**Other arms (CI, both scenarios):** `*_tslib_ci_ofa_*` (OFA/GPT-2, campaigns
`results_n*_ofa_ci_tslib.json`: 25ch matches published at 0.956, 4ch overshoots
at 0.885) and `*_tslib_bbbertbaseuncased_ci_*` (Time-LLM/BERT, campaigns
`results_n*_bert_ci_tslib.json`; matches at K=12-96, **diverges at K=192** in
both scenarios -- a real training instability of BERT at the longest horizon,
recorded as-is).

**Old-prompt vintage (kept for the multi-seed evidence):**
`n25_h*_spectrumllm` without `_tslib` = campaign `results_n25_all_tslib.json`,
the only **3-seed** campaign (seeds 0/1/2, hence `tslib/seed-1`, `tslib/seed-2`
remote); the `res4_*` and split-study campaigns are superseded by the four
protocol cells above but document the campaign that settled the split question.

**Remote `literal/` (150 files):** the first campaign, superseded as a
reproduction but the only 5-seed evidence (4ch all-channel, seeds 0-4). Its
metrics are campaign `reeval_runs.json`; its protocol search is
`dlinear/DLINEAR_VARIANTS.md`.

## 5. The record

Every result record at full fidelity -- per-epoch histories, trivial references,
both metric spaces, the published tables, and the checkpoint reload audit --
is consolidated in **`results_archive.json`** (one file, 16 campaigns, 241
runs; built 2026-11-05 by a one-off consolidation, verified by regenerating
`results_master.csv` byte-identically from it). `results_master.csv` and
`MATRIX.md` are the flat and readable views of the same data, kept as
snapshots. The inventory of every local and remote checkpoint with SHA-256 is
`checkpoints_manifest.tsv`; `checkpoint_storage.py verify` re-checks the
remote side.
