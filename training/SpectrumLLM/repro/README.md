# Reproducing the SpectrumLLM experiments

Everything here runs from the repository root. The experiment-by-experiment story is in
`../EXPERIMENT_LOG.md`; what the model is and what we conclude about the paper is in
`../MODEL_CARD.md`.

## 1. Get the data

The three ElectroSense Madrid CSVs belong to the paper's authors and declare **no licence**, so they
are not vendored here. Fetch them yourself at the pinned commit:

```bash
mkdir -p data/electrosense_madrid
for f in spec4_data.csv spec25_data.csv spec151_data.csv; do
  curl -L -o "data/electrosense_madrid/$f" \
    "https://raw.githubusercontent.com/LIUC-000/SP/64b7fa5e96/$f"
done
sha256sum data/electrosense_madrid/*.csv
```

Expected sha256 prefixes: `spec4` `d4b0e31f03433e85`, `spec25` `4c884030643ffe4b`,
`spec151` `5317f35687555637`. Each file is 2872 rows, 15-minute cadence,
2021-05-28 00:00 to 2021-06-26 21:44, no NaNs, no duplicate timestamps, one 14-minute gap.
Column headers are channel centre frequencies in MHz; all three span 500-800 MHz.

`data/electrosense_madrid/*` is gitignored (only `.gitkeep` is tracked), matching how every other
testbed directory in `data/` is handled.

> Note on provenance: every other site in `data/` is synced from object storage with
> `training/ray/scripts/sync-cluster-data.sh --site <name>`. This one is not, because it cannot be
> re-hosted. That is a deliberate deviation from the repo convention.

## 2. Run

```bash
export PYTHONPATH=$PWD/models

# the paper's 4-channel table, as the article describes the architecture
python -u training/SpectrumLLM/repro/run_paper_regime.py \
  --data-dir data/electrosense_madrid --datasets spec4_data.csv \
  --horizons 12 24 48 96 192 --seeds 0 1 2 --out results.json

# the same, with Time-LLM's channel handling -- this is what reproduces the published numbers
python -u training/SpectrumLLM/repro/run_paper_regime.py \
  --data-dir data/electrosense_madrid --datasets spec25_data.csv \
  --horizons 12 24 48 96 192 --seeds 0 1 2 \
  --channel-independent --micro-batch-size 4 --split tslib --out results.json

# controls
#   --backbone random                          untrained transformer
#   --backbone google-bert/bert-base-uncased   the paper's other LLM
```

### `--split` is the one that matters

The article states a 7:3 split (section VI-A). That fixes the training fraction and leaves the
rest ambiguous, so the runner implements both readings:

| value | the other 30% | fitting windows (K=12) | scored windows (K=12) |
|---|---|---|---|
| `literal` | all test; validation carved from the end of the training span | 1626 | 851 |
| `tslib` | 10% validation then 20% test, chronological | 1807 | 563 |

`tslib` copies Time-Series-Library's `Dataset_Custom` exactly (`num_train = int(0.7n)`,
`num_test = int(0.2n)`, `num_vali` the remainder, each non-training span offset back by one
lookback, scaler fitted on the training span only). Seven of the article's eleven compared
methods come from that library.

`tslib` is the **default**. On 25 channels it moves Time-LLM GPT-2 from a mean ratio of 0.887 to
**0.970** and DLinear from 0.884 to 0.995 against the published rows. Pass `--split literal` to
reproduce the earlier campaign; every result row records which was used. Evidence:
`dlinear/DLINEAR_VARIANTS.md` (450 fits over 45 configurations) and `results_split_study.json`.

Channel handling defaults to the **article's** arrangement — one token spans every channel
(§IV-B). `--channel-independent` switches to Time-LLM's, folding channels into the batch. Which
is better depends on channel count, and the two disagree in opposite directions on the two
published scenarios; see the grid in section 3.

## 3. Hyperparameters worth searching on new data

Mean ratio to the published row, Time-LLM (GPT-2), from this repo's runs — the grid that should
inform where you search:

| channel mode | split | 4 channels | 25 channels |
|---|---|---|---|
| all-channel (default) | literal | 1.013 | 1.209 |
| all-channel | tslib (default) | 0.943 | not run |
| channel-independent | literal | 0.937 | 0.887 |
| channel-independent | tslib | 0.879 | 0.970 |

Search these, roughly in order of how much they moved our results:

| axis | flag | default | suggested range | what we found |
|---|---|---|---|---|
| split | `--split` | `tslib` | `tslib`, `literal` | largest single effect on 25 channels: 0.887 -> 0.970 |
| channel mode | `--channel-independent` | off (all-channel) | on, off | reverses sign with channel count: all-channel wins at N=4, loses badly at N=25 |
| learning rate | `--lr` | 1e-4 | 1e-4, 5e-4, 1e-3 | worth ~6% on the OFA arm (0.928 -> 0.988); the article states none |
| backbone | `--backbone` | `openai-community/gpt2` | gpt2, `google-bert/bert-base-uncased`, `random` | random is a frozen untrained control, not capacity-matched |
| `d_ff` | `--d-ff` | 32 | 32, 64 | **do not widen far.** Head params / target values is 0.43 at 32, 1.7 at 128, 10.2 at 768, independent of channel count. At 128 we measured over-fitting (MAD 0.087 -> 0.104) |
| `d_model` | `--d-model` | 32 | 16, 32, 64 | Time-LLM's own default is 16; keep >= patch_length x channels to avoid squeezing the all-channel token |
| patch stride | `--patch-stride` | 8 | 4, 8, 16 | changes token count (24 at stride 8, 46 at 4, 178 at 1); untested beyond stride 8 |
| epochs / patience | `--epochs`, `--patience` | 10 / 10 | 10-50 | beyond 10 changed nothing in our sweep |
| prompt statistics | `--prompt-stats` | `raw` | `raw`, `normalized` | `raw` describes the input in its own units; `normalized` reproduces pre-2026-11-04 runs |
| prompt on/off | `--prompt-mode` | `static` | `static`, `none` | ablation for the Prompt-as-Prefix contribution |
| seeds | `--seeds` | 0 1 2 | >= 3 | per-cell seed sd is ~2.4% of MSE; single-seed cells have swung by 20% |

**The resume key does not include `d_model`, `d_ff`, `patch_stride`, `lr` or `epochs`** -- only
`(dataset, horizon, seed, backbone, space, channel_independent, arm, split, prompt_stats)`. Give
each sweep point its own `--out` file or the runner will skip it as already done.

The runner is **resumable**: it skips any
`(dataset, horizon, seed, backbone, space, channel_independent, arm, split)` already present in
`--out`. Checkpoints land in `checkpoints/repro/2d/spectrumllm/seed-N/`.

### Running it on a new dataset

The CSV needs a `date` column plus one column per channel, and **every channel header must parse
as a float** -- `float(c)` is applied to each and the values are stored in the checkpoint as
`frequencies`. Then:

1. Put the file in the directory you pass as `--data-dir` and name it with `--datasets`.
2. Edit the four module constants at the top of `run_paper_regime.py`, which are not flags:
   `SEQ_LEN` (lookback, 192), `TRAIN_FRAC` (0.7), `DAY_STEPS` (96 = one day at 15-minute
   sampling, used only by the seasonal-naive reference and meaningless at another cadence), and
   `SITE`, which goes into the checkpoint chunk id alongside the band and channel count -- rename
   it or new checkpoints collide with the ElectroSense ones.
3. Lower `--micro-batch-size` if you have more channels. Channel-independent mode expands one
   sample into N sequences, so a 32-sample batch is 32xN sequences through the backbone; 4 works
   for 25 channels on a 24 GB card. It splits only the backward pass, so the optimizer step stays
   at `--batch-size` samples and runs stay comparable across channel counts.

Budget roughly 2,500-3,500 s per horizon for 25 channels at 10 epochs on a Quadro RTX 6000.

### Where the backbone weights come from

`models/SpectrumLLM.py` calls `AutoConfig` / `AutoModel` / `AutoTokenizer.from_pretrained(backbone)`,
default `openai-community/gpt2`, so weights are pulled from the Hugging Face Hub on first use into
the standard cache (`$HF_HOME`, default `~/.cache/huggingface/hub`) -- 526 MB for GPT-2. Two
things to know: `n_layer` is overridden to `--llm-layers` (6) *before* `from_pretrained`, so the
full 12-layer checkpoint downloads but only 6 layers are built; and every backbone parameter gets
`requires_grad = False` immediately after loading, so the LLM is frozen and what trains is the
reprogramming layer -- 95% of the trainable mass is `mapping_layer`, sized by GPT-2's 50257
vocabulary. Set `local_files_only: true` in the model config for an air-gapped machine, or
`backbone: random` for the untrained control (256-wide, so not capacity-matched).

In a container, mount the cache or every run re-downloads it:

```bash
docker build -t spectrum-split:latest training/SpectrumLLM/repro/docker
docker run --rm --gpus all -v "$PWD:$PWD" -v "$HOME/hf_cache:/root/.cache/huggingface" \
  -w "$PWD" -e PYTHONPATH="$PWD/models" --entrypoint python spectrum-split:latest \
  training/SpectrumLLM/repro/run_paper_regime.py --datasets spec25_data.csv \
  --data-dir data/electrosense_madrid --channel-independent --micro-batch-size 4 \
  --split tslib --seeds 0 --out results.json
```

`--micro-batch-size` splits each optimizer step into smaller forward/backward passes without
changing the step size. Channel-independent mode turns one sample into N sequences, so 32 samples
is 800 sequences at N=25 and will not fit a 24 GB card. Use 4 for 25 channels, 8 for 4 channels.
This is mathematically equivalent to the unsplit batch -- there is no batch-norm in the model and
RevIN is per instance.

## 4. Check the implementation

```bash
python training/SpectrumLLM/repro/verify_spectrumllm.py   # 32 executable checks
python -m pytest tests/test_spectrum_llm.py               # 10 unit tests
```

## 5. What is here (after the 2026-11-04 consolidation)

Everything from the `spectrum-split-repro` scratch folder now lives here: its unique results
were moved in, its duplicates (an older `run_paper_regime.py`, a second copy of
`results_split_study.json` and `search_spectrumllm.py`, the data CSVs, older model-file
copies) were deleted, and its 15 checkpoints (3.0 GB) moved to `checkpoints/repro/2d/spectrumllm/seed-0/`
under the repo root. One 1-epoch smoke test (`smoke_tslib.json`) was deleted; every
configuration it covered has a full 10-epoch run.

**Start with `MATRIX.md`** -- the readable inventory of what exists for every
(arm, scenario, split, channel mode), with known gaps and caveats. Then:

| file | contents |
|---|---|
| `MATRIX.md` | coverage matrix per arm and protocol cell, hyperparameter coverage, known gaps, caveats |
| `results_master.csv` | **all 331 runs**, one row each, with status (`primary` / `sweep` / `superseded_lr1e-3`), provenance and sha-verified checkpoint presence |
| **`results_archive.json`** | **the record**: every kept campaign's runs at full fidelity (per-epoch history, trivial references, both metric spaces), the published tables with their 14 internal inconsistencies, the 70-file checkpoint reload audit, and the headline findings -- 16 campaigns, 241 runs in one file |

The loose per-campaign result JSONs were consolidated into `results_archive.json`
and deleted on 2026-11-05; the campaign keys inside the archive are their
original filenames. Superseded campaigns whose checkpoints no longer exist
(old OFA/BERT literal runs, the lr 1e-3 campaign, protocol sweeps, split probes)
were dropped entirely -- their conclusions live in the documents below.

Analysis documents:

| file | contents |
|---|---|
| `RESULTS_PAPER_REGIME.md` | the reproduction result in detail, including the DLinear protocol search and the direct Time-Series-Library verification |
| `CHECKPOINTS.md` | the checkpoint guide: filename decoder, reload rules, what every weight file means |
| `TIMELLM_OFA_REFERENCE.md` | line-by-line comparison of our port against Time-LLM's released code |
| `dlinear/` | **archived DLinear working set** (gitignored): the 450-fit protocol search, all diagnostic campaigns, the direct Time-Series-Library run logs, and the seed sweeps that established the protocol and its noise band |

Scripts and configs:

| file | contents |
|---|---|
| `run_paper_regime.py` | the runner; `--split tslib` is the default, `--channel-independent` switches channel handling |
| `search_spectrumllm.py`, `config_search_spectrumllm.yaml` | staged hyperparameter search driving the runner |
| `config_search_node.yaml` | the same search configured for a worker node (smoke first, relative `data_dir`) |
| `make_preds.py`, `pred_plot_K96.npz` | forecast extraction from the K=96 25-channel checkpoint for plotting |
| `verify_spectrumllm.py` | the 32 implementation checks |
| `run_dlinear.sh` | runs the settled DLinear reproduction in the container (`training/SpectrumLLM/run_dlinear.py`) |
| `make_figures.py` | plots (channel heatmaps, spec4-vs-spec25 crop check); runs in the `tsslcd-ours:mpl` image |

Deliberately **not** in git, each regenerable or re-fetchable: the article text (IEEE copyright),
`timellm_ref/` (OFA declares no licence), `data/` (re-download per section 1), and the
checkpoints themselves. On disk: 15 checkpoints (3.0 GB) under `checkpoints/repro/` -- the
split-study and `res4` campaign's weights, all sha-verified. The first campaign's 150 exist
only as SHA-256 hashes in `checkpoints_manifest.tsv`; the 40 OFA/BERT checkpoints only as
hashes inside `results_ofa_bert.json`. Chunk ids encode neither split nor capacity, so several
earlier runs' weights were overwritten at the same paths -- `results_master.csv`'s
`checkpoint_on_disk` column distinguishes `yes` / `overwritten` / `no` per run.

## 6. Hyperparameters the article never states

Patch length and stride, prototype count, d_model, d_ff, attention heads, epochs, batch size,
learning rate, optimizer, seed, validation split, early stopping, and hardware. Ours come from
Time-LLM's released defaults and are recorded in every result row and every checkpoint, so any
number here can be traced to the settings that produced it. See `TIMELLM_OFA_REFERENCE.md`.

## 7. Notes on the first campaign (November 1 cleanup)

Thirteen stale result files were deleted outright in the November 1 cleanup. Every number from
that campaign was regenerated by reloading the checkpoints; none was carried over from an
earlier aggregate. The consolidation of 2026-11-04 (section 5) added the split-study and
`res4` campaigns and dissolved the scratch folder they were run in.

Not ours and left alone: everything under `training/DeepSPred/`, `training/TSS-LCD/` and
`training/LinearAutoRegressive/`, and the 1.5 GB `results.tar.gz` at the repository root.

### Two reproducibility hazards found while regenerating

* **Random-backbone runs need their seed to reload.** The untrained transformer is frozen and
  therefore never written to the checkpoint; it exists only via `torch.manual_seed(seed)` before
  the model is built. Reload without re-seeding and you get 0.6454 where the run recorded 0.2840.
* **Test MSE depends on the evaluation batch size.** Prompts are tokenised with `padding=True`
  and no attention mask, so each is padded to the longest in its batch and the padding takes part
  in attention. Same weights on spec151: 1.1723 at batch 32, 1.2174 at batch 1. Always report the
  batch alongside the metric.
