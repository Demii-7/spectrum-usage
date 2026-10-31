# Reproduction vs. benchmark: layout and config ownership

Companion to `MODEL_CARD_TEMPLATE.md`. That template governs what we *record* about a source paper;
this one governs where the *code and configs* live. For the shared training pipeline see
`training/INTEGRATED_PIPELINE.md` and `training/README.md`; this file covers only the split between
reproducing a source paper and benchmarking in our own protocol. Applies to every model in
`training/`.

## 1. The two questions are different, so the code paths are separate

| | **Reproduction** | **Benchmark** |
|---|---|---|
| Question | Do the source paper's own numbers hold up? | How does this architecture rank under *our* protocol? |
| Regime | Whatever the paper fixed — its sampling interval, lookback, horizons, split, metric | Ours, identical across all models |
| Data | The paper's data, as released | Our testbed data |
| Hyperparameters | The paper's, or its cited instantiation's, where the paper is silent | Tuned by our sweep |
| Changes over time | **Frozen.** The paper does not change | **Expected to move** as the manuscript's protocol settles |
| Entry point | Standalone script under `training/<Model>/repro/` | `training/configs/*.yaml` → `registry.py` → shared runner |
| Lives in the paper as | Model-card §13, "Reproduction results" | A row in the results tables |

They share exactly one thing: `models/<Model>.py`. Nothing else crosses.

A result from one is never a result from the other. A number produced under the paper's regime does
not belong in the benchmark table, and vice versa; the model card keeps them in separate subsections
for this reason.

## 2. Layout

```
models/<Model>.py                                architecture. Defaults are PAPER-FAITHFUL (§3)
training/<Model>/README.md                       build, shapes, layer-by-layer
training/<Model>/MODEL_CARD.md                   the audit (MODEL_CARD_TEMPLATE.md)
training/<Model>/requirements.txt                extra deps beyond the core image
training/<Model>/repro/                          REPRODUCTION — self-contained
    run_<paper>_regime.py                        standalone runner; regime as module constants
    export_results_csv.py                        JSON runs -> tidy CSVs for plotting
    verify_<model>.py                            executable checks of the port vs the source
    RESULTS_<REGIME>.md                          tables, in the paper's own metric space
    <SOURCE>_REFERENCE.md                        ledger: what the paper fixes vs what code fills in
    results_*.json                               raw run records, one object per run
    verification.json                            machine-readable output of verify_*.py
    data/                                        the paper's data            [gitignored]
    checkpoints/                                 trained weights             [gitignored]
    <paper>_text.txt, <upstream>_ref/            source material             [gitignored]
training/configs/<protocol>_<model>_<repr>.yaml  BENCHMARK config
training/ray/registry.py                         BENCHMARK search space and anchors
tests/test_<model>.py                            unit tests for the architecture
```

## 3. Ownership rule

**`models/<Model>.py` defaults are paper-faithful.** Where the paper states a value, use it. Where it
is silent, take the value from the code it cites as its instantiation, and cite the file and line in
a comment. These defaults will move when our reading of the paper improves.

**The reproduction script hardcodes the paper's regime as module constants**, not config. Lookback,
split ratio, seasonal period and similar are properties of the paper, not knobs: a reader changing
them is no longer reproducing anything. Expose only what genuinely varies across its runs — horizon,
seed, dataset file.

**The benchmark config states every hyperparameter explicitly and inherits nothing.** This is the rule
that is easiest to break and hardest to notice when broken: if a registry anchor omits a key, the
model's default silently supplies it, so improving a paper-faithful default silently moves the
benchmark. Write out every field, including the ones that happen to match today.

## 4. What is tracked

Track: code, configs, model card, results tables, raw run records (JSON), exported CSVs, the
verification output.

Do not track, and add to `.gitignore` with a comment giving the re-fetch recipe:

- the source paper's text (publisher copyright)
- vendored third-party source (check the licence; "no LICENSE file" means redistribution is *not*
  granted, however public the repository is)
- trained weights (hundreds of MB per run)
- upstream datasets (pin the commit or DOI in the comment instead)

The test: an exclusion is acceptable only if the file can be regenerated from what *is* tracked —
a pinned commit, a DOI, a Zotero key, or a script in `repro/`. Record that pointer in the ignore
comment, not only in prose.

## 5. Conventions for the reproduction runner

- **Resumable.** Key completed runs by `(dataset, horizon, seed, backbone, space)` and skip them, so
  an interrupted sweep continues instead of restarting.
- **Record the metric space in every row.** Which units the error is in is a property of the run, not
  a footnote; a column makes it impossible to misread a result later.
- **Checkpoint every run.** Save the trainable tensors only — a frozen backbone is reconstructible
  from its identifier — together with the config and the normalisation statistics, without which a
  prediction cannot be returned to physical units. Record the path and a checksum in the result row.
- **Carry the trivial references in the same row** as the model's score: persistence, lookback mean
  and a seasonal naive, computed on the same windows. They cost nothing and they are the first thing
  a reviewer asks for.
- **Keep per-epoch history** in the result record so learning curves can be plotted without re-running.

## 6. Checklist for a new model

1. `models/<Model>.py`, defaults paper-faithful, each non-obvious value carrying a source comment.
2. `tests/test_<model>.py` — shapes, freezing, gradient reach, and any invariant the paper asserts.
3. `training/<Model>/README.md` and `MODEL_CARD.md`.
4. `repro/verify_<model>.py` — executable checks against the source, written **before** the first run.
5. `repro/run_<paper>_regime.py` — regime hardcoded, §5 conventions.
6. Benchmark config and registry anchors, every field explicit.
7. `.gitignore` entries for the four excluded categories, each with its re-fetch pointer.

## 7. Failure modes seen in this repo

Each of these happened; each is cheap to avoid.

- **A reproduction schedule leaking into the benchmark config.** Hyperparameters chosen for the
  paper's sampling interval and lookback were written into the benchmark YAML, where they both
  contradicted the Ray sweep and applied a foreign decision to our data.
- **Silent inheritance.** Registry anchors that omitted a key picked it up from the model default, so
  a reproduction-motivated change would have moved benchmark results with nothing in the diff.
- **Metric space assumed rather than established.** A paper that never states whether its MSE is in
  physical or normalised units can be read either way, and the two readings supported opposite
  conclusions. Settle it with a probe — fit a simple model the paper also reports and see which space
  reproduces its cell — before quoting any number.
- **Weights discarded.** Early runs kept the best state in memory and exited, leaving metrics with no
  recoverable model behind them — the same gap we criticise source papers for.
- **A test that passed by luck.** An assertion on the order of tied values held until the tie broke
  differently. Assert the set, or assert the property that actually holds.
- **Numbers written before the computation that verifies them returned.** Quote a figure only from a
  file you have read back.
