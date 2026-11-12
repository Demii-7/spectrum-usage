# Campaign tools

The minimal chain: **train -> checkpoint + MSE record -> plot**.

## 1. Train

```bash
cd training/DeepSPred/repro
python run_paper_faithful.py \
    --data ../../../data/datasets/deepspred/FM-Spectrum-Dataset \
    --horizon 20 --seed 0 \
    --render-spec tukey224d --range-scope full
```

- Data: the authors' `.xls` release; FM/LTE are mirrored on the object store
  (`spectrum/datasets/deepspred/`, see `data/sync_object_storage.py` / COLLABORATOR_RUN.md).
  Frame caches build automatically on first use (CPU, minutes) under
  `data/datasets/deepspred/frames/`.
- `--render-spec` selects the rendering (`rect126`, `rect127c~12`, `hann126`, `hamm128`,
  `hann128s`, `tukey224`, `tukey224d`, `tukey224d@54-218`, `tukey224@54-218`, ...);
  `--range-scope {full,train}` sets the dB min-max scope. Diagnostic flags: `--stop-rule none`,
  `--drop-path`, `--head residual`, `--input-scale raw`, `--init-from <ckpt>` (transfer).
- The run key (and therefore the checkpoint name and resume identity) encodes dataset,
  horizon, seed, arch, lr, patch/window, rendering, scope and every diagnostic tag.

## 2. Checkpoint + results record

- Checkpoint: `checkpoints/repro/deepspred/<sha1(run_key)[:12]>.pt` (self-describing: run key,
  arch, regime, head, input-scale).
- Record: appended to `repro/results.json` under the run key (default `--results`); re-running
  an existing key is skipped (resumable). The record carries the frame-wise MSE in all four
  candidate spaces, PSNR, the persistence/lookback-mean references and the dataset metadata.

**Caveat:** the runner computes its record in one in-memory pass that peaks at ~75 GB (SSIM
over 1,722 windows x 20 frames) and can OOM when two runs finish together. If the process dies
after the checkpoint was saved, regenerate the record with:

```bash
python campaign_tools/regen_fast.py <sha12> results.json   # batch-1, MSE-only, ~10 min, ~8 GB
```

(MSE-only records carry `ssim: null` and `_recovered: true`; the streaming scorer was
validated exact against a surviving runner record to 4e-8.)

## 3. Plot

```bash
python campaign_tools/plot.py                 # FM campaign records -> plots/frame_vs_mse.png
python campaign_tools/plot.py --all           # include legacy and LTE records
python campaign_tools/plot.py --results myresults.json --out myplot
```

Solid lines: trained models (label = rendering + seed + diagnostic tags + MSE@8); dotted
same-color: that record's lookback-mean reference; black dashed: the paper's claimed curve.

Provenance and findings of the 2026-11-11/12 campaign: `../REPRODUCTION.md` section 8
(`../../REPRODUCTION.md` from here); run archive `../campaign_trainlogs.tar.gz`;
report `../campaign_report.md`. Rendering experiments (trivial-reference probes,
normalization variants, ensemble/blend scoring, cache builders) live in `../../probe/`.
Reference: 2405.19138 (companion paper by the same group, uploaded 2026-11-12 for
author-pipeline hints; re-fetchable from arxiv.org/abs/2405.19138).
