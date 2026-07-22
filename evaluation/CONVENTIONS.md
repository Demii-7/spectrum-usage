# Evaluation Conventions

Keep source code separate from generated artifacts.

## Source Code

- Put table and metric generation in `evaluation/analysis/`.
- Put figure rendering in `evaluation/plotting/`.
- Keep orchestration scripts in `evaluation/scripts/` only when they run a
  multi-step workflow.
- Put reusable training and evaluation code in `training/common/`.

## Inputs

- Store human-authored frequency annotations in `data/annotations/<dataset>/`.
- Store raw or synchronized measurements under `data/<dataset>/` on training
  hosts. Git should not track large measurement files.
- Pass input paths through command-line arguments. Do not encode a local
  username or host path in analysis code.

## Generated Artifacts

- Store model artifacts under `runs/<run-name>/` as described in `RUNS.md`.
- Store analysis tables derived from one model run inside that run directory.
- Store figures derived from one model run beside their source table.
- Store cross-run or dataset-wide tables under `evaluation/results/tables/` and
  figures under `evaluation/results/figures/`.
- Do not treat an untracked generated table as a source input when a committed
  analysis script can regenerate it from raw data, annotations, and run
  artifacts.

## Reproducible Analysis

An analysis command should identify its run directory, raw-data root,
annotations, horizon, and output path. Keep the generated table used by a plot
next to the figure. Commit the code and small human-authored metadata; archive
large run and result directories separately.
