# Inference-cost benchmark: runbook for the new H100 cluster

Goal: median inference time (and parameters, forward passes, peak memory) for every checkpointed POWDER model, on idle GPUs of one type.
Script: `training/ray/inference_benchmark.py` (tests: `tests/test_inference_benchmark.py`). Results go to
`s3://spectrum/inference-benchmark/<run>/inference_benchmark_<stamp>.csv`.

## What gets timed (158 models at the last check; weights are in `s3://spectrum`, nothing is copied to the cluster by hand)

| source | what | models |
|---|---|---|
| `oct` | 1-minute 600-800 MHz campaign and round 2 (`checkpoints-oct2026`, `checkpoints-oct2026-r2`): 1D, 2D, 4D | 20 |
| `band` | 1-minute 2400-2600 MHz runs (`oct2026-band-runs`) | 17 (plus one 600_800 4D entry to look at) |
| `p15` | 15-minute pooled models, rolled out and direct, both bands (`powder15-results-w`) | 102 |
| `p15tuned` | 15-minute tuned finals (`tuning/powder15`); the four 1D LSTM groups were still finishing on the A100 | 19 so far |

One checkpoint per model (seed 42 if it exists, else the lowest), because timing does not depend on the seed. Direct models trained for different horizons are
different models and are all timed. Each is timed at batch size 1 and 32 on about 500 test windows (1-minute stride 20, 15-minute stride 1), 3 warm-up calls
discarded, median of the rest, CUDA-synchronized timers around the pipeline's `forecast` (so one-step models include their roll-out).

Not covered, deliberately: per-receiver and pooled-6 models (same architectures and input shapes as the 2D models, so the same cost), lookback-mean baselines (no
parameters), TSS-LCD and Time-LLM/OFA (their own pipelines; TSS-LCD's weights are in `tsslcd-authors-full/weights/`), ElectroSense models (no weights saved), TSL
baselines (weights in `tsl-checkpoints/`, a different library).

## Checklist once the cluster is up

1. **Cluster**: all GPUs the same type, nothing else running (`ray status` shows every GPU free). Ray has no priority, so submit the benchmark first. The launcher
   refuses to start unless every GPU is free (`--force` is for smoke tests and marks the rows `idle=false`).
2. **Access**: the workers need the Chameleon bucket credentials (`SPECTRUM_*`) like every other job; the cluster probe checks it
   (`python -m training.ray.cluster_probe --check-spectrum-read`).
3. **Code**: `python3 training/ray/scripts/sync-cluster-code.py --cluster <name> --revision HEAD` (needs commits `e596b8f` or later; they are local-only today).
4. **Data**: the processed 1-minute power CSVs for the four core receivers, both bands (`sync-cluster-data.sh`, see `training/ray/AGENTS.md`). The 15-minute traces
   are converted from them on first use (`ensure_powder15`, once per worker) and the 4D map cache is built on first use.
5. **Dry look**: `python -m training.ray.inference_benchmark --manifest-only` lists what will run.
6. **Smoke** (optional, a minute or two, marks the rows non-idle): `--max-per-source 2 --force --output-prefix inference-benchmark/smoke`.
7. **Run**: `python -u -m training.ray.inference_benchmark --output-prefix inference-benchmark/h100` as a Ray job with a unique submission id.

## Checked so far (on the A100 cluster, CPU only, not a timing run)

The 1-minute 600-800, 2400-2600, 15-minute pooled and 15-minute tuned paths each load their checkpoint, rebuild the config, and produce a row. Timings from that
smoke test are meaningless (CPU, shared machine). Output: `inference-benchmark/smoke-cpu/`.

## Second run: the models added on 2026-11-08 (planned for the next new H100 cluster)

The first run (159 models, `inference-benchmark/h100/`) covered the October campaign, the 2400-2600 band runs and the first 15-minute sets. Not yet timed:

| what | source flag | notes |
|---|---|---|
| OFA, the two language-model controls (`ofa_rand`, `ofa_scratch`), PatchTST and its ablations, DLinear, MLP residual head, tuned direct ResidualLinearAR/LinearAR, NLinear, RevIN-MLP, and Time-LLM once its finals finish | `--sources direct` | 15-minute and 1-minute, both bands; one checkpoint per model and horizon; ElectroSense runs are not covered by the benchmark script |
| 15-minute tuned finals, now including the 1D LSTMs | `--sources p15tuned` | the earlier run predates the 600-800 and 2400-2600 1D LSTM finals |

Extra setup on a fresh cluster: the OFA and Time-LLM timings need `transformers` and GPT-2 on every worker (`python -m training.ray.llm_powder --setup`, as done for the 2026-11-07 cluster). The linear models are timed on CPU and marked with an asterisk in every table. The cluster must be idle (the launcher checks), and Time-LLM must have finished so its weights exist.

    python -m training.ray.inference_benchmark --sources direct p15tuned --output-prefix inference-benchmark/h100-second
