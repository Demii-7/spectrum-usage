# Ray tuning

The package runs the integrated trainer in-process and passes a callback to
`training.common.train_integrated.train_one_model`. Prunable forecasting epochs
are reported with Ray 2.54's `ray.tune.report`; pretraining and specialized
stage events are retained for manifests but never advance ASHA. The completed
integrated run directory, including its `.pt` checkpoints, is attached as the
final Ray checkpoint.

## Stages

Search uses exactly 12 candidates by default and seed 42 only. Tiny, small, and
reference architecture bundles are guaranteed through
`BasicVariantGenerator(points_to_evaluate=...)`; remaining candidate slots are
random samples from narrow optimizer domains and valid coupled architecture
bundles. Models with a retained historical configuration reserve one slot for
that named preset, keeping the total at 12. Search does not multiply candidates
across rerank seeds.

The generated plan defines a separate rerank stage. The launch command runs the
search stage only; reranking is submitted separately from the resulting search
selections. Reranking evaluates the search winner, the simplest configuration
within 1% of the winner, and the exact reference configuration with seeds 41,
42, and 43. No one-standard-error rule is used.

The preferred objective is `val_mean_horizon_mae_db`. If a forecasting trainer
does not emit it, the adapter uses `val_loss` and writes an explicit
non-comparability warning to the trial result manifest. TSS-LCD reports this
metric during its diffusion stage using a deterministic 50-step DDIM validation
protocol. STS-PredNet is not publication-ready until it emits comparable
physical dB metrics. DeepSPred is blocked because frame output cannot currently
support exact-minute horizon evaluation.

Completed trials attach the trainer-restored checkpoint to metrics from its
selected best epoch. Metrics from the last trained epoch are retained with a
`final_` prefix for diagnostics and do not replace the selected objective.

Executable configurations require explicit UTC `data.split.ranges` entries for
`train`, `validation`, and `test`. Dry-run records validation errors in the plan
without importing Ray or training:

```bash
python3 -m training.ray.run \
  --config path/to/config.yaml \
  --models vanillalstm \
  --candidates 12 \
  --dry-run
```

For MinIO, pass `--minio-bucket` and optional endpoint/prefix flags. Credentials
must be supplied through `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; they
are never persisted in manifests.

`run_non_asha_subprocess` exists only for isolated diagnostics. It does not
provide epoch reporting or ASHA pruning and is not used by the Tune path.
