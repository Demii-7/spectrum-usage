# DSwinLSTM-I at 15 minutes, both bands, one seed (task N; status)

Last updated 2026-11-12 03:05 UTC. FINAL (H100) cluster. Single seed (42), inherited configuration, no search.

## Configuration (`inherited:1min-r2`)
The 1-minute round-2 winner of `checkpoints-oct2026-r2/4d/dswinlstm_i/` (seed-42 manifest entry, exported 2026-11-06) is in
`training/configs/fixed/dswinlstm_i_inherited_1min_r2.json` (embed 32, hidden [32,64], Swin depths [1,2,2,1], heads [2,4,4,2], window 2, pixel feedback,
batch 2, Adam 1e-4, 50 epochs, 308,348 parameters). Adapted as `apply_powder15_4d` does for the other 4D models: lookback 96, horizons 4/12/24/48,
direct output length 48 (= the longest horizon; the 1-minute model emitted 60), train/val/test stride 1, default map `p15_<band>_core4-p2`.
The label is in the run record (`hyperparameters`), the weights sidecar and `result.json` of the spatial cell.

## Code
* `training/ray/tune_fixed.py`: `--fixed-config <json>` (default off; skips the search, trains FINAL runs for `--fixed-seeds`, default 42; resumable per seed),
  `apply_fixed_config`, run costs in the record and sidecar. Test: `tests/test_tune_fixed.py`.
* `training/ray/spatial15.py` (`eval_only_job`): with `--eval-from-tuning` also takes the map-level test metrics from the tuning final's record.
* `training/ray/spatial15_origins.py`: dswinlstm_i is one seed and is scored on the GPU. `evaluation/analysis/spatial15_table.py` and
  `map_construction_vs_per_receiver.py`: DSwinLSTM-I rows with a `note` column (single seed; inherited configuration; base map only).
* `training/ray/scripts/run_spatial15_tables.sh`: tables and paired comparison on the cluster, uploaded to `s3://spectrum/analysis-outputs/`.

## Smoke (600-800, 2 epochs, GPU share 0.1 on a busy H100)
345 s per epoch, peak GPU memory 0.46 GB, 202 ms per inference sample; whole job (data, 2 epochs, test scoring, upload) about 13 minutes.
Full run: 50 epochs = about 4.8 h wall per band, 9.6 h wall for both (GPU-time at the 0.15 share reserved: about 1.4 GPU-hours; at a whole H100
about 10 GPU-hours), far below the 60 GPU-hour ceiling (1-minute: 840 s per epoch at 15x more windows than here). The scoring pipeline was smoke-tested on the
2-epoch weights (`spatial15-smoke-dswin/`, `tuning-smoke-dswin/`, `spatial15-smoke-origins/`; those numbers are from an untrained model and mean nothing).

## Jobs (watchdog: `training/ray/watchdog/final.json`)
| id | what | state |
|---|---|---|
| `claude-final-dswin15-600_800-1`, `-2400_2600-1` | finals, seed 42, `tuning/powder15_4d/<band>/4d/dswinlstm_i/` | running since 02:54 UTC |
| `claude-final-dswin15-score-1b` | after both: eval-only (map, receivers, left-out/added), permuted x5, per-origin errors | queued (watchdog) |
| `claude-final-dswin15-tables-2` | after scoring: `run_spatial15_tables.sh` | queued (watchdog) |

Outputs: `spatial15-tuned-oct2026/<band>/core4-p2/dswinlstm_i/seed-42/`, `.../permuted/dswinlstm_i/seed-42/p*/`,
`spatial15-origin-errors-oct2026/<band>/core4-p2/dswinlstm_i/seed-42/per_origin.npz`; tables `s3://spectrum/analysis-outputs/spatial15_*.csv`
and `map_construction_vs_per_receiver*.csv` (fetch, then commit under `evaluation/results/`).

## Results
None yet (finals expected about 07:50 UTC, scoring and tables about an hour later).

## Caveats
Single seed; configuration inherited from the 1-minute search (not tuned at 15 minutes); no retrained spatial variants (base map, left-out/added receivers and
permuted coordinates only); map-level MAE is the model's own 100-cell map and is not comparable with the 1D/2D or receiver numbers.
