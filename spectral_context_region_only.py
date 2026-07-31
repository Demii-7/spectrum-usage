import sys
sys.path.insert(0, "/workspace")

import ray.tune as tune
import training.ray.spectral_context as sc

_orig_grid_search = tune.grid_search

def _patched_grid_search(values):
    if isinstance(values, list) and "full_context" in values and "region_only" in values:
        return _orig_grid_search(["region_only"])
    return _orig_grid_search(values)

tune.grid_search = _patched_grid_search
sc.SEEDS = (40,)

sc.main([
    "--config", "training/configs/ray_tuning_2d.yaml",
    "--model", "temporalconvnet",
    "--search-path", "spectrum-usage/spectrum-usage/ray/production-20260728-tcn-independent-v2/temporalconvnet/temporalconvnet-search",
    "--output-dir", "runs/ray/production-20260730-spectral-context-v2/temporalconvnet-region-only",
    "--table", "evaluation/results/tables/spectral_context_temporalconvnet_region_only.csv",
    "--prefix", "spectrum-usage/ray/production-20260730-spectral-context-v2/temporalconvnet-region-only",
    "--gpus-per-task", "0.5",
    "--max-concurrent", "1",
])
