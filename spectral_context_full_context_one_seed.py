import sys

sys.path.insert(0, "/workspace")

import ray.tune as tune
import training.ray.spectral_context as spectral_context


_original_grid_search = tune.grid_search


def _full_context_only(values):
    if isinstance(values, list) and "full_context" in values and "region_only" in values:
        return _original_grid_search(["full_context"])
    return _original_grid_search(values)


tune.grid_search = _full_context_only
spectral_context.SEEDS = (40,)
spectral_context.main([
    "--config", "training/configs/ray_tuning_2d.yaml",
    "--model", "temporalconvnet",
    "--search-path", "spectrum-usage/spectrum-usage/ray/production-20260728-tcn-independent-v2/temporalconvnet/temporalconvnet-search",
    "--output-dir", "runs/ray/production-20260731-spectral-context-v3/temporalconvnet-full-context",
    "--table", "evaluation/results/tables/spectral_context_temporalconvnet_full_context.csv",
    "--prefix", "spectrum-usage/ray/production-20260731-spectral-context-v3/temporalconvnet-full-context",
    "--gpus-per-task", "0.5",
    "--max-concurrent", "1",
])
