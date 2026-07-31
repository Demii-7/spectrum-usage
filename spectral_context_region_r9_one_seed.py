import sys

sys.path.insert(0, "/workspace")

import ray.tune as tune
import training.ray.spectral_context as spectral_context


_original_grid_search = tune.grid_search
_original_regions = spectral_context.non_noise_regions


def _region_only(values):
    if isinstance(values, list) and "full_context" in values and "region_only" in values:
        return _original_grid_search(["region_only"])
    return _original_grid_search(values)


tune.grid_search = _region_only
spectral_context.SEEDS = (40,)
spectral_context.non_noise_regions = lambda path: [
    region for region in _original_regions(path) if str(region["band_id"]) == "R9"
]
spectral_context.main([
    "--config", "training/configs/ray_tuning_2d.yaml",
    "--model", "temporalconvnet",
    "--search-path", "spectrum-usage/spectrum-usage/ray/production-20260728-tcn-independent-v2/temporalconvnet/temporalconvnet-search",
    "--output-dir", "runs/ray/production-20260731-spectral-context-v3/temporalconvnet-r9-region-only",
    "--table", "evaluation/results/tables/spectral_context_temporalconvnet_r9_region_only.csv",
    "--prefix", "spectrum-usage/ray/production-20260731-spectral-context-v3/temporalconvnet-r9-region-only",
    "--gpus-per-task", "0.5",
    "--max-concurrent", "1",
])
