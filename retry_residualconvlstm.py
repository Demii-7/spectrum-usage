import sys, importlib
sys.path.insert(0, "/workspace")

import training.ray.replicate as rep

rep.SEEDS = (42, 43, 44)
rep.main([
    "--config", "training/configs/ray_tuning_4d.yaml",
    "--model", "residualconvlstm",
    "--search-path", "spectrum-usage/spectrum-usage/ray/production-20260728-v5-4d/residualconvlstm/residualconvlstm-search",
    "--output-dir", "runs/ray/production-20260729-replicate-v2/residualconvlstm-retry-v1",
    "--bucket", "spectrum-usage",
    "--prefix", "spectrum-usage/ray/production-20260729-replicate-v2/residualconvlstm-retry-v1",
    "--endpoint", "http://minio:9000",
    "--max-concurrent", "3",
])
