#!/usr/bin/env bash
# Build the strong-model spectral-ablation tables (direct models + 2400-2600 tuned 2D models) on the cluster and upload them to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
python -u evaluation/analysis/spectral_ablation_final.py --strong --out evaluation/results/spectral_ablation_strong.csv > evaluation/results/spectral_ablation_strong.txt 2>&1
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in ("spectral_ablation_strong.csv", "spectral_ablation_strong_by_region.csv", "spectral_ablation_strong.txt"):
    p = "evaluation/results/" + f
    if os.path.exists(p):
        c.put_object(Bucket="spectrum", Key="analysis-outputs/" + f, Body=open(p, "rb").read()); print("uploaded", f, os.path.getsize(p))
PY
