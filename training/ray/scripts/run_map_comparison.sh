#!/usr/bin/env bash
# Run the paired map-construction comparison on the cluster (reads per-origin dumps from S3) and upload its outputs to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
python -u evaluation/analysis/map_construction_vs_per_receiver.py --out evaluation/results/map_construction_vs_per_receiver.csv > evaluation/results/map_construction_vs_per_receiver.txt 2>&1
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in ("evaluation/results/map_construction_vs_per_receiver.csv", "evaluation/results/map_construction_vs_per_receiver.txt"):
    if os.path.exists(f):
        c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
