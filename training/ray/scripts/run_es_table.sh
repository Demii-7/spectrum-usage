#!/usr/bin/env bash
# Build the ElectroSense reproduction tables (old single-run seeds and the seed-fixed reruns) on the cluster and upload them to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
python -u evaluation/analysis/es_reproduction_table.py --out evaluation/results/es_reproduction_table.csv > evaluation/results/es_reproduction_table.txt 2>&1
python -u evaluation/analysis/es_reproduction_table.py --seedfix --out evaluation/results/es_reproduction_table_seedfix.csv > evaluation/results/es_reproduction_table_seedfix.txt 2>&1
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in ("es_reproduction_table.csv", "es_reproduction_table.txt", "es_reproduction_table_seedfix.csv", "es_reproduction_table_seedfix.txt"):
    p = "evaluation/results/" + f
    if os.path.exists(p):
        c.put_object(Bucket="spectrum", Key="analysis-outputs/" + f, Body=open(p, "rb").read()); print("uploaded", f, os.path.getsize(p))
PY
