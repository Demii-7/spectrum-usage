#!/usr/bin/env bash
# Level-drift analysis (task J, items 5-9) on the cluster: reads s3://spectrum/level-drift-oct2026/, writes evaluation/results/level_drift_*.csv, uploads them to
# s3://spectrum/analysis-outputs/ and keeps the printed report there as level_drift_report.txt.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
python -u evaluation/analysis/level_drift_analysis.py > evaluation/results/level_drift_report.txt 2>&1
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
p = "evaluation/results/level_drift_report.txt"
c.put_object(Bucket="spectrum", Key="analysis-outputs/level_drift_report.txt", Body=open(p, "rb").read()); print("uploaded", p, os.path.getsize(p))
PY
