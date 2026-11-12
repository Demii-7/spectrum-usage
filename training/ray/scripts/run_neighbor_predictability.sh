#!/usr/bin/env bash
# Item 1 of Part 4 (CPU): neighbour predictability of the core receivers' deviations, 15 minutes, both bands. Runs on the cluster (the data live there),
# writes evaluation/results/neighbor_predictability{,_by_pair}.csv and uploads them (and the printed report) to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
python -u evaluation/analysis/neighbor_predictability.py > evaluation/results/neighbor_predictability_report.txt 2>&1
status=$?
cat evaluation/results/neighbor_predictability_report.txt | tail -60
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for name in ("neighbor_predictability.csv", "neighbor_predictability_by_pair.csv", "neighbor_predictability_report.txt"):
    p = f"evaluation/results/{name}"
    if os.path.exists(p):
        c.put_object(Bucket="spectrum", Key=f"analysis-outputs/{name}", Body=open(p, "rb").read()); print("uploaded", p, os.path.getsize(p))
PY
exit $status
