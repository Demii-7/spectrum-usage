#!/usr/bin/env bash
# Part 4 item 2, scoring: per-origin errors of plain RevIN-MLP (seeds 40-42, only what is missing) and of RevIN-MLP with neighbours at the four physical
# receivers (15 minutes), then the paired table; CSV + report go to s3://spectrum/analysis-outputs/. Safe to rerun (--skip-done).
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
python -u -m training.ray.origin_errors --res 15min --models revinmlp --seeds 40 41 42 --skip-done --max-parallel 16 &
python -u -m training.ray.neighbor_origin_errors --seeds 40 41 42 --skip-done --max-parallel 8 &
wait
python -u evaluation/analysis/revinmlp_neighbors_table.py --out evaluation/results/revinmlp_neighbors.csv > evaluation/results/revinmlp_neighbors_report.txt 2>&1
status=$?
tail -40 evaluation/results/revinmlp_neighbors_report.txt
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for name in ("revinmlp_neighbors.csv", "revinmlp_neighbors_report.txt"):
    p = f"evaluation/results/{name}"
    if os.path.exists(p):
        c.put_object(Bucket="spectrum", Key=f"analysis-outputs/{name}", Body=open(p, "rb").read()); print("uploaded", p)
PY
exit $status
