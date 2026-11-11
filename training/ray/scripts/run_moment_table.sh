#!/usr/bin/env bash
# Zero-shot MOMENT: design choice on validation (choose) or the test-week tables (table), run on the cluster; results are uploaded to s3://spectrum/analysis-outputs/.
# Usage (Ray job): bash training/ray/scripts/run_moment_table.sh choose|table
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:${PYTHONPATH:-}"
mkdir -p evaluation/results
if [ "$1" = "choose" ]; then
  python -u evaluation/analysis/moment_zeroshot_table.py choose > evaluation/results/moment_zeroshot_choose.txt 2>&1
  python - <<'PY'
import sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
import os
for f in ("evaluation/analysis/moment_zeroshot_design.json", "evaluation/results/moment_zeroshot_choose.txt"):
    if os.path.exists(f):
        c.put_object(Bucket="spectrum", Key="analysis-outputs/" + f.split("/")[-1], Body=open(f, "rb").read()); print("uploaded", f)
PY
else
  python -u evaluation/analysis/moment_zeroshot_table.py table "${@:2}" > evaluation/results/moment_zeroshot_table.txt 2>&1
  python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in ("evaluation/results/moment_zeroshot_15min.csv", "evaluation/results/moment_zeroshot_table.txt"):
    if os.path.exists(f):
        c.put_object(Bucket="spectrum", Key="analysis-outputs/" + f.split("/")[-1], Body=open(f, "rb").read()); print("uploaded", f)
PY
fi
