#!/usr/bin/env bash
# Run the Part 1 table scripts on the cluster (they only read S3 summaries) and upload the resulting CSVs/text to s3://spectrum/analysis-outputs/ so that only the
# small result files need to be fetched. Usage (as a Ray job): bash training/ray/scripts/run_tables.sh
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:${PYTHONPATH:-}"
mkdir -p evaluation/results
python -u evaluation/analysis/tsslcd_table.py > evaluation/results/tsslcd_all.txt 2>&1
python -u evaluation/analysis/part1_per_horizon.py > evaluation/results/part1_per_horizon_15min.txt 2>&1
python -u evaluation/analysis/part1_per_horizon_1min.py > evaluation/results/part1_per_horizon_1min.txt 2>&1
python -u evaluation/analysis/part1_per_horizon_1min_stride1.py > evaluation/results/part1_per_horizon_1min_stride1.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/part1_per_horizon_*.csv") + glob.glob("evaluation/results/part1_per_horizon_*.txt") + ["evaluation/results/tsslcd_all.csv", "evaluation/results/tsslcd_all.txt"]):
    if os.path.exists(f):
        c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read())
        print("uploaded", f, os.path.getsize(f))
PY
