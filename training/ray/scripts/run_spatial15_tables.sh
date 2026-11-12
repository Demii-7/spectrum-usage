#!/usr/bin/env bash
# Run the 15-minute spatial table (spatial15_table.py: cells, receivers, coverage, S2-S5 incl. the single-seed DSwinLSTM-I rows) and the paired
# map-construction comparison on the cluster, and upload the outputs to s3://spectrum/analysis-outputs/ (fetch only these small files).
# Usage (Ray job): bash training/ray/scripts/run_spatial15_tables.sh
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:${PYTHONPATH:-}"
# table scripts that build their own boto3 client read the generic variables: map the benchmark's SPECTRUM_* ones onto them
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
mkdir -p evaluation/results
python -u evaluation/analysis/spatial15_table.py --out evaluation/results/spatial15 > evaluation/results/spatial15_tables.txt 2>&1
python -u evaluation/analysis/map_construction_vs_per_receiver.py --out evaluation/results/map_construction_vs_per_receiver.csv > evaluation/results/map_construction_vs_per_receiver.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
files = sorted(glob.glob("evaluation/results/spatial15_*.csv") + ["evaluation/results/spatial15_tables.txt"] + glob.glob("evaluation/results/map_construction_vs_per_receiver*"))
for f in files:
    if os.path.exists(f):
        c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
