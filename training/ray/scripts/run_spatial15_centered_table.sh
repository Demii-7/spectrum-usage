#!/usr/bin/env bash
# Centred vs as-trained table for the frozen neural 4D finals (task 3 of the neural-4D brief) on the cluster; outputs go to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:.:${PYTHONPATH:-}"
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
mkdir -p evaluation/results
python -u evaluation/analysis/spatial15_centered_table.py --out evaluation/results/spatial15_centered.csv --upload > evaluation/results/spatial15_centered.txt 2>&1
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in ("evaluation/results/spatial15_centered.txt",):
    if os.path.exists(f):
        c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f)
PY
