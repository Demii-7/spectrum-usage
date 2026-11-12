#!/usr/bin/env bash
# Entropy-advantage tables (per bin and per region) on the cluster; outputs go to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
if not os.path.exists("evaluation/results/october_entropy.csv"):   # untracked or not synced: fetch it from the bucket
    open("evaluation/results/october_entropy.csv", "wb").write(spectrum_s3_client().get_object(Bucket="spectrum", Key="analysis-inputs/october_entropy.csv")["Body"].read())
PY
python -u evaluation/analysis/entropy_advantage_october.py > evaluation/results/entropy_advantage_october.txt 2>&1
python -u evaluation/analysis/entropy_advantage_regions_october.py > evaluation/results/entropy_advantage_october_regions.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/entropy_advantage_october*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
