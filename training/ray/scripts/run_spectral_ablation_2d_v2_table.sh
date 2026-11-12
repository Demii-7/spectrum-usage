#!/usr/bin/env bash
# Spectral-context ablation v2 of the rolled-out 2D models: build the tables on the cluster and upload the small CSVs to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:${PYTHONPATH:-}"
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
mkdir -p evaluation/results
python -u evaluation/analysis/spectral_ablation_2d_v2.py --out evaluation/results/spectral_ablation_2d_v2.csv > evaluation/results/spectral_ablation_2d_v2.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/spectral_ablation_2d_v2*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
