#!/usr/bin/env bash
# Four-receiver minus solo-receiver table (STS-PredNet, and DSwinLSTM-I when its per-origin files exist) on the cluster; uploads to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:.:${PYTHONPATH:-}"
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
mkdir -p evaluation/results
python -u evaluation/analysis/solo_vs_core4.py --out evaluation/results/solo_vs_core4.csv --models "${@:-stsprednet}" --upload > evaluation/results/solo_vs_core4.txt 2>&1
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
spectrum_s3_client().put_object(Bucket="spectrum", Key="analysis-outputs/solo_vs_core4.txt", Body=open("evaluation/results/solo_vs_core4.txt", "rb").read()); print("uploaded solo_vs_core4.txt")
PY
