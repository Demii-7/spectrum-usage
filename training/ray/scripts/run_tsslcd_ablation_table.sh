#!/usr/bin/env bash
# TSS-LCD spectral-context ablation: tables (with the other 15-minute ablation rows) and checks, on the cluster; outputs to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:${PYTHONPATH:-}"
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
mkdir -p evaluation/results
python -u evaluation/analysis/spectral_ablation_final.py --strong --out evaluation/results/spectral_ablation_strong.csv > evaluation/results/spectral_ablation_strong.txt 2>&1
python -u evaluation/analysis/tsslcd_ablation_check.py > evaluation/results/tsslcd_spectral_ablation_checks.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/spectral_ablation_strong*") + glob.glob("evaluation/results/tsslcd_spectral_ablation_checks*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", os.path.basename(f), os.path.getsize(f))
PY
