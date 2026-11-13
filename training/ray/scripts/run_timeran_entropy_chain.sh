#!/usr/bin/env bash
# After the TimeRAN 1-minute stride-1 rescoring (claude-uc-rescore-timeran1min-1): build the TimeRAN stride-1 table and rerun the entropy-advantage tables
# (per bin and per region) with TimeRAN (timeran_lp) and TSS-LCD tuned finals; upload the small files to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:.:${PYTHONPATH:-}"
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
mkdir -p evaluation/results
python -u evaluation/analysis/part1_timeran_1min_stride1.py > evaluation/results/part1_timeran_1min_stride1_stdout.txt 2>&1
bash training/ray/scripts/run_entropy_tables.sh
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/part1_timeran_1min_stride1*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
