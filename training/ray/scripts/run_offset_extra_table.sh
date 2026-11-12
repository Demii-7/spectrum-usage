#!/usr/bin/env bash
# Offset-response extra (Residual ConvLSTM, DSwinLSTM-I, TSS-LCD) tables on the cluster; outputs go to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
for h in 4 12 24 48; do
  python -u evaluation/analysis/offset_response_extra_table.py --horizon $h --out evaluation/results/offset_response_extra.csv > evaluation/results/offset_response_extra_h$h.txt 2>&1
done
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/offset_response_extra*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
