#!/usr/bin/env bash
# Tabulate the 1-minute offset test on the cluster (the table script reads S3 summaries) and upload the outputs to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
: > evaluation/results/offset_response_1min.txt
for h in 1 15 60; do
  python -u evaluation/analysis/offset_response_table.py --prefix offset-response-1min-oct2026/ --out evaluation/results/offset_response_1min.csv --horizon $h >> evaluation/results/offset_response_1min.txt 2>&1
done
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in ("offset_response_1min.csv", "offset_response_1min.txt"):
    p = "evaluation/results/" + f
    if os.path.exists(p):
        c.put_object(Bucket="spectrum", Key="analysis-outputs/" + f, Body=open(p, "rb").read()); print("uploaded", f, os.path.getsize(p))
PY
