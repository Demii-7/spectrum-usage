#!/usr/bin/env bash
# Offset-response CSVs (15-minute rolled-out and direct, regenerated from S3) and the TSS-LCD configuration/timing tables, on the cluster; outputs go to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
python -u evaluation/analysis/offset_response_table.py --horizon 12 --out evaluation/results/offset_response.csv > evaluation/results/offset_response_rolled_h12.txt 2>&1
python -u evaluation/analysis/offset_response_table.py --horizon 48 --out evaluation/results/offset_response.csv > evaluation/results/offset_response_rolled_h48.txt 2>&1
python -u evaluation/analysis/offset_response_table.py --prefix offset-response-direct-oct2026/ --horizon 12 --out evaluation/results/offset_response_direct.csv > evaluation/results/offset_response_direct_h12.txt 2>&1
python -u evaluation/analysis/offset_response_table.py --prefix offset-response-direct-oct2026/ --horizon 48 --out evaluation/results/offset_response_direct.csv > evaluation/results/offset_response_direct_h48.txt 2>&1
python -u evaluation/analysis/tsslcd_config_table.py > evaluation/results/tsslcd_config_table.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/offset_response*.csv") + glob.glob("evaluation/results/offset_response*_h*.txt") + glob.glob("evaluation/results/tsslcd_config*") + glob.glob("evaluation/results/tsslcd_search*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
