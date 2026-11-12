#!/usr/bin/env bash
# TSS-LCD 15-minute 2400-2600: winner check (verdict JSON), then the configuration/winner/timing tables, uploaded to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
python -u evaluation/analysis/tsslcd_winner_check.py --resolution 15min --band 2400_2600 || exit 1
python -u evaluation/analysis/tsslcd_config_table.py > evaluation/results/tsslcd_config_table.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/tsslcd_config*") + glob.glob("evaluation/results/tsslcd_search*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
