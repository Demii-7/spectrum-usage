#!/usr/bin/env bash
# Class-crossed frozen-input frequency ablation tables on the cluster (CPU; reads S3 only): the long CSV with checks (all seeds found, plus the seed-42-only version), Part 3 (benefit by class) and
# the identity check against stored per-origin files; outputs go to s3://spectrum/analysis-outputs/ (fetch the small files). New file names only: the existing frequency_ablation*.csv are untouched.
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:.:${PYTHONPATH:-}"
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
OUT=/tmp/freq_classes_out
mkdir -p $OUT
python -u evaluation/analysis/frequency_ablation_classes_table.py --out $OUT > $OUT/frequency_ablation_classes_stdout.txt 2>&1
python -u evaluation/analysis/frequency_ablation_classes_table.py --out $OUT --suffix _seed42 --seeds 42 >> $OUT/frequency_ablation_classes_stdout.txt 2>&1
python -u evaluation/analysis/frequency_benefit_by_class.py --out $OUT/frequency_benefit_by_class.csv > $OUT/frequency_benefit_by_class.txt 2>&1
python -u evaluation/analysis/frequency_ablation_identity.py --out $OUT/frequency_ablation_identity.csv > $OUT/frequency_ablation_identity.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("/tmp/freq_classes_out/*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
