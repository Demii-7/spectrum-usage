#!/usr/bin/env bash
# Catch-up tables on the cluster (CPU): Part 1 (15-minute incl. TCN seed 42 and the 1D LSTM 2400-2600 seeds, 1-minute stride 1 incl. Time-LLM), July transfer in all regimes, neighbour value
# (base 4D models: compare + ablation incl. ResidualConvLSTM 2400-2600 and DSwinLSTM-I). Outputs go to s3://spectrum/analysis-outputs/ (fetch the small files).
# Usage (Ray job): bash training/ray/scripts/run_catchup_tables.sh
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:${PYTHONPATH:-}"
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
mkdir -p evaluation/results
bash training/ray/scripts/run_tables.sh
python -u evaluation/analysis/transfer_july_all_table.py > evaluation/results/transfer_july_all_regimes.txt 2>&1
python -u evaluation/analysis/neighbor_value.py --out evaluation/results > evaluation/results/neighbor_value_catchup.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/transfer_july_all_regimes.*") + glob.glob("evaluation/results/neighbor_value*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
