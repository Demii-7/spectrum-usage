#!/usr/bin/env bash
# Tables of the nearest-cell readout of the linear 4D finals (CPU), after the two per-origin scoring jobs: spatial15_nearest_readout.{csv,_paired.csv,txt}
# and the receiver-level comparison against the per-receiver 1D/2D models and persistence under both readouts
# (map_construction_vs_per_receiver_{asevaluated,nearest}.csv); uploads to s3://spectrum/analysis-outputs/.
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
export PYTHONPATH=".:evaluation/analysis:${PYTHONPATH:-}"
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
python -u evaluation/analysis/spatial15_nearest_readout.py --out evaluation/results/spatial15_nearest_readout --upload
python -u evaluation/analysis/map_construction_vs_per_receiver.py --origin-prefix spatial15-origin-errors-oct2026-asevaluated --models linearar4d \
  --out evaluation/results/map_construction_vs_per_receiver_asevaluated.csv > evaluation/results/map_construction_vs_per_receiver_asevaluated.txt 2>&1
python -u evaluation/analysis/map_construction_vs_per_receiver.py --origin-prefix spatial15-origin-errors-oct2026-nearest --models linearar4d \
  --out evaluation/results/map_construction_vs_per_receiver_nearest.csv > evaluation/results/map_construction_vs_per_receiver_nearest.txt 2>&1
python - <<'PY'
import glob, os, sys
sys.path.insert(0, ".")
from training.ray.storage import spectrum_s3_client
c = spectrum_s3_client()
for f in sorted(glob.glob("evaluation/results/map_construction_vs_per_receiver_asevaluated*") + glob.glob("evaluation/results/map_construction_vs_per_receiver_nearest*")):
    c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read()); print("uploaded", f, os.path.getsize(f))
PY
