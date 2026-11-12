#!/usr/bin/env bash
# 15-minute test-week MAE of revinmlp, revinmlp_mix, itransformer, patchtst (strong direct models) -> s3://spectrum/analysis-outputs/strong_models_15min.{csv,txt}
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
cd evaluation/analysis && python -u strong_models_table.py --out /workspace/spectrum-usage/evaluation/results/strong_models_15min.csv --upload
