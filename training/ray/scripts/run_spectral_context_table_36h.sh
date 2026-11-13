#!/usr/bin/env bash
# Spectral-context table at a 36-hour shift on the cluster (CPU; reads S3 only); outputs to s3://spectrum/analysis-outputs/spectral_context_table_36h*.
set -u
cd /workspace/spectrum-usage || exit 1
export PYTHONPATH="evaluation/analysis:.:${PYTHONPATH:-}"
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
mkdir -p /tmp/spectral36
python -u evaluation/analysis/spectral_context_table_36h.py --out /tmp/spectral36 --upload "$@"
