#!/usr/bin/env bash
# Paired Part 4 table (every 4D model + RevIN-MLP +/- neighbours vs per-receiver models) -> s3://spectrum/analysis-outputs/spatial_paired_all.csv
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
export S3_ENDPOINT_URL="${SPECTRUM_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}" AWS_ACCESS_KEY_ID="${SPECTRUM_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}" AWS_SECRET_ACCESS_KEY="${SPECTRUM_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
python -u evaluation/analysis/spatial_paired_all.py --out /workspace/spectrum-usage/evaluation/results/spatial_paired_all.csv --upload
