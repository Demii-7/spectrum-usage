#!/usr/bin/env bash
# Model-free smoothing diagnostics of the eight s2 map constructions (CPU), both bands. Runs on the cluster (the data live there), writes
# evaluation/results/smoothing_diagnostics.{csv,txt} and uploads them to s3://spectrum/analysis-outputs/ (fetch only these small files).
set -u
cd /workspace/spectrum-usage || exit 1
mkdir -p evaluation/results
export PYTHONPATH=".:evaluation/analysis:${PYTHONPATH:-}"
python -u evaluation/analysis/smoothing_diagnostics.py --out evaluation/results/smoothing_diagnostics --upload
