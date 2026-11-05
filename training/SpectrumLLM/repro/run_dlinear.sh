#!/bin/bash
# Run the settled DLinear reproduction (exact Time-Series-Library protocol) in
# the training container and save its results.
# See training/SpectrumLLM/run_dlinear.py for the protocol; results land in
# repro/results_dlinear_settled.json (one row per dataset x horizon).
set -e
cd "$(dirname "$0")/../../.."   # repository root
docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" \
  --entrypoint python spectrum-split:latest \
  training/SpectrumLLM/run_dlinear.py \
  --out training/SpectrumLLM/repro/results_dlinear_settled.json "$@"
