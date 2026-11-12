#!/usr/bin/env bash
# Build the target-trained external table on the cluster and upload it to analysis-outputs.
cd /workspace/spectrum-usage
python -u evaluation/analysis/transfer_external_target_trained.py --out /tmp/transfer_external_target_trained.csv && python - <<'P'
from training.ray.storage import spectrum_s3_client
spectrum_s3_client().upload_file("/tmp/transfer_external_target_trained.csv", "spectrum", "analysis-outputs/transfer_external_target_trained.csv")
print("uploaded transfer_external_target_trained.csv")
P
