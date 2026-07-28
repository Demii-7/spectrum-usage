#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

: "${RAY_HEAD_ADDRESS:=ray-head:6379}"
if [[ -r "${AWS_ACCESS_KEY_ID_FILE:-}" ]]; then
  export AWS_ACCESS_KEY_ID="$(<"$AWS_ACCESS_KEY_ID_FILE")"
fi
if [[ -r "${AWS_SECRET_ACCESS_KEY_FILE:-}" ]]; then
  export AWS_SECRET_ACCESS_KEY="$(<"$AWS_SECRET_ACCESS_KEY_FILE")"
fi

exec ray start \
  --address="${RAY_HEAD_ADDRESS}" \
  --node-ip-address="${RAY_NODE_IP:-$(hostname -i | awk '{print $1}')}" \
  --num-gpus="${RAY_NUM_GPUS:-1}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT:-10002}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT:-10999}" \
  --block
