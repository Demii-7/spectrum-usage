#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

: "${RAY_HEAD_ADDRESS:=ray-head:6379}"
source /usr/local/bin/load-storage-env.sh

EXTRA_ARGS=()
if [[ -n "${RAY_OBJECT_STORE_MEMORY:-}" ]]; then
  EXTRA_ARGS+=(--object-store-memory="$RAY_OBJECT_STORE_MEMORY")
fi
if [[ -n "${RAY_MEMORY:-}" ]]; then
  EXTRA_ARGS+=(--memory="$RAY_MEMORY")
fi

exec ray start \
  --address="${RAY_HEAD_ADDRESS}" \
  --node-ip-address="${RAY_NODE_IP:-$(hostname -i | awk '{print $1}')}" \
  --num-gpus="${RAY_NUM_GPUS:-1}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT:-10002}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT:-10999}" \
  "${EXTRA_ARGS[@]}" \
  --block
