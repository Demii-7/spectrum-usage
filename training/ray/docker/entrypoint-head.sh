#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

source /usr/local/bin/load-storage-env.sh

exec ray start \
  --head \
  --node-ip-address="${RAY_NODE_IP:-$(hostname -i | awk '{print $1}')}" \
  --port="${RAY_GCS_PORT:-6379}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
  --ray-client-server-port="${RAY_CLIENT_PORT:-10001}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT:-10002}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT:-10999}" \
  --num-cpus="${RAY_HEAD_CPUS:-0}" \
  --num-gpus=0 \
  --block
