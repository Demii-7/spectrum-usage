#!/usr/bin/env bash
# Wait for the host guard before allowing Ray to accept work.
set -euo pipefail
python - <<'PY'
import os
import time

allowed = set(range(72)) | set(range(96, 168))
for _ in range(120):
    if set(os.sched_getaffinity(0)) == allowed:
        break
    time.sleep(0.5)
else:
    raise SystemExit("CPU-set guard did not apply the 144-CPU allocation")
PY
source /usr/local/bin/load-storage-env.sh
if [[ "${H100_THREE_ROLE:?}" == head ]]; then
  exec ray start --head --node-ip-address="$(hostname -i | cut -d' ' -f1)" \
    --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --ray-client-server-port=10001 --min-worker-port=10002 --max-worker-port=10999 \
    --num-cpus=0 --num-gpus=0 --memory=1073741824 --object-store-memory=1073741824 --block
fi
exec ray start --address=ray-head:6379 --node-ip-address="$(hostname -i | cut -d' ' -f1)" \
  --num-cpus=140 --num-gpus=3 --object-store-memory=17179869184 \
  --min-worker-port=10002 --max-worker-port=10999 --block
