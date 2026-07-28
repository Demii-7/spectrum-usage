#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
RUN=false
ADDRESS="${RAY_JOBS_ADDRESS:-http://192.168.1.201:8265}"

while (($#)); do
  case "$1" in
    --run) RUN=true ;;
    --address) ADDRESS="${2:?--address requires a URL}"; shift ;;
    *) printf 'Usage: %s [--run] [--address URL]\n' "$0" >&2; exit 2 ;;
  esac
  shift
done

printf 'Jobs API: %s\n' "$ADDRESS"
printf '%s\n' 'Acceptance: discover all three host GPU UUIDs, then submit cluster_probe with hostname/UUID, fractional GPU, model import, and MinIO cross-worker checks.'
if [[ "$RUN" != true ]]; then
  printf '%s\n' 'DRY RUN: pass --run to check the Jobs API and submit the acceptance probe.'
  exit 0
fi
read -r -p "Type RUN-RAY-ACCEPTANCE to continue: " confirmation
[[ "$confirmation" == RUN-RAY-ACCEPTANCE ]] || { printf '%s\n' 'Aborted.' >&2; exit 1; }

manager_hostname="$(hostname -s)"
head_uuid="$(gpu_uuid_on_host 192.168.1.201)"
worker_2_uuid="$(gpu_uuid_on_host 192.168.1.120)"
worker_1_uuid="$(gpu_uuid_on_host 192.168.1.130)"
probe_command=(python -m training.ray.cluster_probe --address auto
  --expected-gpu-node "$manager_hostname=$head_uuid"
  --expected-gpu-node "ray-worker-2=$worker_2_uuid"
  --expected-gpu-node "ray-worker-1=$worker_1_uuid"
  --fractional-gpu-check 0.25
  --check-model-imports
  --check-minio-cross-worker)
printf 'Acceptance command:'
printf ' %q' "${probe_command[@]}"
printf '\n'
curl --fail --silent --show-error "${ADDRESS%/}/api/version" >/dev/null
payload="$(python3 -c 'import json, shlex, sys; print(json.dumps({"entrypoint": "cd /workspace/spectrum-usage && exec " + shlex.join(sys.argv[1:])}))' "${probe_command[@]}")"
response="$(curl --fail-with-body --silent --show-error --header 'Content-Type: application/json' --data "$payload" "${ADDRESS%/}/api/jobs/")"
submission_id="$(python3 -c 'import json, sys; print(json.load(sys.stdin)["submission_id"])' <<<"$response")"
printf 'Submitted acceptance job %s\n' "$submission_id"

for _ in {1..120}; do
  response="$(curl --fail-with-body --silent --show-error "${ADDRESS%/}/api/jobs/${submission_id}")"
  status="$(python3 -c 'import json, sys; print(json.load(sys.stdin)["status"])' <<<"$response")"
  case "$status" in
    SUCCEEDED) break ;;
    FAILED|STOPPED) printf 'Acceptance job ended with status %s\n' "$status" >&2; curl --silent --show-error "${ADDRESS%/}/api/jobs/${submission_id}/logs" >&2; exit 1 ;;
    *) sleep 2 ;;
  esac
done
[[ "${status:-}" == SUCCEEDED ]] || { printf '%s\n' 'Acceptance job timed out.' >&2; exit 1; }
curl --fail-with-body --silent --show-error "${ADDRESS%/}/api/jobs/${submission_id}/logs"
printf '\nAcceptance passed for Jobs API, three H100 workers, fractional GPU scheduling, model imports, and MinIO cross-worker access.\n'
