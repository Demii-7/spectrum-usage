#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

failed=0
for command in docker nvidia-smi ssh sha256sum; do
  if command -v "$command" >/dev/null 2>&1; then
    printf 'OK: %s is installed\n' "$command"
  else
    printf 'FAIL: %s is not installed\n' "$command" >&2
    failed=1
  fi
done
[[ -d "$RAY_REPO_PATH/.git" ]] || { printf 'FAIL: repository is missing at %s\n' "$RAY_REPO_PATH" >&2; failed=1; }

for index in "${!RAY_NODE_IPS[@]}"; do
  ip="${RAY_NODE_IPS[$index]}"
  expected="${RAY_NODE_NAMES[$index]}"
  verify_host_identity "$ip" "$expected" >/dev/null || failed=1
  check_nvidia_runtime "$ip" || failed=1
  state="$(run_on_host "$ip" "docker info --format '{{.Swarm.LocalNodeState}}'" 2>/dev/null || true)"
  printf 'INFO: %s Swarm state: %s\n' "$ip" "${state:-unavailable}"
done
check_all_gpus_idle || failed=1

exit "$failed"
