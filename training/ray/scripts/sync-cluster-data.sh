#!/usr/bin/env bash
# Run from a workstation with SSH access to every host in RAY_INVENTORY.
# Extra arguments select the inventory, e.g. --site powder --node nuc2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "${RAY_INVENTORY:-$SCRIPT_DIR/../docker/chameleon-nodes.sh}"
: "${RAY_REPO_PATH:=/home/cc/spectrum-usage}"
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi
SELECTION=("$@")
if ((${#SELECTION[@]} == 0)); then SELECTION=(--all-sites); fi
HOSTS=("${RAY_NODE_SSH_HOSTS[@]}")
((${#HOSTS[@]} > 0)) || { printf 'No SSH hosts in inventory\n' >&2; exit 2; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

worker_exec() {
  local host="$1" command
  shift
  printf -v command '%q ' "$@"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$RAY_SSH_USER@$host" \
    "container=\$(docker ps -q --filter label=com.docker.swarm.service.name=spectrum_ray-worker); test -n \"\$container\" && docker exec -i -w /workspace/spectrum-usage \"\$container\" $command"
}

# WORKDIR inside the container differs from the host's bind-mount source.
# Use an absolute container path for every command.
CONTAINER_REPO=/workspace/spectrum-usage
SYNC_SCRIPT="$CONTAINER_REPO/data/sync_object_storage.py"
INVENTORY="$CONTAINER_REPO/data/.spectrum-sync/inventory.json"
CHECKSUMS="$CONTAINER_REPO/data/.spectrum-sync/checksums.json"
for host in "${HOSTS[@]}"; do
  rsync -az -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
    --relative "$LOCAL_REPO/./data/sync_object_storage.py" "$RAY_SSH_USER@$host:$RAY_REPO_PATH/"
done
worker_exec "${HOSTS[0]}" python "$SYNC_SCRIPT" --spectrum-storage \
  "${SELECTION[@]}" --dry-run --summary-only --inventory-out "$INVENTORY"
worker_exec "${HOSTS[0]}" python -c 'import pathlib,sys; sys.stdout.write(pathlib.Path(sys.argv[1]).read_text())' \
  "$INVENTORY" > "$TMP/inventory.json"

for host in "${HOSTS[@]:1}"; do
  worker_exec "$host" python -c \
    'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(sys.stdin.read())' \
    "$INVENTORY" < "$TMP/inventory.json"
done
PIDS=()
for host in "${HOSTS[@]}"; do
  OPTIONS=(--spectrum-storage --inventory-in "$INVENTORY" --summary-only)
  if [[ "$DRY_RUN" == true ]]; then OPTIONS+=(--dry-run); else OPTIONS+=(--checksums-out "$CHECKSUMS"); fi
  worker_exec "$host" python "$SYNC_SCRIPT" "${OPTIONS[@]}" > "$TMP/$host.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=false
for index in "${!HOSTS[@]}"; do
  if ! wait "${PIDS[$index]}"; then FAILED=true; fi
  printf '\n%s:\n' "${HOSTS[$index]}"
  cat "$TMP/${HOSTS[$index]}.log"
done
[[ "$FAILED" == false ]] || exit 1
if [[ "$DRY_RUN" == true ]]; then exit 0; fi
for host in "${HOSTS[@]}"; do
  worker_exec "$host" python -c 'import pathlib,sys; sys.stdout.write(pathlib.Path(sys.argv[1]).read_text())' \
    "$CHECKSUMS" > "$TMP/$host.json"
done
python3 - "$TMP" "${HOSTS[@]}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
hosts = sys.argv[2:]
expected = json.loads((root / f"{hosts[0]}.json").read_text())
for host in hosts[1:]:
    actual = json.loads((root / f"{host}.json").read_text())
    if actual != expected:
        raise SystemExit(f"ERROR: processed data checksums differ on {host}")
print(f"Matching SHA-256 checksums for {len(expected['objects'])} files across {len(hosts)} hosts")
PY
