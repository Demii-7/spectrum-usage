#!/usr/bin/env bash
# Run from a workstation with SSH access to every host in RAY_INVENTORY.
# Extra arguments select the inventory, e.g. --site powder --node nuc2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "${RAY_INVENTORY:-$SCRIPT_DIR/../docker/chameleon-nodes.sh}"
: "${RAY_REPO_PATH:=/home/cc/spectrum-usage}"
: "${RAY_STACK_NAME:=spectrum}"
DRY_RUN=false
CHECKPOINTS=false
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --checkpoints) CHECKPOINTS=true; shift ;;
    *) break ;;
  esac
done
SELECTION=("$@")
if [[ "$CHECKPOINTS" == true ]]; then
  SELECTION=(--checkpoints "${SELECTION[@]}")
  if declare -p RAY_CHECKPOINT_PREFIXES >/dev/null 2>&1; then
    for prefix in "${RAY_CHECKPOINT_PREFIXES[@]}"; do
      SELECTION+=(--checkpoint-prefix "$prefix")
    done
  fi
elif ((${#SELECTION[@]} == 0)); then
  if declare -p RAY_DATA_SITES >/dev/null 2>&1 && ((${#RAY_DATA_SITES[@]} > 0)); then
    SELECTION=(--sites "${RAY_DATA_SITES[@]}")
  else
    SELECTION=(--all-sites)
  fi
fi
HOSTS=("${RAY_NODE_SSH_HOSTS[@]}")
((${#HOSTS[@]} > 0)) || { printf 'No SSH hosts in inventory\n' >&2; exit 2; }
if [[ "$CHECKPOINTS" == true ]] && declare -p RAY_CHECKPOINT_SKIP_HOSTS >/dev/null 2>&1; then
  CHECKPOINT_HOSTS=()
  for host in "${HOSTS[@]}"; do
    skip=false
    for excluded in "${RAY_CHECKPOINT_SKIP_HOSTS[@]}"; do
      if [[ "$host" == "$excluded" ]]; then skip=true; break; fi
    done
    if [[ "$skip" == false ]]; then CHECKPOINT_HOSTS+=("$host"); fi
  done
  HOSTS=("${CHECKPOINT_HOSTS[@]}")
  if ((${#HOSTS[@]} == 0)); then
    printf 'Checkpoint sync is disabled for all hosts in this inventory.\n'
    exit 0
  fi
fi
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=15)
if [[ -n "${RAY_SSH_CONFIG:-}" ]]; then
  SSH_OPTIONS+=(-F "$RAY_SSH_CONFIG")
fi
if [[ -n "${RAY_SSH_KNOWN_HOSTS:-}" ]]; then
  SSH_OPTIONS+=(-o "UserKnownHostsFile=$RAY_SSH_KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new)
fi
printf -v RSYNC_SSH '%q ' ssh "${SSH_OPTIONS[@]}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

worker_exec() {
  local host="$1" command index service="${RAY_STACK_NAME}_ray-worker"
  shift
  for index in "${!RAY_NODE_SSH_HOSTS[@]}"; do
    if [[ "${RAY_NODE_SSH_HOSTS[$index]}" == "$host" && -n "${RAY_NODE_SERVICES[$index]:-}" ]]; then
      service="${RAY_NODE_SERVICES[$index]}"
      break
    fi
  done
  printf -v command '%q ' "$@"
  ssh "${SSH_OPTIONS[@]}" "$RAY_SSH_USER@$host" \
    "container=\$(docker ps -q --filter label=com.docker.swarm.service.name=$service); test -n \"\$container\" && docker exec -i -w /workspace/spectrum-usage \"\$container\" $command"
}

# WORKDIR inside the container differs from the host's bind-mount source.
# Use an absolute container path for every command.
CONTAINER_REPO=/workspace/spectrum-usage
SYNC_SCRIPT="$CONTAINER_REPO/data/sync_object_storage.py"
INVENTORY="$CONTAINER_REPO/data/.spectrum-sync/inventory.json"
CHECKSUMS="$CONTAINER_REPO/data/.spectrum-sync/checksums.json"
if [[ "$CHECKPOINTS" == true ]]; then
  INVENTORY="$CONTAINER_REPO/data/.spectrum-sync/checkpoint-inventory.json"
  CHECKSUMS="$CONTAINER_REPO/data/.spectrum-sync/checkpoint-checksums.json"
fi
for host in "${HOSTS[@]}"; do
  rsync -az -e "$RSYNC_SSH" \
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
  if [[ "$CHECKPOINTS" == true ]]; then OPTIONS+=(--checkpoints); fi
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
        raise SystemExit(f"ERROR: synced object checksums differ on {host}")
print(f"Matching SHA-256 checksums for {len(expected['objects'])} files across {len(hosts)} hosts")
PY
