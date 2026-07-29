#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
APPLY=false

if (($#)); then
  [[ "$1" == --apply && $# -eq 1 ]] || { printf 'Usage: %s [--apply]\n' "$0" >&2; exit 2; }
  APPLY=true
fi

printf '%s\n' 'Plan: verify all H100s are idle and no containers are running, then restart Docker on all four hosts.'
if [[ "$APPLY" != true ]]; then
  printf '%s\n' 'DRY RUN: pass --apply to perform the coordinated restart.'
  exit 0
fi

verify_all_host_identities
check_all_gpus_idle
for ip in "${RAY_NODE_IPS[@]}"; do
  running="$(run_on_host "$ip" "docker ps --quiet")"
  [[ -z "$running" ]] || { printf 'FAIL: %s has running containers; refusing Docker restart.\n' "$ip" >&2; exit 1; }
done
read -r -p "Type RESTART-DOCKER-ON-ALL-RAY-HOSTS to continue: " confirmation
[[ "$confirmation" == RESTART-DOCKER-ON-ALL-RAY-HOSTS ]] || { printf '%s\n' 'Aborted.' >&2; exit 1; }

for ip in "${RAY_NODE_IPS[@]}"; do
  if [[ "$ip" == "$RAY_HEAD_IP" ]]; then
    sudo systemctl restart docker
  else
    ssh "${RAY_SSH_OPTIONS[@]}" "$RAY_SSH_USER@$ip" sudo systemctl restart docker
  fi
done
check_all_nvidia_runtimes
printf '%s\n' 'Docker restarted and NVIDIA is the default runtime on every Ray host.'
