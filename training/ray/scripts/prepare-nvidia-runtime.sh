#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
APPLY=false
ALLOW_BUSY=false

while (($#)); do
  case "$1" in
    --apply) APPLY=true ;;
    --allow-busy-gpus) ALLOW_BUSY=true ;;
    *) printf 'Usage: %s [--apply] [--allow-busy-gpus]\n' "$0" >&2; exit 2 ;;
  esac
  shift
done

printf '%s\n' 'Plan: run nvidia-ctk runtime configure --runtime=docker --set-as-default on all four hosts.'
printf '%s\n' 'Docker will NOT be restarted. An operator must separately schedule and perform any required restart.'
if [[ "$APPLY" != true ]]; then
  printf '%s\n' 'DRY RUN: pass --apply to configure daemon.json.'
  exit 0
fi
verify_all_host_identities
if [[ "$ALLOW_BUSY" != true ]]; then check_all_gpus_idle; else printf '%s\n' 'WARNING: explicit busy-GPU override enabled.'; fi
read -r -p "Type CONFIGURE-NVIDIA-RUNTIME-NO-RESTART to continue: " confirmation
[[ "$confirmation" == CONFIGURE-NVIDIA-RUNTIME-NO-RESTART ]] || { printf '%s\n' 'Aborted.' >&2; exit 1; }

for index in "${!RAY_NODE_IPS[@]}"; do
  ip="${RAY_NODE_IPS[$index]}"
  verify_host_identity "$ip" "${RAY_NODE_NAMES[$index]}" >/dev/null
  if [[ "$ip" == "$RAY_HEAD_IP" ]]; then
    sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
  else
    ssh -tt "${RAY_SSH_OPTIONS[@]}" "$RAY_SSH_USER@$ip" 'sudo nvidia-ctk runtime configure --runtime=docker --set-as-default'
  fi
done
printf '%s\n' 'Configuration written. Docker was not restarted; default-runtime checks will fail until an operator restarts Docker on each changed host.'
