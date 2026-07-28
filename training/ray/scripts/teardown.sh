#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
APPLY=false
REMOVE_SECRETS=false
ALLOW_BUSY=false
STACK_NAME="${RAY_STACK_NAME:-spectrum-ray}"

while (($#)); do
  case "$1" in
    --apply) APPLY=true ;;
    --remove-secrets) REMOVE_SECRETS=true ;;
    --allow-busy-gpus) ALLOW_BUSY=true ;;
    *) printf 'Usage: %s [--apply] [--remove-secrets] [--allow-busy-gpus]\n' "$0" >&2; exit 2 ;;
  esac
  shift
done

printf 'Plan: remove stack %s. MinIO volume and Swarm remain intact.\n' "$STACK_NAME"
[[ "$REMOVE_SECRETS" == true ]] && printf '%s\n' 'Plan also removes the two Ray MinIO Docker secrets.'
if [[ "$APPLY" != true ]]; then
  printf '%s\n' 'DRY RUN: pass --apply to execute.'
  exit 0
fi

verify_all_host_identities
if [[ "$ALLOW_BUSY" != true ]]; then check_all_gpus_idle; else printf '%s\n' 'WARNING: explicit busy-GPU override enabled.'; fi

read -r -p "Type REMOVE-RAY-STACK to continue: " confirmation
[[ "$confirmation" == REMOVE-RAY-STACK ]] || { printf '%s\n' 'Aborted.' >&2; exit 1; }
docker stack rm "$STACK_NAME"
if [[ "$REMOVE_SECRETS" == true ]]; then
  docker secret rm ray_minio_root_user ray_minio_root_password
fi
