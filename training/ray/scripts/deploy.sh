#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
APPLY=false
ALLOW_BUSY=false
STACK_NAME="${RAY_STACK_NAME:-spectrum-ray}"
STACK_FILE="$RAY_REPO_PATH/training/ray/docker/stack.yaml"
RAY_IMAGE="${RAY_IMAGE:-$(ray_local_image)}"
export RAY_IMAGE

while (($#)); do
  case "$1" in
    --apply) APPLY=true ;;
    --allow-busy-gpus) ALLOW_BUSY=true ;;
    *) printf 'Usage: %s [--apply] [--allow-busy-gpus]\n' "$0" >&2; exit 2 ;;
  esac
  shift
done

printf 'Stack: %s\nLocal image: %s\nManifest: %s\n' "$STACK_NAME" "$RAY_IMAGE" "$STACK_FILE"
if [[ "$APPLY" != true ]]; then
  printf 'DRY RUN: docker stack deploy --resolve-image never --compose-file %q %q\n' "$STACK_FILE" "$STACK_NAME"
  exit 0
fi
verify_all_host_identities
verify_all_repo_commits
if [[ "$ALLOW_BUSY" != true ]]; then check_all_gpus_idle; else printf '%s\n' 'WARNING: explicit busy-GPU override enabled.'; fi
check_all_nvidia_runtimes

expected_image_id=""
for ip in "${RAY_NODE_IPS[@]}"; do
  image_id="$(run_on_host "$ip" "docker image inspect --format '{{.Id}}' '$RAY_IMAGE'")"
  if [[ -z "$expected_image_id" ]]; then expected_image_id="$image_id"; fi
  [[ "$image_id" == "$expected_image_id" ]] || { printf 'FAIL: %s has image ID %s, expected %s\n' "$ip" "$image_id" "$expected_image_id" >&2; exit 1; }
done
printf 'OK: all nodes have identical image ID %s\n' "$expected_image_id"

read -r -p "Type DEPLOY-RAY to continue: " confirmation
[[ "$confirmation" == DEPLOY-RAY ]] || { printf '%s\n' 'Aborted.' >&2; exit 1; }
docker stack deploy --resolve-image never --compose-file "$STACK_FILE" "$STACK_NAME"
