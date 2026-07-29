#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
APPLY=false
ALLOW_BUSY=false
IMAGE="$(ray_local_image)"

while (($#)); do
  case "$1" in
    --apply) APPLY=true ;;
    --allow-busy-gpus) ALLOW_BUSY=true ;;
    *) printf 'Usage: %s [--apply] [--allow-busy-gpus]\n' "$0" >&2; exit 2 ;;
  esac
  shift
done

printf 'Image: %s\n' "$IMAGE"
printf '%s\n' 'Plan: build once on 192.168.1.201, then stream that exact image to 192.168.1.120 and 192.168.1.130 with docker image save/load.'
if [[ "$APPLY" != true ]]; then
  printf 'DRY RUN: docker build --file training/ray/docker/Dockerfile.core --tag %q .\n' "$IMAGE"
  printf '%s\n' 'DRY RUN: docker image save IMAGE | ssh HOST docker image load'
  printf 'After apply: export RAY_IMAGE=%q\n' "$IMAGE"
  exit 0
fi
verify_all_host_identities
if [[ "$ALLOW_BUSY" != true ]]; then check_all_gpus_idle; else printf '%s\n' 'WARNING: explicit busy-GPU override enabled.'; fi
check_all_nvidia_runtimes
read -r -p "Type BUILD-AND-DISTRIBUTE-RAY to continue: " confirmation
[[ "$confirmation" == BUILD-AND-DISTRIBUTE-RAY ]] || { printf '%s\n' 'Aborted.' >&2; exit 1; }

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  printf 'Using existing immutable local image %s.\n' "$IMAGE"
else
  docker build --file "$RAY_REPO_PATH/training/ray/docker/Dockerfile.core" --tag "$IMAGE" "$RAY_REPO_PATH"
fi
for ip in 192.168.1.120 192.168.1.130 192.168.1.189; do
  docker image save "$IMAGE" | ssh "${RAY_SSH_OPTIONS[@]}" "$RAY_SSH_USER@$ip" docker image load >/dev/null
  run_on_host "$ip" "docker image inspect '$IMAGE' >/dev/null"
  printf 'Loaded %s on %s.\n' "$IMAGE" "$ip"
done
printf 'export RAY_IMAGE=%q\n' "$IMAGE"
