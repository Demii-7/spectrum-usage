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

printf 'Manager/head/MinIO/GPU worker: %s\n' "$RAY_HEAD_IP"
printf '%s\n' 'Workers: ray-worker-2=192.168.1.120, ray-worker-1=192.168.1.130'
printf '%s\n' 'Plan: initialize Swarm, join both workers over SSH, label all three GPU nodes, pin head/MinIO to the manager, and create MinIO secrets interactively.'
if [[ "$APPLY" != true ]]; then
  printf '%s\n' 'DRY RUN: pass --apply to execute.'
  exit 0
fi
verify_all_host_identities
if [[ "$ALLOW_BUSY" != true ]]; then check_all_gpus_idle; else printf '%s\n' 'WARNING: explicit busy-GPU override enabled.'; fi
check_all_nvidia_runtimes
read -r -p "Type INITIALIZE-RAY-SWARM to continue: " confirmation
[[ "$confirmation" == INITIALIZE-RAY-SWARM ]] || { printf '%s\n' 'Aborted.' >&2; exit 1; }

manager_state="$(docker info --format '{{.Swarm.LocalNodeState}}')"
if [[ "$manager_state" == inactive ]]; then
  docker swarm init --advertise-addr "$RAY_HEAD_IP"
elif [[ "$manager_state" != active ]]; then
  printf 'Unsupported manager Swarm state: %s\n' "$manager_state" >&2
  exit 1
fi

worker_token="$(docker swarm join-token -q worker)"
for index in 1 2; do
  ip="${RAY_NODE_IPS[$index]}"
  expected_name="${RAY_NODE_NAMES[$index]}"
  verify_host_identity "$ip" "$expected_name" >/dev/null
  worker_state="$(run_on_host "$ip" "docker info --format '{{.Swarm.LocalNodeState}}'")"
  if [[ "$worker_state" == inactive ]]; then
    printf '%s\n' "$worker_token" | ssh "${RAY_SSH_OPTIONS[@]}" "$RAY_SSH_USER@$ip" \
      "read -r token; docker swarm join --token \"\$token\" '$RAY_HEAD_IP:2377'"
  elif [[ "$worker_state" != active ]]; then
    printf 'Unsupported Swarm state on %s: %s\n' "$ip" "$worker_state" >&2
    exit 1
  fi
done
unset worker_token

manager_node="$(docker node inspect self --format '{{.Description.Hostname}}')"
docker node update --label-add ray.head=true --label-add ray.gpu=true "$manager_node"
docker node update --label-add ray.gpu=true ray-worker-2
docker node update --label-add ray.gpu=true ray-worker-1

create_secret() {
  local secret_name="$1" prompt="$2" value
  if docker secret inspect "$secret_name" >/dev/null 2>&1; then
    printf 'Secret %s already exists; leaving it unchanged.\n' "$secret_name"
    return
  fi
  read -r -s -p "$prompt: " value
  printf '\n'
  [[ -n "$value" ]] || { printf '%s may not be empty.\n' "$secret_name" >&2; exit 1; }
  printf '%s' "$value" | docker secret create "$secret_name" - >/dev/null
  unset value
}

create_secret ray_minio_root_user 'MinIO root user'
create_secret ray_minio_root_password 'MinIO root password'
printf '%s\n' 'Swarm nodes, labels, and external secrets are ready.'
