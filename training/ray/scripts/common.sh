#!/usr/bin/env bash

RAY_REPO_PATH="${RAY_REPO_PATH:-/home/cc/spectrum-usage}"
RAY_HEAD_IP="${RAY_HEAD_HOST:-192.168.1.201}"
RAY_GPU_IDLE_THRESHOLD="${RAY_GPU_IDLE_THRESHOLD:-5}"
RAY_SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
RAY_SSH_USER="${RAY_SSH_USER:-cc}"
RAY_NODE_IPS=("192.168.1.201" "192.168.1.120" "192.168.1.130")
RAY_NODE_NAMES=("" "ray-worker-2" "ray-worker-1")

run_on_host() {
  local ip="$1" command="$2"
  if [[ "$ip" == "$RAY_HEAD_IP" ]]; then
    bash -c "$command"
  else
    ssh "${RAY_SSH_OPTIONS[@]}" "$RAY_SSH_USER@$ip" "$command"
  fi
}

verify_host_identity() {
  local ip="$1" expected_name="$2" actual_name
  actual_name="$(run_on_host "$ip" 'hostname -s')"
  if [[ -n "$expected_name" && "$actual_name" != "$expected_name" ]]; then
    printf 'FAIL: %s reports hostname %s, expected %s\n' "$ip" "$actual_name" "$expected_name" >&2
    return 1
  fi
  printf '%s' "$actual_name"
}

check_nvidia_runtime() {
  local ip="$1" default_runtime runtimes
  default_runtime="$(run_on_host "$ip" "docker info --format '{{.DefaultRuntime}}'")"
  runtimes="$(run_on_host "$ip" "docker info --format '{{json .Runtimes}}'")"
  if [[ "$runtimes" != *nvidia* ]]; then
    printf 'FAIL: %s has no NVIDIA Docker runtime\n' "$ip" >&2
    return 1
  fi
  if [[ "$default_runtime" != nvidia ]]; then
    printf 'FAIL: %s Docker default runtime is %s, not nvidia\n' "$ip" "$default_runtime" >&2
    return 1
  fi
  printf 'OK: %s Docker default runtime is nvidia\n' "$ip"
}

gpu_uuid_on_host() {
  local ip="$1"
  run_on_host "$ip" "nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]'"
}

check_gpu_idle() {
  local ip="$1" expected_name="$2" actual_name gpu_output gpu_line gpu_name gpu_uuid utilization processes
  local -a gpu_lines
  actual_name="$(verify_host_identity "$ip" "$expected_name")" || return 1
  gpu_output="$(run_on_host "$ip" "nvidia-smi --query-gpu=name,uuid,utilization.gpu --format=csv,noheader,nounits")"
  mapfile -t gpu_lines <<<"$gpu_output"
  ((${#gpu_lines[@]} == 1)) || { printf 'FAIL: %s (%s) reports %s GPUs, expected exactly one H100\n' "$actual_name" "$ip" "${#gpu_lines[@]}" >&2; return 1; }
  gpu_line="${gpu_lines[0]}"
  IFS=',' read -r gpu_name gpu_uuid utilization <<<"$gpu_line"
  gpu_name="${gpu_name# }"; gpu_name="${gpu_name% }"
  gpu_uuid="${gpu_uuid# }"; gpu_uuid="${gpu_uuid% }"
  utilization="${utilization//[[:space:]]/}"
  [[ "$gpu_name" == *H100* ]] || { printf 'FAIL: %s (%s) GPU is %s, expected H100\n' "$actual_name" "$ip" "$gpu_name" >&2; return 1; }
  [[ "$utilization" =~ ^[0-9]+$ ]] || { printf 'FAIL: invalid GPU utilization from %s: %s\n' "$ip" "$utilization" >&2; return 1; }
  if ((utilization > RAY_GPU_IDLE_THRESHOLD)); then
    printf 'FAIL: %s (%s, %s) GPU utilization is %s%%, threshold is %s%%\n' "$actual_name" "$ip" "$gpu_uuid" "$utilization" "$RAY_GPU_IDLE_THRESHOLD" >&2
    return 1
  fi
  processes="$(run_on_host "$ip" "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true")"
  if [[ "$processes" =~ [0-9] ]]; then
    printf 'FAIL: %s (%s, %s) has GPU compute processes: %s\n' "$actual_name" "$ip" "$gpu_uuid" "$processes" >&2
    return 1
  fi
  printf 'OK: %s (%s, %s) H100 is idle at %s%%\n' "$actual_name" "$ip" "$gpu_uuid" "$utilization"
}

check_all_gpus_idle() {
  local index
  for index in "${!RAY_NODE_IPS[@]}"; do
    check_gpu_idle "${RAY_NODE_IPS[$index]}" "${RAY_NODE_NAMES[$index]}"
  done
}

check_all_nvidia_runtimes() {
  local ip
  for ip in "${RAY_NODE_IPS[@]}"; do check_nvidia_runtime "$ip"; done
}

verify_all_host_identities() {
  local index
  for index in "${!RAY_NODE_IPS[@]}"; do
    verify_host_identity "${RAY_NODE_IPS[$index]}" "${RAY_NODE_NAMES[$index]}" >/dev/null
  done
}

verify_all_repo_commits() {
  local expected_commit ip actual_commit dirty
  expected_commit="$(git -C "$RAY_REPO_PATH" rev-parse HEAD)"
  for ip in "${RAY_NODE_IPS[@]}"; do
    actual_commit="$(run_on_host "$ip" "git -C '$RAY_REPO_PATH' rev-parse HEAD")"
    [[ "$actual_commit" == "$expected_commit" ]] || {
      printf 'FAIL: %s has repository commit %s, expected %s\n' "$ip" "$actual_commit" "$expected_commit" >&2
      exit 1
    }
    dirty="$(run_on_host "$ip" "git -C '$RAY_REPO_PATH' status --porcelain --untracked-files=no")"
    [[ -z "$dirty" ]] || {
      printf 'FAIL: %s has tracked repository changes:\n%s\n' "$ip" "$dirty" >&2
      exit 1
    }
  done
  printf 'OK: all nodes use clean repository commit %s\n' "$expected_commit"
}

ray_source_hash() {
  sha256sum \
    "$RAY_REPO_PATH/training/ray/docker/Dockerfile.core" \
    "$RAY_REPO_PATH/training/ray/docker/entrypoint-head.sh" \
    "$RAY_REPO_PATH/training/ray/docker/entrypoint-worker.sh" \
    "$RAY_REPO_PATH/training/ray/requirements/core.txt" \
    | sha256sum | cut -c1-12
}

ray_local_image() {
  printf 'spectrum-ray:2.54.0-%s' "$(ray_source_hash)"
}
