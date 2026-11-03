# Independent cluster with a four-GPU host and two one-GPU workers.
RAY_HEAD_IP=192.168.1.175
RAY_NODE_IPS=(192.168.1.175 192.168.1.39 192.168.1.185)
RAY_NODE_NAMES=(ray-standalone ray-worker-1 ray-worker-2)
RAY_NODE_SSH_HOSTS=(129.114.25.96 129.114.26.81 129.114.27.86)
RAY_NODE_SERVICES=(standalone_ray-worker standalone_ray-single-gpu-worker standalone_ray-single-gpu-worker)
RAY_SSH_USER=cc
RAY_STACK_NAME=standalone
# Optional scoped host-key record for rebuilt instances at reused public IPs.
if [[ -z "${RAY_SSH_KNOWN_HOSTS:-}" && -f "$(dirname "${BASH_SOURCE[0]}")/secrets/standalone_known_hosts" ]]; then
  RAY_SSH_KNOWN_HOSTS="$(dirname "${BASH_SOURCE[0]}")/secrets/standalone_known_hosts"
fi
