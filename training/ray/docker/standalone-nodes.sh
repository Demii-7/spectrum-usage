# Rebuilt H100 cluster: six one-GPU hosts, including the head's GPU worker.
RAY_HEAD_IP=192.168.1.185
RAY_NODE_IPS=(192.168.1.185 192.168.1.39 192.168.1.57 192.168.1.219 192.168.1.104 192.168.1.82)
RAY_NODE_NAMES=(ray-worker-2 ray-worker-1 ray-new-1 ray-new-2 ray-new-3 ray-new-4)
RAY_NODE_SSH_HOSTS=(ray-worker-2 ray-worker-1 ray-new-1 ray-new-2 ray-new-3 ray-new-4)
RAY_NODE_SERVICES=(standalone_ray-worker standalone_ray-worker standalone_ray-worker standalone_ray-worker standalone_ray-worker standalone_ray-worker)
RAY_SSH_USER=cc
RAY_STACK_NAME=standalone
RAY_SSH_CONFIG="${RAY_SSH_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/h100-ssh.conf}"
RAY_NEW_NODE_LEASE_END=2026-11-09T13:40:00Z
RAY_OLD_NODE_LEASE_END=2026-11-10T18:00:00Z
