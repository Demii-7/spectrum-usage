# Independent restricted allocation on h4-lm: GPU 0 belongs to the collaborator.
RAY_HEAD_IP=10.52.2.210
RAY_NODE_IPS=(10.52.2.210)
RAY_NODE_NAMES=(h4-lm)
RAY_NODE_SSH_HOSTS=(ray-h100-three-1)
RAY_NODE_SERVICES=(h100three_ray-worker)
RAY_SSH_USER=cc
RAY_STACK_NAME=h100three
RAY_DATA_SITES=(aerpaw ara cosmos powder)
RAY_CHECKPOINT_SKIP_HOSTS=(ray-h100-three-1)
RAY_SSH_CONFIG="${RAY_SSH_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/h100-three-ssh.conf}"
