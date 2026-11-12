# Independent TACC cluster. Add a fifth node here and to tacc-ssh.conf.
RAY_HEAD_IP=10.52.2.186
RAY_NODE_IPS=(10.52.2.186 10.52.0.30 10.52.0.165 10.52.2.197)
RAY_NODE_NAMES=(ray-tacc-1 ray-tacc-2 ray-tacc-3 ray-tacc-4)
RAY_NODE_SSH_HOSTS=(ray-tacc-1 ray-tacc-2 ray-tacc-3 ray-tacc-4)
RAY_NODE_SERVICES=(tacc_ray-worker tacc_ray-worker tacc_ray-worker tacc_ray-worker)
RAY_SSH_USER=cc
RAY_STACK_NAME=tacc
RAY_SSH_CONFIG="${RAY_SSH_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/tacc-ssh.conf}"
