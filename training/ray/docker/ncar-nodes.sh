# Independent aarch64 GH200 cluster. Source and data only, no checkpoint archive.
RAY_HEAD_IP=172.20.0.252
RAY_NODE_IPS=(172.20.0.252 172.20.0.132)
RAY_NODE_NAMES=(ray-ncar-1 ray-ncar-2)
RAY_NODE_SSH_HOSTS=(ray-ncar-1 ray-ncar-2)
RAY_NODE_SERVICES=(ncar_ray-worker ncar_ray-worker-extra)
RAY_SSH_USER=cc
RAY_STACK_NAME=ncar
RAY_DATA_SITES=(aerpaw ara cosmos powder)
RAY_CHECKPOINT_SKIP_HOSTS=(ray-ncar-1 ray-ncar-2)
RAY_SSH_CONFIG="${RAY_SSH_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/ncar-ssh.conf}"
