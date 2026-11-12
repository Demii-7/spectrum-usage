# Mixed ARM GH200 and x86 CPU cluster. No bulk checkpoint archive.
RAY_HEAD_IP=172.20.0.252
RAY_NODE_IPS=(172.20.0.252 172.20.0.132 172.20.0.136 172.20.0.196)
RAY_NODE_NAMES=(ray-ncar-1 ray-ncar-2 ray-ncar-cpu-3 ray-ncar-cpu-4)
RAY_NODE_SSH_HOSTS=(ray-ncar-1 ray-ncar-2 ray-ncar-cpu-3 ray-ncar-cpu-4)
RAY_NODE_SERVICES=(ncar_ray-worker ncar_ray-worker-extra ncar_ray-worker-cpu ncar_ray-worker-cpu)
RAY_SSH_USER=cc
RAY_STACK_NAME=ncar
RAY_DATA_SITES=(aerpaw ara cosmos powder)
RAY_CHECKPOINT_SKIP_HOSTS=(ray-ncar-1 ray-ncar-2 ray-ncar-cpu-3 ray-ncar-cpu-4)
RAY_SSH_CONFIG="${RAY_SSH_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/ncar-ssh.conf}"
