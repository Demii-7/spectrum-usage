# Eight RTX hosts plus a four-GPU V100 expansion worker.
RAY_HEAD_IP=10.140.81.200
RAY_NODE_IPS=(10.140.81.200 10.140.82.107 10.140.82.132 10.140.82.89 10.140.82.191 10.140.83.1 10.140.81.125 10.140.83.84 10.140.82.90)
RAY_NODE_NAMES=(ray-uc-1 ray-uc-2 ray-uc-3 ray-uc-4 ray-uc-5 ray-uc-6 ray-uc-7 ray-uc-8 ray-v100)
RAY_NODE_SSH_HOSTS=(ray-uc-1 ray-uc-2 ray-uc-3 ray-uc-4 ray-uc-5 ray-uc-6 ray-uc-7 ray-uc-8 ray-uc-extra-1)
RAY_NODE_SERVICES=(uc_ray-worker uc_ray-worker uc_ray-worker uc_ray-worker uc_ray-worker uc_ray-worker uc_ray-worker uc_ray-worker uc_ray-worker-extra)
RAY_SSH_USER=cc
RAY_STACK_NAME=uc
# Keep CSV inventory scans out of experiment/checkpoint artifact namespaces.
RAY_DATA_SITES=(aerpaw ara cosmos powder)
# Expansion host has code and datasets only; do not copy the checkpoint archive.
RAY_CHECKPOINT_SKIP_HOSTS=(ray-uc-extra-1)
RAY_SSH_CONFIG="${RAY_SSH_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/uc-ssh.conf}"
