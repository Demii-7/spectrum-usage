# Six-host H100 cluster; head lease ends 2026-11-14 01:49 UTC.
RAY_HEAD_IP=192.168.1.226
RAY_NODE_IPS=(192.168.1.226 192.168.1.122 192.168.1.47 192.168.1.15 192.168.1.45 192.168.1.9)
RAY_NODE_NAMES=(ray-final-1 ray-final-2 ray-final-3 ray-final-4 ray-final-5 ray-final-6)
RAY_NODE_SSH_HOSTS=(ray-final-1 ray-final-2 ray-final-3 ray-final-4 ray-final-5 ray-final-6)
RAY_NODE_SERVICES=(final_ray-worker final_ray-worker final_ray-worker final_ray-worker final_ray-worker-extra final_ray-worker-extra)
RAY_SSH_USER=cc
RAY_STACK_NAME=final
RAY_DATA_SITES=(aerpaw ara cosmos powder)
RAY_CHECKPOINT_PREFIXES=(checkpoints/1d/ checkpoints/2d/ checkpoints/4d/ checkpoints/MANIFEST.tsv)
# Expansion workers keep code and datasets only.
RAY_CHECKPOINT_SKIP_HOSTS=(ray-final-5 ray-final-6)
RAY_SSH_CONFIG="${RAY_SSH_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/final-ssh.conf}"
