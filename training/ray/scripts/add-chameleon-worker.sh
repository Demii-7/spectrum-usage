#!/usr/bin/env bash
# Run from your workstation with an SSH agent containing the Chameleon key.
set -euo pipefail
if [[ $# != 2 ]]; then
  printf 'Usage: bash %s PUBLIC_IP PRIVATE_IP\n' "$0" >&2
  exit 2
fi
PUBLIC_IP="$1"
PRIVATE_IP="$2"
[[ "$PUBLIC_IP" =~ ^[0-9.]+$ && "$PRIVATE_IP" =~ ^[0-9.]+$ ]] || exit 2
HEAD="${RAY_HEAD_SSH:-cc@129.114.27.86}"
IMAGE="${RAY_IMAGE:-spectrum-ray:2.54.0-chameleon}"
[[ "$IMAGE" =~ ^[a-zA-Z0-9:./_-]+$ ]] || exit 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ssh "cc@$PUBLIC_IP" 'nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader; df -h /'
ssh "cc@$PUBLIC_IP" 'bash -s' < "$SCRIPT_DIR/install-host.sh"
# Copy the running source snapshot and exact image over the private network.
ssh -A "$HEAD" "rsync -az -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' /home/cc/spectrum-usage/ cc@$PRIVATE_IP:/home/cc/spectrum-usage/"
ssh -A "$HEAD" "set -o pipefail; docker image save '$IMAGE' | ssh -o BatchMode=yes cc@$PRIVATE_IP docker image load"
ssh "$HEAD" 'docker swarm join-token -q worker' | ssh "cc@$PUBLIC_IP" \
  "read -r token; if [ \"\$(docker info --format '{{.Swarm.LocalNodeState}}')\" = inactive ]; then docker swarm join --token \"\$token\" --advertise-addr '$PRIVATE_IP' --data-path-addr '$PRIVATE_IP' 192.168.1.43:2377 >/dev/null; fi"
NAME="$(ssh "cc@$PUBLIC_IP" 'hostname -s')"
[[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
ssh "$HEAD" "docker node update --label-add ray.gpu=true '$NAME'; docker node ls"
printf 'Worker joined. Run cluster_probe with --check-spectrum-read to verify it.\n'
