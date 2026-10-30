#!/usr/bin/env bash
# Run on each Ubuntu GPU host before joining the cluster.
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y docker.io curl ca-certificates gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update -qq
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo usermod -aG docker cc
sudo systemctl enable docker
sudo systemctl restart docker
sudo docker info --format 'runtime={{.DefaultRuntime}} swarm={{.Swarm.LocalNodeState}}'
