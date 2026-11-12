#!/usr/bin/env bash
# Run on a cluster host with the private IPs of its trusted cluster peers.
set -euo pipefail
if (($# == 0)); then
  printf 'Usage: bash %s PRIVATE_IP [PRIVATE_IP ...]\n' "$0" >&2
  exit 2
fi
if ! command -v firewall-cmd >/dev/null || ! sudo firewall-cmd --state >/dev/null 2>&1; then
  printf 'firewalld is inactive; no firewalld rules to configure.\n'
  exit 0
fi
for ip in "$@"; do
  [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || exit 2
  sudo firewall-cmd --zone=trusted --add-source="$ip/32"
  sudo firewall-cmd --permanent --zone=trusted --add-source="$ip/32"
done
sudo firewall-cmd --zone=trusted --list-sources
