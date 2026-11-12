#!/bin/sh
# Restrict only this stack's published Ray/MinIO ports; other users' ports stay
# under their existing policy. Floating-IP traffic arrives on the default NIC.
set -eu
EXT=$(ip -4 route show default | awk '{for (i=1;i<NF;i++) if ($i=="dev") {print $(i+1); exit}}')
test -n "$EXT"
CHAIN=H100THREE-PORTS
iptables -N "$CHAIN" 2>/dev/null || iptables -F "$CHAIN"
iptables -A "$CHAIN" -m conntrack --ctstate RELATED,ESTABLISHED -j RETURN
for network in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 127.0.0.0/8; do
  iptables -A "$CHAIN" -s "$network" -j RETURN
done
iptables -A "$CHAIN" -i "$EXT" -p tcp -m multiport --dports 8265,10001,9000,9001 -j DROP
iptables -A "$CHAIN" -j RETURN
iptables -C INPUT -j "$CHAIN" 2>/dev/null || iptables -I INPUT 1 -j "$CHAIN"
iptables -N DOCKER-USER 2>/dev/null || true
iptables -C DOCKER-USER -j "$CHAIN" 2>/dev/null || iptables -I DOCKER-USER 1 -j "$CHAIN"
