#!/bin/sh
# Host firewall for Ray cluster heads with a public floating IP.
#
# Floating IPs are 1:1 NAT onto the private address, so the source address is
# the only way to tell internet traffic from cluster traffic. Allowed inbound:
#   - SSH (tcp/22) from anywhere
#   - ICMP
#   - anything from private ranges (cluster nodes, Swarm, Docker bridges)
#   - return traffic for connections this host or its containers opened
# Everything else new from the internet is dropped, both to host services
# (INPUT) and to Docker-published ports such as the Ray dashboard 8265, Ray
# client 10001 and MinIO 9000/9001 (DOCKER-USER, which Docker evaluates before
# its own accept rules). Reach those services over SSH tunnels instead.
#
# Install:  sudo install -m 755 host-firewall.sh /usr/local/sbin/spectrum-firewall
#           sudo install -m 644 spectrum-firewall.service /etc/systemd/system/
#           sudo systemctl daemon-reload && sudo systemctl enable --now spectrum-firewall
# Remove:   sudo /usr/local/sbin/spectrum-firewall --remove
set -eu
IPT=iptables
PRIVATE="10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 169.254.0.0/16 127.0.0.0/8"
EXT=$(ip route show default | awk '{for (i=1;i<NF;i++) if ($i=="dev") {print $(i+1); exit}}')

remove() {
    while $IPT -D INPUT -j SPECTRUM-IN 2>/dev/null; do :; done
    while $IPT -D DOCKER-USER -j SPECTRUM-FWD 2>/dev/null; do :; done
    for c in SPECTRUM-IN SPECTRUM-FWD; do
        $IPT -F "$c" 2>/dev/null || true
        $IPT -X "$c" 2>/dev/null || true
    done
}

if [ "${1:-}" = "--remove" ]; then
    remove
    echo "spectrum firewall removed"
    exit 0
fi

[ -n "$EXT" ] || { echo "cannot find default-route interface" >&2; exit 1; }

# Host services.
$IPT -N SPECTRUM-IN 2>/dev/null || $IPT -F SPECTRUM-IN
$IPT -A SPECTRUM-IN -i lo -j RETURN
$IPT -A SPECTRUM-IN -m conntrack --ctstate RELATED,ESTABLISHED -j RETURN
$IPT -A SPECTRUM-IN -p tcp --dport 22 -j RETURN
$IPT -A SPECTRUM-IN -p icmp -j RETURN
for n in $PRIVATE; do $IPT -A SPECTRUM-IN -s "$n" -j RETURN; done
$IPT -A SPECTRUM-IN -i "$EXT" -j DROP
$IPT -C INPUT -j SPECTRUM-IN 2>/dev/null || $IPT -A INPUT -j SPECTRUM-IN

# Docker-published ports (DNAT'd traffic goes through FORWARD, not INPUT).
$IPT -N DOCKER-USER 2>/dev/null || true
$IPT -N SPECTRUM-FWD 2>/dev/null || $IPT -F SPECTRUM-FWD
$IPT -A SPECTRUM-FWD -m conntrack --ctstate RELATED,ESTABLISHED -j RETURN
for n in $PRIVATE; do $IPT -A SPECTRUM-FWD -s "$n" -j RETURN; done
$IPT -A SPECTRUM-FWD -i "$EXT" -j DROP
$IPT -C DOCKER-USER -j SPECTRUM-FWD 2>/dev/null || $IPT -I DOCKER-USER 1 -j SPECTRUM-FWD

echo "spectrum firewall applied on $EXT"
