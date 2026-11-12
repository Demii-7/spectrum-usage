#!/usr/bin/env python3
"""Apply this stack's CPU affinity to Swarm containers, including replacements.

Swarm supports CPU quotas but has no cpuset field in its service spec. This
host guard adds the cpuset without changing the Docker daemon's defaults or
touching other users' containers.
"""
import json
import subprocess
import time

CPUSET = "0-71,96-167"
STACK = "h100three"


def enforce():
    listed = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label=com.docker.stack.namespace={STACK}"],
        capture_output=True, text=True, check=True,
    )
    for container in listed.stdout.split():
        inspected = subprocess.run(["docker", "inspect", container], capture_output=True, text=True)
        if inspected.returncode:
            continue  # Swarm may remove a completed/replaced task during the scan.
        if json.loads(inspected.stdout)[0]["HostConfig"]["CpusetCpus"] != CPUSET:
            subprocess.run(["docker", "update", "--cpuset-cpus", CPUSET, container], check=True)
            print(f"Pinned {container} to {CPUSET}", flush=True)


if __name__ == "__main__":
    while True:
        enforce()
        time.sleep(1)
