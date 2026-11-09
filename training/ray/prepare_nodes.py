"""One-time per-node preparation, run before launching jobs on a fresh cluster.

Converts the 1-minute POWDER traces to the 15-minute traces on every worker (``ensure_powder15``), once per node, so tasks that start
together never race to write the same file. Run as a Ray job; one task per node is pinned by node id.

    python -m training.ray.prepare_nodes --powder15
"""

from __future__ import annotations

import argparse
import json


def prepare_node(spec: dict) -> dict:
    import os
    import socket
    import sys

    os.chdir(spec["repo_root"])
    sys.path.insert(0, spec["repo_root"])
    row = {"host": socket.gethostname()}
    try:
        if spec.get("powder15"):
            from training.ray.powder15 import BANDS, SPATIAL_NODES, ensure_powder15

            row["powder15"] = {band: len(ensure_powder15(band, spec["repo_root"], SPATIAL_NODES)) for band in BANDS}
        row["ok"] = True
    except Exception as error:
        row.update(ok=False, error=f"{type(error).__name__}: {error}"[:300])
    return row


def main() -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--powder15", action="store_true", help="convert the 15-minute traces of the core and held-out receivers")
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = parser.parse_args()
    ray.init(address=args.address, ignore_reinit_error=True)
    nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("CPU", 0) >= 1]
    refs = [ray.remote(num_cpus=1, num_gpus=0, max_retries=0,
                       scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))(prepare_node)
            .remote({"repo_root": args.repo_root, "powder15": args.powder15}) for n in nodes]
    for row in ray.get(refs):
        print("[prepare] " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
