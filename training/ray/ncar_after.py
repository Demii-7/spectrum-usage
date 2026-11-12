"""Run a command once no job whose submission id starts with ``--wait-prefix`` is pending or running (checked every ``--every`` s, after an
initial ``--grace`` so a just-restarted job is seen). Used to queue work behind the NCAR large-model stage.

    python -m training.ray.ncar_after --wait-prefix ncar-tx-cells-gpufull -- python -m training.ray.ncar_tx_cells ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def active(prefix: str, address: str) -> list[str]:
    from ray.job_submission import JobStatus, JobSubmissionClient

    client = JobSubmissionClient(address)
    return [j.submission_id for j in client.list_jobs()
            if (j.submission_id or "").startswith(prefix) and j.status in (JobStatus.PENDING, JobStatus.RUNNING)]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-prefix", required=True)
    p.add_argument("--address", default="http://127.0.0.1:8265")
    p.add_argument("--every", type=int, default=300)
    p.add_argument("--grace", type=int, default=600, help="seconds with no matching job before starting (survives a restart gap)")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    idle_since = None
    while True:
        jobs = active(a.wait_prefix, a.address)
        now = time.time()
        if jobs:
            idle_since = None
        elif idle_since is None:
            idle_since = now
        elif now - idle_since >= a.grace:
            break
        print(f"[after] {time.strftime('%H:%M:%SZ', time.gmtime())} waiting on {jobs or 'nothing (grace period)'}", flush=True)
        time.sleep(a.every if jobs else min(a.every, 60))
    print(f"[after] starting: {' '.join(command)}", flush=True)
    sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
