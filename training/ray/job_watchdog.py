"""Resubmit Ray jobs whose driver crashed, so unattended campaigns survive an out-of-memory kill or a lost node.

Run it as a Ray job on the cluster's head, so it keeps going when nobody is connected:

    ray job submit --submission-id claude-watchdog-<cluster>-1 -- python -u -m training.ray.job_watchdog --jobs-file training/ray/watchdog/<cluster>.json

The jobs file is a list of ``{"id": ..., "entrypoint": ..., "max_restarts": N}``. For each tracked id the newest submission in the chain
``id``, ``id-r1``, ``id-r2`` ... is inspected; if it FAILED (driver crash, node loss, memory kill) and fewer than ``max_restarts`` restarts
were used, the same entrypoint is submitted as the next ``-rN``. A job that SUCCEEDED or was STOPPED on purpose is never resubmitted, and
a chain that does not exist yet is left alone (start jobs yourself). Every drivers' work must be resumable (finished work skipped), which
is true of all tracked entrypoints (``--skip-done`` or per-record skipping in the bucket).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def chain_ids(base: str, existing: set[str]) -> list[str]:
    """The submission ids of one job's chain that exist, oldest first."""
    ids = [base] if base in existing else []
    n = 1
    while f"{base}-r{n}" in existing:
        ids.append(f"{base}-r{n}")
        n += 1
    return ids


def decide(spec: dict, statuses: dict[str, str]) -> str | None:
    """The id to submit next for ``spec`` given {submission_id: status}, or None."""
    ids = chain_ids(spec["id"], set(statuses))
    if not ids:
        return None
    restarts = len(ids) - 1
    if statuses[ids[-1]] == "FAILED" and restarts < int(spec.get("max_restarts", 6)):
        return f"{spec['id']}-r{restarts + 1}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs-file", required=True)
    parser.add_argument("--address", default="http://127.0.0.1:8265")
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = parser.parse_args()
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(args.address)
    while True:
        specs = json.loads((Path(args.repo_root) / args.jobs_file).read_text())
        statuses = {job.submission_id: str(job.status.value if hasattr(job.status, "value") else job.status).upper()
                    for job in client.list_jobs()}
        for spec in specs:
            next_id = decide(spec, statuses)
            if next_id:
                client.submit_job(submission_id=next_id, entrypoint=spec["entrypoint"])
                print(f"[watchdog] {time.strftime('%F %T', time.gmtime())} resubmitted {spec['id']} as {next_id}", flush=True)
        print(f"[watchdog] {time.strftime('%F %T', time.gmtime())} checked {len(specs)} jobs", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
