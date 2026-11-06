"""Stop a ``tune_fixed`` driver when its search trials are all recorded and resubmit it with the current code.

A ``python -m training.ray.tune_fixed`` driver ships its worker functions by value, so a code patch (for example the one that makes
final runs save weights) does not reach a driver that is already running. The driver is resumable: finished trials and finals are
skipped. This watcher waits until every group has the expected number of recorded trials (successful or not, because the driver
proceeds past failed ones), stops the old job within seconds of that, and submits the same entrypoint under a new id, so only the
finals that just started are lost and no trial is.

    python -m training.ray.restart_when_trials_done --job-id claude-tune-p15-1d-lstm --new-job-id claude-tune-p15-1d-lstm-2 \
        --entrypoint "cd /workspace/spectrum-usage && python -u -m training.ray.tune_fixed ..." --groups tuning/powder15/600_800/1d/vanillalstm1d ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def trials_complete(counts: dict[str, int], expected: int) -> bool:
    """True when every group has at least ``expected`` recorded trials."""
    return bool(counts) and all(n >= expected for n in counts.values())


def count_trials(client, bucket: str, prefix: str) -> int:
    """Trial records directly under ``prefix`` (not the ``/final`` subfolder)."""
    total = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix.rstrip("/")) + 1:]
            if "/" not in name and name.startswith("trial-") and name.endswith(".json"):
                total += 1
    return total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-id", required=True)
    p.add_argument("--new-job-id", required=True)
    p.add_argument("--entrypoint", required=True)
    p.add_argument("--groups", nargs="+", required=True, help="bucket prefixes of the tuning groups to watch")
    p.add_argument("--expected", type=int, default=16)
    p.add_argument("--poll", type=int, default=30)
    p.add_argument("--max-hours", type=float, default=72.0)
    p.add_argument("--bucket", default=None)
    p.add_argument("--dashboard", default="http://127.0.0.1:8265")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from ray.job_submission import JobStatus, JobSubmissionClient

    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    jobs = JobSubmissionClient(args.dashboard)
    deadline = time.time() + args.max_hours * 3600
    while time.time() < deadline:
        status = jobs.get_job_status(args.job_id)
        if status != JobStatus.RUNNING:
            print(f"[restart] {args.job_id} is {status.value}, not RUNNING; nothing to do", flush=True)
            return 0
        counts = {g: count_trials(client, bucket, g) for g in args.groups}
        print(f"[restart] {time.strftime('%H:%M:%S')} trials {sorted(counts.values())}", flush=True)
        if trials_complete(counts, args.expected):
            print(f"[restart] all trials recorded; stopping {args.job_id}", flush=True)
            jobs.stop_job(args.job_id)
            while jobs.get_job_status(args.job_id) == JobStatus.RUNNING:
                time.sleep(2)
            jobs.submit_job(entrypoint=args.entrypoint, submission_id=args.new_job_id)
            print(f"[restart] submitted {args.new_job_id}", flush=True)
            return 0
        time.sleep(args.poll)
    print("[restart] gave up at the deadline", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
