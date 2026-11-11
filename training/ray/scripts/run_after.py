"""Run commands after other Ray jobs have ended (so an idle cluster picks up queued follow-up work unattended).

    python training/ray/scripts/run_after.py --wait-ids ID1 ID2 --then "python -u -m ..." --then "python -u -m ..."

Polls the Ray Jobs API every 60 seconds until every waited job is SUCCEEDED, FAILED or STOPPED (a missing id counts as ended), then starts
every ``--then`` command at the same time and waits for them. Exit code 1 if any command failed.
"""
import argparse
import subprocess
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wait-ids", nargs="*", default=[])
    p.add_argument("--then", action="append", default=[])
    p.add_argument("--address", default="http://127.0.0.1:8265")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(args.address)
    while True:
        status = {j.submission_id: str(j.status).split(".")[-1] for j in client.list_jobs() if j.submission_id}
        live = [i for i in args.wait_ids if status.get(i) in ("PENDING", "RUNNING")]
        print(f"[run_after] waiting on {live}", flush=True)
        if not live:
            break
        time.sleep(args.interval)
    procs = [subprocess.Popen(c, shell=True) for c in args.then]
    codes = [proc.wait() for proc in procs]
    print(f"[run_after] exit codes {codes}", flush=True)
    raise SystemExit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
