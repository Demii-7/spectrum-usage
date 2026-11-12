"""Replace running Ray drivers by the same command with a higher ``--max-parallel``, unattended (run it as a Ray job on the head).

    ray job submit --submission-id claude-<cluster>-relaunch-1 -- python -u -m training.ray.relaunch_wider --spec-file training/ray/relaunch/<cluster>.json

Spec file: list of ``{"base": id, "new_id": id, "entrypoint": cmd, "trigger": "now" | "after_completion" | "gpus_free:N"}``. For each spec (one thread each) the
newest submission of the chain ``base``, ``base-r1`` ... is found; with ``after_completion`` the job's log is polled until one more finished
unit (a line containing " ok=") has appeared, so the restart lands right after a task finished (that slot loses nothing). The running driver is then
stopped and ``new_id`` is submitted. Entrypoints must be resumable (``--skip-done``). If the old job has already ended nothing is stopped and
``new_id`` is submitted only when the old job did not SUCCEED. The watchdog (job_watchdog.py) should list ``new_id`` so a crash is resubmitted.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

END = ("SUCCEEDED", "FAILED", "STOPPED")


def status_of(client, jid):
    for j in client.list_jobs():
        if j.submission_id == jid:
            return str(j.status.value if hasattr(j.status, "value") else j.status).upper()
    return None


def newest(client, base):
    ids = {j.submission_id for j in client.list_jobs() if j.submission_id}
    cur = base if base in ids else None
    n = 1
    while f"{base}-r{n}" in ids:
        cur = f"{base}-r{n}"
        n += 1
    return cur


def handle(client, spec, interval):
    tag = spec["new_id"]
    cur = newest(client, spec["base"])
    if cur is None:
        print(f"[relaunch] {tag}: no job in chain {spec['base']}, submitting directly", flush=True)
    else:
        if str(spec.get("trigger", "")).startswith("gpus_free:"):
            import ray
            need, ok = float(spec["trigger"].split(":")[1]), 0
            while status_of(client, cur) in ("PENDING", "RUNNING"):
                free = ray.available_resources().get("GPU", 0.0)
                ok = ok + 1 if free >= need else 0
                print(f"[relaunch] {tag}: free GPU {free:.2f} (need {need}), {ok}/3", flush=True)
                if ok >= 3:
                    break
                time.sleep(interval)
        if spec.get("trigger") == "after_completion":
            seen = client.get_job_logs(cur).count(" ok=")
            print(f"[relaunch] {tag}: waiting for a task to finish on {cur} ({seen} done)", flush=True)
            while status_of(client, cur) in ("PENDING", "RUNNING"):
                cur = newest(client, spec["base"]) or cur
                if client.get_job_logs(cur).count(" ok=") > seen:
                    break
                time.sleep(interval)
        st = status_of(client, cur)
        if st in ("PENDING", "RUNNING"):
            client.stop_job(cur)
            for _ in range(60):
                if status_of(client, cur) in END:
                    break
                time.sleep(5)
            print(f"[relaunch] {tag}: stopped {cur}", flush=True)
        elif st == "SUCCEEDED":
            print(f"[relaunch] {tag}: {cur} already SUCCEEDED, nothing to do", flush=True)
            return
    client.submit_job(submission_id=spec["new_id"], entrypoint=spec["entrypoint"])
    print(f"[relaunch] {time.strftime('%F %T', time.gmtime())} submitted {spec['new_id']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec-file", required=True)
    p.add_argument("--address", default="http://127.0.0.1:8265")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args()
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(a.address)
    import ray
    ray.init(address="auto", logging_level="ERROR")
    specs = json.loads((Path(a.repo_root) / a.spec_file).read_text())
    ts = [threading.Thread(target=handle, args=(client, s, a.interval)) for s in specs]
    [t.start() for t in ts]
    [t.join() for t in ts]


if __name__ == "__main__":
    main()
