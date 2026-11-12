"""Submit a Ray job once enough GPUs are free (cluster-side, so no operator is needed).

    python training/ray/scripts/start_when_gpus_free.py --min-free 2.5 --id NEW_ID --entrypoint "bash -c '...'"

Polls ray.available_resources() every --interval seconds; submits ``--entrypoint`` as ``--id`` after --consecutive polls in a row with at least
--min-free GPUs free (so a momentary gap does not trigger it). Does nothing if ``--id`` already exists.
"""
import argparse
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-free", type=float, required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--entrypoint", required=True)
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--consecutive", type=int, default=3)
    p.add_argument("--address", default="http://127.0.0.1:8265")
    a = p.parse_args()
    import ray
    from ray.job_submission import JobSubmissionClient
    ray.init(address="auto", logging_level="ERROR")
    client = JobSubmissionClient(a.address)
    ok = 0
    while True:
        if any(j.submission_id == a.id for j in client.list_jobs()):
            print(f"[start_when_free] {a.id} already exists, nothing to do", flush=True)
            return
        free = ray.available_resources().get("GPU", 0.0)
        ok = ok + 1 if free >= a.min_free else 0
        print(f"[start_when_free] {time.strftime('%F %T', time.gmtime())} free GPU {free:.2f}, {ok}/{a.consecutive}", flush=True)
        if ok >= a.consecutive:
            client.submit_job(submission_id=a.id, entrypoint=a.entrypoint)
            print(f"[start_when_free] submitted {a.id}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
