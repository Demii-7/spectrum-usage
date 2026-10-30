"""Supervise the whole campaign: submit one `campaign` job per model and keep them going.

Run this itself as a Ray job on the head node (so it keeps running when your laptop
or SSH tunnel is gone). It submits per-model jobs through the local Jobs API, never
more than ``--max-concurrent-jobs`` at once, and resubmits any job that fails (the
per-model driver is idempotent and resumes, so resubmission is safe). It exits when
every model is done or has used up its resubmissions.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = "training/configs/oct2026"
OUTPUT_ROOT = "/workspace/spectrum-usage/runs/oct2026"

# Cheapest and most informative first, so partial completion is still useful.
# (model, representation, config file)
PLAN: tuple[tuple[str, str, str], ...] = (
    ("lookbackmean2d", "2d", "ray_tuning_linear_2d"),
    ("lookbackmean1d", "1d", "ray_tuning_linear_1d_all_bins"),
    ("lookbackmean4d", "4d", "ray_tuning_linear_4d"),
    ("linearar2d", "2d", "ray_tuning_linear_2d"),
    ("residuallinearar2d", "2d", "ray_tuning_linear_2d"),
    ("vanillalstm", "2d", "ray_tuning_2d"),
    ("residualvanillalstm", "2d", "ray_tuning_2d"),
    ("lstmattn", "2d", "ray_tuning_2d"),
    ("autoformer_csa", "2d", "ray_tuning_2d"),
    ("temporalconvnet", "2d", "ray_tuning_2d"),
    ("dswinlstm_i", "4d", "ray_tuning_4d"),
    ("linearar4d", "4d", "ray_tuning_linear_4d"),
    ("residuallinearar4d", "4d", "ray_tuning_linear_4d"),
    ("linearar1d", "1d", "ray_tuning_linear_1d_all_bins"),
    ("residuallinearar1d", "1d", "ray_tuning_linear_1d_all_bins"),
    ("vanillalstm1d", "1d", "ray_tuning_vanillalstm_1d"),
    ("residualvanillalstm", "1d", "ray_tuning_residualvanillalstm_1d"),
    ("convlstm", "4d", "ray_tuning_4d"),
    ("residualconvlstm", "4d", "ray_tuning_4d"),
)
TERMINAL_BAD = {"FAILED", "STOPPED"}


@dataclass
class Job:
    model: str
    representation: str
    config: str
    attempts: int = 0
    submission_id: str | None = None
    state: str = "queued"  # queued | running | done | gave_up
    last_failure: float = 0.0
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.model}-{self.representation}"


def build_entrypoint(job: Job, args: argparse.Namespace) -> str:
    out = f"{args.output_root.rstrip('/')}/{job.representation}/{job.model}"
    parts = ["python", "-m", "training.ray.campaign",
             "--config", f"{CONFIG_DIR}/{job.config}.yaml",
             "--model", job.model,
             "--prefix", f"{args.prefix.rstrip('/')}/{job.representation}",
             "--output-dir", out,
             "--candidates", str(args.candidates)]
    options = {"gpu_fraction": "--gpu-fraction", "max_concurrent_search": "--max-concurrent-search",
               "max_concurrent_replicate": "--max-concurrent-replicate",
               "grace_period": "--grace-period", "candidates": "--candidates"}
    for key, flag in options.items():
        if key in job.overrides:
            if flag in parts:
                parts[parts.index(flag) + 1] = str(job.overrides[key])
            else:
                parts += [flag, str(job.overrides[key])]
    if args.skip_export:
        parts.append("--skip-export")
    return f"cd /workspace/spectrum-usage && {shlex.join(parts)}"


def make_jobs(models: list[str] | None, overrides: dict[str, dict[str, Any]]) -> list[Job]:
    jobs = []
    for model, representation, config in PLAN:
        if models and f"{model}" not in models and f"{model}:{representation}" not in models:
            continue
        job = Job(model, representation, config)
        job.overrides = dict(overrides.get(job.key, overrides.get(model, {})))
        jobs.append(job)
    return jobs


def _status_name(status: Any) -> str:
    return str(getattr(status, "value", status)).upper()


def reconcile(jobs: list[Job], client) -> None:
    """Adopt jobs a previous supervisor already submitted (this one may be a restart)."""
    existing: dict[str, tuple[int, str]] = {}
    for detail in client.list_jobs():
        submission = detail.submission_id or ""
        if not submission.startswith("campaign-"):
            continue
        base, _, attempt = submission.rpartition("-")
        if attempt.isdigit():
            known = existing.get(base)
            if known is None or int(attempt) > known[0]:
                existing[base] = (int(attempt), _status_name(detail.status))
    for job in jobs:
        found = existing.get(f"campaign-{job.key}")
        if not found:
            continue
        attempt, status = found
        job.attempts = attempt + 1
        job.submission_id = f"campaign-{job.key}-{attempt}"
        if status == "SUCCEEDED":
            job.state = "done"
        elif status in {"PENDING", "RUNNING"}:
            job.state = "running"


def step(jobs: list[Job], client, args: argparse.Namespace, now: float) -> None:
    for job in jobs:
        if job.state != "running":
            continue
        status = _status_name(client.get_job_status(job.submission_id))
        if status == "SUCCEEDED":
            job.state = "done"
            print(f"[supervisor] {job.key}: done", flush=True)
        elif status in TERMINAL_BAD:
            job.last_failure = now
            if job.attempts >= args.max_attempts:
                job.state = "gave_up"
                print(f"[supervisor] {job.key}: {status}, no attempts left", flush=True)
            else:
                job.state = "queued"
                print(f"[supervisor] {job.key}: {status}, will resubmit", flush=True)
    running = sum(1 for job in jobs if job.state == "running")
    for job in jobs:
        if running >= args.max_concurrent_jobs:
            break
        if job.state != "queued" or now - job.last_failure < args.retry_delay:
            continue
        job.submission_id = f"campaign-{job.key}-{job.attempts}"
        client.submit_job(entrypoint=build_entrypoint(job, args), submission_id=job.submission_id,
                          metadata={"model": job.model, "representation": job.representation})
        job.attempts += 1
        job.state = "running"
        running += 1
        print(f"[supervisor] submitted {job.submission_id}", flush=True)


def write_status(jobs: list[Job], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [{"model": j.model, "representation": j.representation, "state": j.state,
          "attempts": j.attempts, "submission_id": j.submission_id} for j in jobs],
        indent=2) + "\n", encoding="utf-8")


def supervise(args: argparse.Namespace, client) -> int:
    overrides = json.loads(Path(args.overrides).read_text()) if args.overrides else {}
    jobs = make_jobs(args.models, overrides)
    reconcile(jobs, client)
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
    status_path = Path(args.output_root) / "supervisor_status.json"
    while True:
        now = time.time()
        step(jobs, client, args, now)
        write_status(jobs, status_path)
        if all(job.state in {"done", "gave_up"} for job in jobs):
            break
        if deadline and now > deadline:
            print("[supervisor] deadline reached; leaving running jobs in place", flush=True)
            break
        time.sleep(args.poll_seconds)
    failed = [job.key for job in jobs if job.state == "gave_up"]
    print(f"[supervisor] finished; done={sum(j.state == 'done' for j in jobs)} gave_up={failed}", flush=True)
    return 1 if failed else 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="spectrum-usage/ray/oct2026-core4")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--address", default="http://127.0.0.1:8265")
    parser.add_argument("--models", nargs="+", help="Subset (model or model:representation)")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--max-concurrent-jobs", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--deadline-hours", type=float, default=None)
    parser.add_argument("--overrides", help="JSON file: {model or model-rep: {gpu_fraction: ..}}")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the entrypoints and exit")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        overrides = json.loads(Path(args.overrides).read_text()) if args.overrides else {}
        for job in make_jobs(args.models, overrides):
            print(f"{job.key}\n  {build_entrypoint(job, args)}")
        return
    from ray.job_submission import JobSubmissionClient

    sys.exit(supervise(args, JobSubmissionClient(args.address)))


if __name__ == "__main__":
    main()
