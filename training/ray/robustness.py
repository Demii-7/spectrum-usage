"""Failure handling, artifact sync, and resume helpers shared by the Ray launchers."""

from __future__ import annotations

from typing import Any, Callable

DEFAULT_MAX_FAILURES = 2


def run_config(name: str, storage_path: str, *, max_failures: int = DEFAULT_MAX_FAILURES,
               sync_artifacts: bool = True):
    """Build a Tune RunConfig that retries failed trials and uploads trial artifacts.

    ``FailureConfig``/``SyncConfig`` are looked up defensively: if the installed Ray
    lacks one, the launcher still runs, with a loud warning instead of silently losing
    the protection.
    """
    from ray import tune
    from ray.tune import RunConfig

    kwargs: dict[str, Any] = {"name": name, "storage_path": storage_path}
    failure = getattr(tune, "FailureConfig", None)
    if failure is not None and max_failures:
        kwargs["failure_config"] = failure(max_failures=max_failures)
    elif max_failures:
        print("[WARN] ray.tune.FailureConfig unavailable; failed trials will not be retried")
    sync = getattr(tune, "SyncConfig", None)
    if sync is not None and sync_artifacts:
        kwargs["sync_config"] = sync(sync_artifacts=True)
    try:
        return RunConfig(**kwargs)
    except TypeError as error:
        print(f"[WARN] RunConfig rejected optional settings ({error}); "
              "falling back to name and storage_path only")
        return RunConfig(name=name, storage_path=storage_path)


def build_or_restore(
    experiment_path: str,
    trainable: Any,
    make_tuner: Callable[[], Any],
    *,
    resume: bool,
):
    """Restore an interrupted experiment (rerunning errored trials), else build a new tuner."""
    from ray import tune

    if resume and tune.Tuner.can_restore(experiment_path):
        print(f"[INFO] restoring experiment {experiment_path}")
        return tune.Tuner.restore(
            experiment_path,
            trainable=trainable,
            resume_unfinished=True,
            restart_errored=True,
        )
    return make_tuner()


def summarize_results(results: Any, *, label: str) -> dict[str, int]:
    """Print and return trial outcome counts; raise if nothing finished successfully."""
    counts = {"trials": 0, "errored": 0, "completed": 0}
    for result in results:
        counts["trials"] += 1
        if result.error is not None:
            counts["errored"] += 1
        elif (result.metrics or {}).get("completed"):
            counts["completed"] += 1
    print(f"[INFO] {label}: {counts}")
    if counts["trials"] and not counts["completed"]:
        raise RuntimeError(f"{label}: no trial completed successfully ({counts})")
    return counts
