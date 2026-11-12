"""Cluster-side poller (Ray job, unattended): when fork AE's DSwinLSTM-I 15-minute search has a winner that beats the inherited configuration, train
the map-construction variants ``core4-p4`` and ``core4-krige`` (seed 42, both bands) with the winner's configuration and score them through the same pipeline.

Rule (user decision, 2026-11-12): run a band only if its verdict in ``s3://spectrum/analysis-outputs/dswin_search_verdict.json`` is ``improves`` (the winner beats the
inherited configuration's validation objective by at least 3%, AE's own trigger) AND the winner is not the inherited anchor (candidate index 2). A winner that differs
by less than 3% is NOT run; the decision and the reason are written to ``analysis-outputs/dswin_p4krige_tuned_status.json`` so the user can decide.
Polls every 10 minutes for up to 24 h. The inherited-configuration runs (``spatial15-tuned-oct2026``) are untouched; the new runs go to the NEW prefix
``spatial15-tuned-dswin-oct2026`` with ``hyperparameters`` = ``tuned15:<candidate id> (fixed configuration ...)`` in every result row, per-origin errors to
``spatial15-origin-errors-dswin-tuned-oct2026``. Runs on UC (GPU share 0.15 each, 4 runs concurrently: one job per band); CPU-only steps hold no GPU.

    python -u training/ray/scripts/dswin_tuned_p4krige_poller.py [--interval 600] [--giveup-hours 24]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

INHERITED_INDEX = 2            # candidate 2 of the search IS the inherited configuration (tune_dswin15)
THRESHOLD = 0.03
BANDS = ("600_800", "2400_2600")
FIXED_KEY = "analysis-outputs/dswin_tuned_fixed_{band}.json"
STATUS_KEY = "analysis-outputs/dswin_p4krige_tuned_status.json"
NEW_PREFIX = "spatial15-tuned-dswin-oct2026"
ORIGIN_PREFIX = "spatial15-origin-errors-dswin-tuned-oct2026"


def decide(entry: dict | None) -> tuple[str, str]:
    """('wait' | 'run' | 'skip', reason) for one band's verdict entry."""
    if not entry:
        return "wait", "no verdict for this band yet"
    verdict = entry.get("verdict")
    if verdict == "no completed trials":
        return "wait", "search has no completed trials yet"
    gain = entry.get("improvement")
    if gain is None:
        return "wait", "verdict without improvement value"
    if entry.get("winner_index") == INHERITED_INDEX:
        return "skip", f"the winner is the inherited configuration (candidate {INHERITED_INDEX}); improvement {gain:+.3f}"
    if gain >= THRESHOLD and verdict == "improves":
        return "run", f"winner {entry.get('winner')} improves the inherited validation objective by {gain:.3f} (>= {THRESHOLD})"
    return "skip", (f"winner {entry.get('winner')} differs from the inherited configuration but improves the validation objective by only "
                    f"{gain:.3f} (< {THRESHOLD}): NOT run, for the user to decide")


def entrypoint(band: str) -> str:
    fixed = f"s3://spectrum/{FIXED_KEY.format(band=band)}"
    common = "--memory-gb 0 --gpu-fraction 0.15 --max-parallel 2 --skip-done"
    return ("bash -c \"cd /workspace/spectrum-usage && "
            f"python -u -m training.ray.spatial15 --band {band} --experiments s2 --variants core4-p4 core4-krige --models dswinlstm_i --seeds 42 "
            f"--fixed-config {fixed} --prefix {NEW_PREFIX} {common} && "
            f"python -u -m training.ray.spatial15_origins --bands {band} --models dswinlstm_i --seeds 42 --variants core4-p4 core4-krige "
            f"--prefix {NEW_PREFIX} --output-prefix {ORIGIN_PREFIX} --skip-done --memory-gb 0 --gpu-fraction 0.15 --max-parallel 2\"")


def table_entrypoint() -> str:
    return ("bash -c \"cd /workspace/spectrum-usage && export PYTHONPATH=evaluation/analysis:.:$PYTHONPATH && "
            "export S3_ENDPOINT_URL=${SPECTRUM_S3_ENDPOINT_URL:-$S3_ENDPOINT_URL} AWS_ACCESS_KEY_ID=${SPECTRUM_AWS_ACCESS_KEY_ID:-$AWS_ACCESS_KEY_ID} "
            "AWS_SECRET_ACCESS_KEY=${SPECTRUM_AWS_SECRET_ACCESS_KEY:-$AWS_SECRET_ACCESS_KEY} && mkdir -p evaluation/results && "
            f"python -u evaluation/analysis/spatial15_table.py --prefix {NEW_PREFIX} --no-references --out evaluation/results/spatial15_dswin_tuned "
            "> evaluation/results/spatial15_dswin_tuned_tables.txt 2>&1 && python - <<'PY'\n"
            "import glob, os, sys\nsys.path.insert(0, '.')\nfrom training.ray.storage import spectrum_s3_client\nc = spectrum_s3_client()\n"
            "for f in sorted(glob.glob('evaluation/results/spatial15_dswin_tuned*')):\n"
            "    c.put_object(Bucket='spectrum', Key='analysis-outputs/' + os.path.basename(f), Body=open(f, 'rb').read()); print('uploaded', f)\nPY\"")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=600)
    p.add_argument("--giveup-hours", type=float, default=24.0)
    a = p.parse_args()
    sys.path.insert(0, "/workspace/spectrum-usage")
    from ray.job_submission import JobSubmissionClient

    from training.ray.storage import spectrum_s3_client
    from training.ray.tune_dswin15 import INHERITED, fixed_from_candidate

    client, jobs = spectrum_s3_client(), JobSubmissionClient("http://127.0.0.1:8265")
    inherited = json.loads(open(f"/workspace/spectrum-usage/{INHERITED}").read())
    deadline = time.time() + a.giveup_hours * 3600
    decided: dict[str, dict] = {}
    launched: dict[str, str] = {}
    existing = {j.submission_id for j in jobs.list_jobs()}

    def write_status(final: bool) -> None:
        body = {"rule": f"run only if the search winner beats the inherited validation objective by >= {THRESHOLD} and is not the inherited anchor",
                "bands": decided, "final": final, "updated": time.strftime("%F %T", time.gmtime())}
        client.put_object(Bucket="spectrum", Key=STATUS_KEY, Body=json.dumps(body, indent=1).encode())

    while len(decided) < len(BANDS) and time.time() < deadline:
        try:
            verdict = json.loads(client.get_object(Bucket="spectrum", Key="analysis-outputs/dswin_search_verdict.json")["Body"].read())
        except Exception:
            verdict = {"bands": {}}
        for band in BANDS:
            if band in decided:
                continue
            entry = (verdict.get("bands") or {}).get(band)
            action, reason = decide(entry)
            print(f"[dswin-p4krige] {time.strftime('%F %T', time.gmtime())} {band}: {action}: {reason}", flush=True)
            if action == "wait":
                continue
            decided[band] = {"decision": action, "reason": reason, "improvement": (entry or {}).get("improvement"), "winner": (entry or {}).get("winner")}
            if action == "run":
                label = f"tuned15:{entry['winner']}"
                fixed = fixed_from_candidate(entry["winner_parameters"], inherited, label)
                client.put_object(Bucket="spectrum", Key=FIXED_KEY.format(band=band), Body=json.dumps(fixed, indent=1).encode())
                sid = f"claude-uc-dswin-tuned-p4krige-{band.split('_')[0]}-1"
                if sid not in existing:
                    jobs.submit_job(submission_id=sid, entrypoint=entrypoint(band))
                launched[band] = sid
                decided[band]["job"] = sid
                decided[band]["configuration"] = label
                print(f"[dswin-p4krige] launched {sid} ({label})", flush=True)
        write_status(False)
        if len(decided) < len(BANDS):
            time.sleep(a.interval)
    write_status(True)
    if launched:
        while time.time() < deadline + 12 * 3600:
            states = {j.submission_id: str(j.status).split(".")[-1] for j in jobs.list_jobs()}
            if all(states.get(s) in ("SUCCEEDED", "FAILED", "STOPPED") for s in launched.values()):
                break
            time.sleep(300)
        if "claude-uc-dswin-tuned-table-1" not in {j.submission_id for j in jobs.list_jobs()}:
            jobs.submit_job(submission_id="claude-uc-dswin-tuned-table-1", entrypoint=table_entrypoint())
        print("[dswin-p4krige] table job submitted", flush=True)
    print("[dswin-p4krige] done " + json.dumps(decided), flush=True)


if __name__ == "__main__":
    main()
