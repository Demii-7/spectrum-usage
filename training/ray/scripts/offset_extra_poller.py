"""Cluster-side poller for the offset-extra test: reruns the 4D offset driver when new seeds' weights appear in S3, then rebuilds the tables.

    python training/ray/scripts/offset_extra_poller.py [--interval 600] [--max-hours 24]

Every ``--interval`` seconds: for each (model, band, seed) of ResidualConvLSTM and DSwinLSTM-I (seeds 40-44) whose tuning-final weights exist and whose
``offset-extra-2026-11/<model>/<band>/humanities/seed-N/summary.json`` is not ok, it runs ``training.ray.offset_response_sts --discover-seeds --skip-done``
(ResidualConvLSTM on CPU, DSwinLSTM-I on a 0.25 GPU share), and the TSS-LCD tuned 15-minute finals (``tsslcd_authors --eval-only --input-offsets``, seeds 40-44,
both bands, 0.1 GPU share), then ``run_offset_extra_table.sh``. DSwinLSTM-I and TSS-LCD run only after their zero-offset checks on the GPU passed (max |difference| <=
1e-4 dB; the smoke runs in ``offset-extra-2026-11-smoke-gpu``); a failed check is recorded in ``offset-extra-2026-11/CHECK_FAILED.json`` and that model is not run.
Stops when all 30 cells are done or after ``--max-hours`` (the table is rebuilt once more before it exits). Meant to run as a Ray job on the head (run_after.py only waits for jobs of the same cluster, and the
missing weights come from jobs on another cluster).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

sys.path.insert(0, "/workspace/spectrum-usage")
from training.ray.storage import spectrum_s3_client  # noqa: E402

BANDS = ("2400_2600", "600_800")
MODELS = {"residualconvlstm": ["--device", "cpu", "--threads", "8", "--max-parallel", "3", "--memory-gb", "8"],
          "dswinlstm_i": ["--device", "cuda", "--gpu-fraction", "0.25", "--threads", "4", "--max-parallel", "4", "--memory-gb", "8"]}
SEEDS = (40, 41, 42, 43, 44)
PREFIX = "offset-extra-2026-11"


SMOKE = "offset-extra-2026-11-smoke-gpu"
TOLERANCE = 1e-4


def read_json(client, bucket: str, key: str):
    try:
        return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:
        return None


def check_state(client, bucket: str, model: str) -> bool | None:
    """True if the GPU zero-offset check of this model passed, False if it failed, None if its smoke results do not exist yet."""
    if model == "dswinlstm_i":
        recs = [read_json(client, bucket, f"{SMOKE}/dswinlstm_i/{band}/humanities/seed-42/summary.json") for band in BANDS]
    else:
        recs = [read_json(client, bucket, f"{SMOKE}/tsslcd-records/savgol-off/seed-40.json")]
    if any(r is None or not r.get("ok") for r in recs):
        return None
    diffs = [r.get("zero_offset_max_abs_diff_db") for r in recs]
    return all(d is not None and d <= TOLERANCE for d in diffs)


def tsslcd_todo(client, bucket: str) -> list:
    return [(band, seed) for band in BANDS for seed in SEEDS
            if not (read_json(client, bucket, f"{PREFIX}/tsslcd/{band}/humanities/seed-{seed}/summary.json") or {}).get("ok")]


def tsslcd_command(band: str) -> list:
    source = "tsslcd-authors-15min-tuned" if band == "600_800" else "tsslcd-authors-15min-tuned-2400"
    return [sys.executable, "-u", "-m", "training.ray.tsslcd_authors", "--eval-only", "--resolution", "15min", "--band", band, "--savgol", "off",
            "--seeds", *map(str, SEEDS), "--weights-prefix", source, "--output-prefix", f"{PREFIX}/tsslcd-records" + ("" if band == "600_800" else "-2400"),
            "--offset-prefix", PREFIX, "--input-offsets", "-20", "-10", "-5", "5", "10", "20", "--gpu-fraction", "0.1"]


def status(client, bucket: str) -> tuple[list, list]:
    """(todo, done): lists of (model, band, seed) with weights and no ok summary / with an ok summary."""
    todo, done = [], []
    for model in MODELS:
        for band in BANDS:
            for seed in SEEDS:
                try:
                    ok = bool(json.loads(client.get_object(Bucket=bucket, Key=f"{PREFIX}/{model}/{band}/humanities/seed-{seed}/summary.json")["Body"].read()).get("ok"))
                except Exception:
                    ok = False
                if ok:
                    done.append((model, band, seed))
                    continue
                prefix = f"tuning/powder15_4d/{band}/4d/{model}/weights/seed-{seed}/"
                if any(o["Key"].endswith(".pt") for o in client.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])):
                    todo.append((model, band, seed))
    return todo, done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=600)
    p.add_argument("--max-hours", type=float, default=24.0)
    p.add_argument("--bucket", default="spectrum")
    a = p.parse_args()
    client = spectrum_s3_client()
    deadline = time.time() + a.max_hours * 3600
    last_built = None             # (cells done at the last table build): rebuild whenever cells appeared, also from drivers started elsewhere
    while True:
        todo, done = status(client, a.bucket)
        total = len(MODELS) * len(BANDS) * len(SEEDS)        # 4D cells (the TSS-LCD cells are counted separately)
        print(f"[offset-extra-poller] {time.strftime('%F %T', time.gmtime())} done {len(done)}/{total}, runnable {len(todo)}", flush=True)
        ran = False
        for model, extra in MODELS.items():
            mine = [t for t in todo if t[0] == model]
            if not mine:
                continue
            if model == "dswinlstm_i":
                state = check_state(client, a.bucket, model)
                if state is None:
                    print("[offset-extra-poller] DSwinLSTM-I: GPU zero-offset check not available yet, waiting", flush=True)
                    continue
                if state is False:
                    client.put_object(Bucket=a.bucket, Key=f"{PREFIX}/CHECK_FAILED_dswinlstm_i.json", Body=json.dumps({"model": model, "tolerance_db": TOLERANCE}).encode())
                    print("[offset-extra-poller] DSwinLSTM-I zero-offset check FAILED, not running it", flush=True)
                    continue
            cmd = [sys.executable, "-u", "-m", "training.ray.offset_response_sts", "--models", model, "--bands", *BANDS, "--seeds", *map(str, SEEDS),
                   "--discover-seeds", "--skip-done", "--layout", "model-first", "--prefix", PREFIX, *extra]
            print("[offset-extra-poller] run", " ".join(cmd[3:]), flush=True)
            subprocess.run(cmd, cwd="/workspace/spectrum-usage", check=False)
            ran = True
        if tsslcd_todo(client, a.bucket):
            state = check_state(client, a.bucket, "tsslcd")
            if state is None:
                print("[offset-extra-poller] TSS-LCD: GPU zero-offset check not available yet, waiting", flush=True)
            elif state is False:
                client.put_object(Bucket=a.bucket, Key=f"{PREFIX}/CHECK_FAILED_tsslcd.json", Body=json.dumps({"model": "tsslcd", "tolerance_db": TOLERANCE}).encode())
                print("[offset-extra-poller] TSS-LCD zero-offset check FAILED, not running it", flush=True)
            else:
                for band in BANDS:
                    print("[offset-extra-poller] run tsslcd", band, flush=True)
                    subprocess.run(tsslcd_command(band), cwd="/workspace/spectrum-usage", check=False)
                ran = True
        todo, done = status(client, a.bucket)
        built_for = (len(done), len(tsslcd_todo(client, a.bucket)))
        if ran or built_for != last_built:
            subprocess.run(["bash", "training/ray/scripts/run_offset_extra_table.sh"], cwd="/workspace/spectrum-usage", check=False)
            last_built = built_for
        if len(done) == len(MODELS) * len(BANDS) * len(SEEDS) and not tsslcd_todo(client, a.bucket):
            print("[offset-extra-poller] all cells done", flush=True)
            return
        if time.time() > deadline:
            print(f"[offset-extra-poller] gave up after {a.max_hours} h: done {len(done)}, missing {sorted(set(t for t in todo))} plus seeds without weights", flush=True)
            subprocess.run(["bash", "training/ray/scripts/run_offset_extra_table.sh"], cwd="/workspace/spectrum-usage", check=False)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
