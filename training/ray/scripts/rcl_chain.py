"""Score the ResidualConvLSTM tuning finals of one band as soon as they exist (cluster-side; no operator needed).

    python training/ray/scripts/rcl_chain.py --band 600_800 --h100-prefix tuning-h100 --deadline-utc 2026-11-12T20:00Z
    python training/ray/scripts/rcl_chain.py --band 2400_2600

Polls S3 until every seed has an ok final under ``tuning/powder15_4d/<band>/4d/residualconvlstm`` -- or under ``--h100-prefix`` (a second
cluster's copy of the same finals, same winner configuration; adopted into ``tuning/`` only when ``tuning/`` has no ok record for the seed,
weights and sidecar first, the record last). At the deadline it goes on with the seeds that exist (every command is resumable with
--skip-done, so a later rerun adds the rest). Then, in parallel:
  1. eval-only scoring at the physical receivers (map, receivers, left-out / added receivers),
  2. permuted coordinates x5 (no retraining),
  3. the neighbour ablation (real, shift96, shift210 = 52.5 h, level), whose ``real`` mode also gives the per-origin errors.
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

MODEL = "residualconvlstm"


def ok_final(client, bucket, prefix, band, seed):
    try:
        rec = json.loads(client.get_object(Bucket=bucket, Key=f"{prefix}/powder15_4d/{band}/4d/{MODEL}/final/seed-{seed}.json")["Body"].read())
        return bool(rec.get("ok"))
    except Exception:
        return False


def adopt(client, bucket, src, dst, band, seed):
    """Copy weights + sidecar, then the final record, from ``src`` to ``dst`` (record last: it is what marks the seed as done)."""
    base_s = f"{src}/powder15_4d/{band}/4d/{MODEL}/"
    base_d = f"{dst}/powder15_4d/{band}/4d/{MODEL}/"
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=base_s + f"weights/seed-{seed}/")
            for o in page.get("Contents", [])]
    for k in keys:
        client.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": k}, Key=base_d + k[len(base_s):])
    client.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": base_s + f"final/seed-{seed}.json"}, Key=base_d + f"final/seed-{seed}.json")
    print(f"[rcl-chain] adopted seed {seed} of {band} from {src} ({len(keys)} weight files)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--band", required=True, choices=("600_800", "2400_2600"))
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--h100-prefix", default=None)
    p.add_argument("--deadline-utc", default=None, help="e.g. 2026-11-12T20:00Z: go on with the seeds that exist then")
    p.add_argument("--interval", type=int, default=600)
    p.add_argument("--gpu-fraction", type=float, default=0.25)
    p.add_argument("--nbr-gpu-fraction", type=float, default=0.5)
    a = p.parse_args()
    import sys
    sys.path.insert(0, "/workspace/spectrum-usage")
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), "spectrum"
    deadline = datetime.strptime(a.deadline_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) if a.deadline_utc else None
    while True:
        have = []
        for s in a.seeds:
            if not ok_final(client, bucket, "tuning", a.band, s) and a.h100_prefix and ok_final(client, bucket, a.h100_prefix, a.band, s):
                adopt(client, bucket, a.h100_prefix, "tuning", a.band, s)
            if ok_final(client, bucket, "tuning", a.band, s):
                have.append(s)
        late = deadline is not None and datetime.now(timezone.utc) >= deadline
        print(f"[rcl-chain] {time.strftime('%F %T', time.gmtime())} {a.band}: finals for seeds {have} of {a.seeds}{' (deadline reached)' if late else ''}", flush=True)
        if len(have) == len(a.seeds) or (late and have):
            break
        time.sleep(a.interval)
    seeds = " ".join(map(str, have))
    common = f"--tuned --experiments s2 --variants core4-p2 --models {MODEL} --seeds {seeds} --prefix spatial15-tuned-oct2026 --memory-gb 16 " \
             f"--max-parallel 5 --gpu-fraction {a.gpu_fraction} --skip-done --band {a.band}"
    cmds = [f"python -u -m training.ray.spatial15 {common} --eval-only --eval-from-tuning",
            f"python -u -m training.ray.spatial15 {common} --permuted 5",
            f"python -u -m training.ray.neighbor_ablation --models {MODEL} --bands {a.band} --seeds {seeds} --gpu-fraction {a.nbr_gpu_fraction} "
            f"--max-parallel 5 --skip-done"]
    procs = [subprocess.Popen(c, shell=True, cwd="/workspace/spectrum-usage") for c in cmds]
    codes = [x.wait() for x in procs]
    print(f"[rcl-chain] {a.band} scoring exit codes {codes}", flush=True)
    raise SystemExit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
