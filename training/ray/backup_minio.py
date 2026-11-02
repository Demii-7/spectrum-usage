"""Back up the Ray campaign's MinIO state to the durable Chameleon bucket, or restore it.

MinIO lives on one cluster host, so the experiment records (trial tables, costs, search
state, campaign markers) vanish with that host. ``backup`` copies them to
``s3://<spectrum bucket>/<backup-prefix>/...``; ``restore`` copies them into a (new) MinIO at
the same keys so ``campaign``/``run``/``replicate`` can ``--resume``. Both skip objects that
are already present with the same size, so re-running is cheap.

Run as a Ray job on the head (it needs the MinIO credentials and the SPECTRUM_* ones).
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MINIO_PREFIX = "spectrum-usage/ray/oct2026-core4"  # also covers ...-core4-r2
DEFAULT_BACKUP_PREFIX = "oct2026-minio-backup"


def plan_copy(direction: str, keys_and_sizes, *, minio_prefix: str, backup_prefix: str,
              dest_prefix: str | None = None, only: str | None = None):
    """Map source objects to destination keys (pure function, unit-tested)."""
    plan = []
    for key, size in keys_and_sizes:
        if direction == "backup":
            relative = key
        else:
            if not key.startswith(f"{backup_prefix}/"):
                continue
            relative = key[len(backup_prefix) + 1:]
        if only and only not in relative:
            continue
        if direction == "backup":
            target = f"{backup_prefix}/{relative}"
        else:
            target = relative
            if dest_prefix and relative.startswith(minio_prefix):
                target = dest_prefix + relative[len(minio_prefix):]
        plan.append((key, target, size))
    return plan


def _list(client, bucket: str, prefix: str):
    out = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        out += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return out


def run(args: argparse.Namespace) -> int:
    import boto3

    from training.ray.storage import MinIOConfig, spectrum_s3_client

    os.environ.update(MinIOConfig(args.minio_bucket, "x", args.minio_endpoint).environment())
    minio = boto3.client("s3", endpoint_url=args.minio_endpoint, region_name="us-east-1")
    durable = spectrum_s3_client()
    durable_bucket = args.durable_bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    if args.direction == "backup":
        src, src_bucket, dst, dst_bucket = minio, args.minio_bucket, durable, durable_bucket
        listing = _list(src, src_bucket, args.minio_prefix)
    else:
        src, src_bucket, dst, dst_bucket = durable, durable_bucket, minio, args.minio_bucket
        listing = _list(src, src_bucket, f"{args.backup_prefix}/{args.minio_prefix}")
    plan = plan_copy(args.direction, listing, minio_prefix=args.minio_prefix,
                     backup_prefix=args.backup_prefix, dest_prefix=args.dest_prefix, only=args.only)
    existing_prefix = (f"{args.backup_prefix}/" if args.direction == "backup"
                       else (args.dest_prefix or args.minio_prefix))
    existing = dict(_list(dst, dst_bucket, existing_prefix))
    todo = [(s, t, n) for s, t, n in plan if existing.get(t) != n]
    print(f"BACKUP {args.direction}: {len(plan)} objects ({sum(n for *_, n in plan)/2**30:.2f} GiB), "
          f"{len(todo)} to copy", flush=True)
    if args.dry_run:
        return 0

    def copy(item):
        source_key, target_key, size = item
        with tempfile.NamedTemporaryFile() as tmp:
            src.download_file(src_bucket, source_key, tmp.name)
            dst.upload_file(tmp.name, dst_bucket, target_key)
        return size

    started, copied, failed = time.time(), 0, []
    with ThreadPoolExecutor(args.threads) as pool:
        for item, future in [(i, pool.submit(copy, i)) for i in todo]:
            try:
                copied += future.result()
            except Exception as error:
                failed.append((item[0], str(error)[:80]))
    print(f"BACKUP_DONE {args.direction}: copied {len(todo) - len(failed)} objects, "
          f"{copied/2**30:.2f} GiB in {time.time()-started:.0f}s; failed={len(failed)} {failed[:3]}", flush=True)
    return 1 if failed else 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("backup", "restore"), required=True)
    parser.add_argument("--minio-bucket", default="spectrum-usage")
    parser.add_argument("--minio-endpoint", default="http://minio:9000")
    parser.add_argument("--minio-prefix", default=DEFAULT_MINIO_PREFIX)
    parser.add_argument("--backup-prefix", default=DEFAULT_BACKUP_PREFIX)
    parser.add_argument("--durable-bucket", default=None, help="default: $SPECTRUM_S3_BUCKET or 'spectrum'")
    parser.add_argument("--only", help="only objects whose key contains this substring")
    parser.add_argument("--dest-prefix", help="restore under a different MinIO prefix (testing)")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
