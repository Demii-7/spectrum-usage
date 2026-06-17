#!/usr/bin/env python3
"""Upload collector output to object storage and prune old raw minute files."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message):
    print(f"[{utc_now()}] {message}", flush=True)


def parse_env_value(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path):
    if path is None:
        return
    if not path.exists():
        return

    loaded = 0
    with open(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                raise ValueError(f"invalid .env line {line_no}: missing '='")

            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"invalid .env line {line_no}: empty key")
            if key in os.environ:
                continue

            os.environ[key] = parse_env_value(value)
            loaded += 1

    log(f"loaded {loaded} value(s) from {path}")


def build_upload_command(args):
    source = str(args.source)
    dest = args.destination

    if args.tool == "aws":
        cmd = ["aws", "s3", "sync", source, dest]
        if args.dry_run:
            cmd.append("--dryrun")
    elif args.tool == "rclone":
        # Use copy, not sync: local pruning must not delete uploaded objects.
        cmd = ["rclone", "copy", source, dest]
        if args.dry_run:
            cmd.append("--dry-run")
    else:
        raise ValueError(f"unsupported tool: {args.tool}")

    cmd.extend(args.sync_arg)
    return cmd


def normalize_fsspec_destination(destination):
    if destination.startswith("s3://"):
        return destination[len("s3://"):].rstrip("/")
    return destination.rstrip("/")


def build_fsspec_kwargs(args):
    fs_kwargs = {}
    endpoint_url = args.endpoint_url or os.environ.get("S3_ENDPOINT_URL", "")
    if endpoint_url:
        fs_kwargs["client_kwargs"] = {"endpoint_url": endpoint_url}
    return fs_kwargs


def remote_size_matches(fs, remote_path, local_size):
    try:
        info = fs.info(remote_path)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return info.get("size") == local_size


def iter_local_files(source):
    for path in source.rglob("*"):
        if path.is_file():
            yield path


def run_fsspec_upload(args):
    try:
        import fsspec
    except ImportError:
        log("fsspec is not installed; install fsspec and s3fs")
        return False

    fs = None if args.dry_run else fsspec.filesystem("s3", **build_fsspec_kwargs(args))
    dest_root = normalize_fsspec_destination(args.destination)
    uploaded = 0
    skipped = 0
    failed = 0

    log(f"running fsspec upload to s3://{dest_root}")
    for local_path in iter_local_files(args.source):
        rel_path = local_path.relative_to(args.source).as_posix()
        remote_path = f"{dest_root}/{rel_path}"
        local_size = local_path.stat().st_size

        if args.dry_run:
            log(f"would upload {local_path} -> s3://{remote_path}")
            uploaded += 1
            continue

        if not args.force and remote_size_matches(fs, remote_path, local_size):
            skipped += 1
            continue

        try:
            parent = remote_path.rsplit("/", 1)[0]
            fs.mkdirs(parent, exist_ok=True)
            fs.put_file(str(local_path), remote_path)
            uploaded += 1
        except Exception as exc:
            failed += 1
            log(f"upload failed for {local_path}: {exc}")

    log(f"fsspec upload complete: uploaded={uploaded}, skipped={skipped}, failed={failed}")
    return failed == 0


def run_upload(args):
    if args.tool == "fsspec":
        return run_fsspec_upload(args)

    cmd = build_upload_command(args)
    log("running upload: " + " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        log(f"upload failed with exit code {completed.returncode}; skipping prune")
        return False
    log("upload completed")
    return True


def iter_prunable_minutes(source):
    for path in source.rglob("minute_*"):
        if path.parent.name != "raw":
            continue
        if path.is_dir() or path.is_file():
            yield path


def prune_old_minutes(source, max_age_seconds, dry_run):
    cutoff = time.time() - max_age_seconds
    removed = 0
    kept = 0

    for path in iter_prunable_minutes(source):
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue

        if mtime >= cutoff:
            kept += 1
            continue

        if dry_run:
            log(f"would remove {path}")
        elif path.is_dir():
            shutil.rmtree(path)
            log(f"removed {path}")
        else:
            path.unlink()
            log(f"removed {path}")
        removed += 1

    log(f"prune complete: removed={removed}, kept_recent={kept}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Periodically upload spectrum collector output to object storage and "
            "remove raw/minute_* paths older than the configured age."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Local collector output directory, for example ./data/week-run",
    )
    parser.add_argument(
        "destination",
        help="Object-storage destination, for example s3://bucket/prefix or remote:prefix",
    )
    parser.add_argument(
        "--tool",
        choices=("aws", "rclone", "fsspec"),
        default="aws",
        help="Upload tool to run. aws uses 'aws s3 sync'; rclone uses 'rclone copy'; fsspec uses s3fs.",
    )
    parser.add_argument(
        "--endpoint-url",
        default="",
        help="S3-compatible endpoint URL for --tool fsspec. Defaults to S3_ENDPOINT_URL if set.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Load environment variables from this file before uploading (default: .env if present).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="For --tool fsspec, upload files even when the remote object has the same size.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=300.0,
        help="Seconds between upload/prune cycles (default: 300).",
    )
    parser.add_argument(
        "--prune-age-seconds",
        type=float,
        default=3600.0,
        help="Remove raw/minute_* paths older than this many seconds after upload succeeds (default: 3600).",
    )
    parser.add_argument(
        "--sync-arg",
        action="append",
        default=[],
        help="Extra argument passed to the upload command. Repeat for multiple arguments.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one upload/prune cycle and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show upload and prune actions without deleting local files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.source = args.source.resolve()
    args.env_file = args.env_file.resolve() if args.env_file else None

    try:
        load_env_file(args.env_file)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.source.exists() or not args.source.is_dir():
        print(f"ERROR: source directory does not exist: {args.source}", file=sys.stderr)
        return 2
    if args.interval_seconds <= 0:
        print("ERROR: --interval-seconds must be positive", file=sys.stderr)
        return 2
    if args.prune_age_seconds <= 0:
        print("ERROR: --prune-age-seconds must be positive", file=sys.stderr)
        return 2

    log(f"source={args.source}")
    log(f"destination={args.destination}")
    log(f"tool={args.tool}")
    log(f"interval_seconds={args.interval_seconds:g}")
    log(f"prune_age_seconds={args.prune_age_seconds:g}")
    log(f"dry_run={args.dry_run}")

    while True:
        if run_upload(args):
            prune_old_minutes(args.source, args.prune_age_seconds, args.dry_run)

        if args.once:
            break

        time.sleep(args.interval_seconds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
