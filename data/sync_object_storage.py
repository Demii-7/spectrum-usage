#!/usr/bin/env python3
"""Synchronize power CSV files from S3-compatible object storage into data/."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from pathlib import Path


def load_env_file(path):
    """Load simple KEY=VALUE entries without overriding the shell environment."""
    if not path.exists():
        return
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"invalid .env line {line_no}: missing '='")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def fs_kwargs(endpoint_url):
    kwargs = {"skip_instance_cache": True, "use_listings_cache": False}
    if endpoint_url:
        kwargs["client_kwargs"] = {"endpoint_url": endpoint_url}
        kwargs["config_kwargs"] = {"s3": {"addressing_style": "path"}}
    return kwargs


def list_processed_objects(fs, bucket, prefix):
    entries = {}
    pending = [prefix]
    while pending:
        current_prefix = pending.pop()
        continuation_token = None
        while True:
            request = {
                "Bucket": bucket,
                "Prefix": current_prefix,
                "Delimiter": "/",
            }
            if continuation_token:
                request["ContinuationToken"] = continuation_token
            response = fs.call_s3("list_objects_v2", **request)
            for item in response.get("Contents", []):
                if item["Key"].endswith("power_1mhz_avg_per_minute.csv"):
                    entries[item["Key"]] = item
            for common_prefix in response.get("CommonPrefixes", []):
                child_prefix = common_prefix["Prefix"]
                if not child_prefix.rstrip("/").endswith("/raw"):
                    pending.append(child_prefix)
            if not response.get("IsTruncated"):
                break
            continuation_token = response["NextContinuationToken"]
    return entries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "spectrum"))
    parser.add_argument(
        "--site",
        default=os.environ.get("S3_SITE", os.environ.get("S3_PREFIX", "powder")),
        help="Testbed/dataset prefix to sync, such as powder or aerpaw.",
    )
    parser.add_argument(
        "--node",
        action="append",
        dest="nodes",
        default=[],
        help="Node within --site to sync. Repeat for multiple nodes; defaults to all nodes.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent / ".env")
    parser.add_argument("--dry-run", action="store_true", help="List downloads without downloading")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("S3_SYNC_WORKERS", "4")),
        help="Maximum files downloaded concurrently (default: 4).",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=int(os.environ.get("S3_SYNC_PARTS", "4")),
        help="Concurrent range downloads per file (default: 4).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0 or args.parts <= 0:
        print("ERROR: --workers and --parts must be positive", file=sys.stderr)
        return 2
    try:
        load_env_file(args.env_file)
        import fsspec
    except (ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Install dependencies with: pip install fsspec s3fs", file=sys.stderr)
        return 2

    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")
    site_prefix = args.site.strip("/") + "/"
    listing_prefixes = [
        f"{site_prefix}{node.strip('/')}/"
        for node in args.nodes
    ] if args.nodes else [site_prefix]
    try:
        fs = fsspec.filesystem("s3", **fs_kwargs(endpoint_url))
        entries = {}
        for listing_prefix in listing_prefixes:
            entries.update(list_processed_objects(fs, args.bucket, listing_prefix))
    except Exception as exc:
        print(f"ERROR: object storage listing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    downloads = []
    for key, info in sorted(entries.items()):
        if not key.endswith("power_1mhz_avg_per_minute.csv"):
            continue
        local_path = args.output_dir / key
        remote_size = info.get("Size")
        remote_mtime = info.get("LastModified")
        if hasattr(remote_mtime, "timestamp"):
            remote_mtime = remote_mtime.timestamp()
        current = local_path.stat() if local_path.exists() else None
        if current and remote_size == current.st_size and (remote_mtime is None or remote_mtime <= current.st_mtime):
            continue
        downloads.append((key, local_path, remote_mtime))

    if args.dry_run:
        for key, local_path, _ in downloads:
            print(f"would download s3://{args.bucket}/{key} -> {local_path}")
        print(f"{len(downloads)} file(s) would be updated")
        return 0

    def download_one(item):
        key, local_path, remote_mtime = item
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = local_path.with_name(local_path.name + ".part")
        print(f"downloading s3://{args.bucket}/{key} -> {local_path}")
        fs.get(
            f"{args.bucket}/{key}",
            str(temporary_path),
            max_concurrency=args.parts,
            chunksize=64 * 1024 * 1024,
        )
        if remote_mtime is not None:
            os.utime(temporary_path, (remote_mtime, remote_mtime))
        os.replace(temporary_path, local_path)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, item) for item in downloads]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures += 1
                print(f"ERROR: download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    if failures:
        print(f"{failures} download(s) failed", file=sys.stderr)
        return 1
    print(f"{len(downloads)} file(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
