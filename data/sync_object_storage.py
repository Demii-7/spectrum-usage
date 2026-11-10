#!/usr/bin/env python3
"""Synchronize processed power CSVs or checkpoints from S3-compatible storage."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath


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
                if child_prefix.rstrip("/").split("/")[-1] != "raw":
                    pending.append(child_prefix)
            if not response.get("IsTruncated"):
                break
            continuation_token = response["NextContinuationToken"]
    return entries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", help="Defaults to S3_BUCKET, or SPECTRUM_S3_BUCKET with --spectrum-storage")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all-sites", action="store_true", help="Sync processed CSVs across the entire bucket")
    selection.add_argument("--sites", nargs="+", help="Sync these measurement prefixes without traversing unrelated bucket roots")
    selection.add_argument("--checkpoints", action="store_true", help="Sync checkpoints/ into the repository instead of processed CSVs into data/")
    parser.add_argument("--checkpoint-prefix", action="append", default=[],
                        help="Limit --checkpoints to this object-key prefix; repeat for multiple prefixes")
    parser.add_argument("--spectrum-storage", action="store_true", help="Use SPECTRUM_* credentials and endpoint instead of the default AWS/MinIO configuration")
    parser.add_argument("--inventory-out", type=Path, help="Save the selected object metadata as JSON")
    parser.add_argument("--inventory-in", type=Path, help="Use a saved inventory instead of listing the bucket")
    parser.add_argument("--checksums-out", type=Path, help="Verify sizes and save local SHA-256 checksums after syncing")
    parser.add_argument("--summary-only", action="store_true", help="Suppress per-file download messages")
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
    parser.add_argument("--output-dir", type=Path, help="Destination root; defaults to data/ for CSVs or the repository root for checkpoints")
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


def storage_settings(args):
    if not args.spectrum_storage:
        return args.bucket or os.environ.get("S3_BUCKET", "spectrum"), os.environ.get("S3_ENDPOINT_URL", ""), {}

    def credential(name):
        if os.environ.get(name):
            return os.environ[name]
        return Path(os.environ[f"{name}_FILE"]).read_text().strip()

    return (
        args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum"),
        os.environ["SPECTRUM_S3_ENDPOINT_URL"],
        {"key": credential("SPECTRUM_AWS_ACCESS_KEY_ID"),
         "secret": credential("SPECTRUM_AWS_SECRET_ACCESS_KEY")},
    )


def validate_key(key, checkpoints=False):
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid object key: {key}")
    if checkpoints:
        if not key.startswith("checkpoints/") or key.endswith("/"):
            raise ValueError(f"Invalid checkpoint key: {key}")
    elif "raw" in path.parts or not key.endswith("power_1mhz_avg_per_minute.csv"):
        raise ValueError(f"Invalid processed-data key: {key}")


def list_checkpoint_objects(fs, bucket, prefixes=None):
    entries = {}
    for prefix in prefixes or ["checkpoints/"]:
        request = {"Bucket": bucket, "Prefix": prefix}
        while True:
            response = fs.call_s3("list_objects_v2", **request)
            for item in response.get("Contents", []):
                if not item["Key"].endswith("/"):
                    entries[item["Key"]] = item
            if not response.get("IsTruncated"):
                break
            request["ContinuationToken"] = response["NextContinuationToken"]
    return entries


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    args = parse_args()
    if args.checkpoint_prefix and not args.checkpoints:
        print("ERROR: --checkpoint-prefix requires --checkpoints", file=sys.stderr)
        return 2
    if any(not prefix.startswith("checkpoints/") or ".." in PurePosixPath(prefix).parts
           for prefix in args.checkpoint_prefix):
        print("ERROR: checkpoint prefixes must stay under checkpoints/", file=sys.stderr)
        return 2
    kind = "checkpoints" if args.checkpoints else "processed-data"
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent
        if args.checkpoints:
            args.output_dir = args.output_dir.parent
    if args.workers <= 0 or args.parts <= 0:
        print("ERROR: --workers and --parts must be positive", file=sys.stderr)
        return 2
    if (args.all_sites or args.sites or args.checkpoints) and args.nodes:
        print("ERROR: --all-sites/--sites/--checkpoints cannot be combined with --node", file=sys.stderr)
        return 2
    try:
        load_env_file(args.env_file)
        import fsspec
    except (ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Install dependencies with: pip install fsspec s3fs", file=sys.stderr)
        return 2

    site_prefix = "" if args.all_sites else args.site.strip("/") + "/"
    listing_prefixes = [
        f"{site_prefix}{node.strip('/')}/"
        for node in args.nodes
    ] if args.nodes else [site_prefix]
    if args.sites:
        listing_prefixes = [site.strip("/") + "/" for site in args.sites]
    try:
        bucket, endpoint_url, credentials = storage_settings(args)
        fs = fsspec.filesystem("s3", **fs_kwargs(endpoint_url), **credentials)
        if args.inventory_in:
            inventory = json.loads(args.inventory_in.read_text())
            if inventory["bucket"] != bucket or inventory["endpoint"] != endpoint_url:
                raise ValueError("Inventory bucket/endpoint does not match selected storage")
            if inventory.get("kind", "processed-data") != kind:
                raise ValueError("Inventory kind does not match selected sync mode")
            entries = inventory["objects"]
        else:
            if args.checkpoints:
                entries = list_checkpoint_objects(fs, bucket, args.checkpoint_prefix)
            else:
                entries = {}
                for listing_prefix in listing_prefixes:
                    entries.update(list_processed_objects(fs, bucket, listing_prefix))
            entries = {key: {"Size": value["Size"], "ETag": value.get("ETag"),
                             "LastModified": value["LastModified"].timestamp()}
                       for key, value in entries.items()}
        for key in entries:
            validate_key(key, checkpoints=args.checkpoints)
        if args.checkpoint_prefix:
            entries = {key: info for key, info in entries.items()
                       if any(key.startswith(prefix) for prefix in args.checkpoint_prefix)}
        if not entries:
            raise ValueError(f"No {kind} objects found")
        if args.inventory_out:
            write_json(args.inventory_out, {"bucket": bucket, "endpoint": endpoint_url, "kind": kind, "objects": entries})
    except Exception as exc:
        print(f"ERROR: object storage listing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    downloads = []
    for key, info in sorted(entries.items()):
        local_path = args.output_dir / key
        remote_size = info.get("Size")
        remote_mtime = info.get("LastModified")
        if hasattr(remote_mtime, "timestamp"):
            remote_mtime = remote_mtime.timestamp()
        current = local_path.stat() if local_path.exists() else None
        if current and remote_size == current.st_size and (remote_mtime is None or remote_mtime <= current.st_mtime):
            continue
        downloads.append((key, local_path, remote_mtime))

    site_sizes = {}
    for key, info in entries.items():
        site = key.split("/", 1)[0]
        summary = site_sizes.setdefault(site, {"files": 0, "bytes": 0})
        summary["files"] += 1
        summary["bytes"] += info["Size"]
    print(json.dumps({"bucket": bucket, "sites": site_sizes, "files": len(entries),
                      "bytes": sum(info["Size"] for info in entries.values()),
                      "files_to_update": len(downloads)}, sort_keys=True))
    if args.dry_run:
        for key, local_path, _ in downloads:
            if not args.summary_only:
                print(f"would download s3://{bucket}/{key} -> {local_path}")
        print(f"{len(downloads)} file(s) would be updated")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(entries[key]["Size"] for key, _, _ in downloads)
    free_bytes = shutil.disk_usage(args.output_dir).free
    if required_bytes + 5 * 1024**3 > free_bytes:
        print(f"ERROR: downloads need {required_bytes} bytes plus 5 GiB free reserve; available={free_bytes}", file=sys.stderr)
        return 1

    def download_one(item):
        key, local_path, remote_mtime = item
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = local_path.with_name(local_path.name + ".part")
        if not args.summary_only:
            print(f"downloading s3://{bucket}/{key} -> {local_path}")
        fs.get(
            f"{bucket}/{key}",
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
    if args.checksums_out:
        checksums = {}
        for key, info in sorted(entries.items()):
            path = args.output_dir / key
            if path.stat().st_size != info["Size"]:
                raise ValueError(f"Size mismatch: {key}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            checksums[key] = {"size": info["Size"], "sha256": digest}
        write_json(args.checksums_out, {"bucket": bucket, "objects": checksums})
        print(f"Verified sizes and SHA-256 checksums for {len(checksums)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
