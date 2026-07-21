#!/usr/bin/env python3
"""Synchronize power CSV files from S3-compatible object storage into data/."""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "spectrum"))
    parser.add_argument("--prefix", default=os.environ.get("S3_PREFIX", "powder"))
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        default=[],
        help="Site to sync. Repeat for multiple sites; defaults to all sites.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent / ".env")
    parser.add_argument("--dry-run", action="store_true", help="List downloads without downloading")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        load_env_file(args.env_file)
        import fsspec
    except (ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Install dependencies with: pip install fsspec s3fs", file=sys.stderr)
        return 2

    remote_root = f"{args.bucket.strip('/')}/{args.prefix.strip('/')}".strip("/")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")
    try:
        fs = fsspec.filesystem("s3", **fs_kwargs(endpoint_url))
        entries = fs.find(remote_root, detail=True)
    except Exception as exc:
        print(f"ERROR: object storage listing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    csv_entries = entries.items() if isinstance(entries, dict) else ((entry, {}) for entry in entries)
    downloaded = 0
    for remote_path, info in sorted(csv_entries):
        if not remote_path.endswith("power_1mhz_avg_per_minute.csv"):
            continue
        relative_remote_path = remote_path.removeprefix(remote_root).lstrip("/")
        site = relative_remote_path.split("/", 1)[0]
        if args.sites and site not in args.sites:
            continue
        relative_path = Path(remote_path).relative_to(Path(args.bucket))
        local_path = args.output_dir / relative_path
        remote_size = info.get("size")
        remote_mtime = info.get("mtime") or info.get("LastModified")
        if hasattr(remote_mtime, "timestamp"):
            remote_mtime = remote_mtime.timestamp()
        current = local_path.stat() if local_path.exists() else None
        if current and remote_size == current.st_size and (remote_mtime is None or remote_mtime <= current.st_mtime):
            continue
        print(f"{'would download' if args.dry_run else 'downloading'} s3://{remote_path} -> {local_path}")
        if not args.dry_run:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fs.get(remote_path, str(local_path))
            if remote_mtime is not None:
                os.utime(local_path, (remote_mtime, remote_mtime))
        downloaded += 1
    print(f"{downloaded} file(s) {'would be ' if args.dry_run else ''}updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
