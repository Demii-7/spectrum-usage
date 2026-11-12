#!/usr/bin/env python3
"""Sync committed source to selected Ray clusters without restarting services."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

SCRIPT_DIR = Path(__file__).resolve().parent
DOCKER_DIR = SCRIPT_DIR.parent / "docker"

TARGETS = {
    "h100-three": [
        ("ray-h100-three-1", ["-F", str(DOCKER_DIR / "h100-three-ssh.conf")]),
    ],
    "ncar": [
        (f"ray-ncar-{index}", ["-F", str(DOCKER_DIR / "ncar-ssh.conf")])
        for index in range(1, 3)
    ] + [
        (f"ray-ncar-cpu-{index}", ["-F", str(DOCKER_DIR / "ncar-ssh.conf")])
        for index in (3, 4)
    ],
    "final": [
        (f"ray-final-{index}", ["-F", str(DOCKER_DIR / "final-ssh.conf")])
        for index in range(1, 7)
    ],
    "mixed-gpu": [
        (name, ["-F", str(DOCKER_DIR / "h100-ssh.conf")])
        for name in ("ray-worker-2", "ray-worker-1", "ray-new-1", "ray-new-2", "ray-new-3", "ray-new-4")
    ],
    "tacc": [
        (name, ["-F", str(DOCKER_DIR / "tacc-ssh.conf")])
        for name in ("ray-tacc-1", "ray-tacc-2", "ray-tacc-3", "ray-tacc-4")
    ],
    "uc": [
        (f"ray-uc-{index}", ["-F", str(DOCKER_DIR / "uc-ssh.conf")])
        for index in range(1, 9)
    ] + [("ray-uc-extra-1", ["-F", str(DOCKER_DIR / "uc-ssh.conf")])],
}

# The former TACC gateway IP now belongs to h100-three; retain its old
# inventory only as an explicit historical selector.
DEFAULT_CLUSTERS = ("final", "uc", "ncar", "h100-three")

# Pending deployment helpers that are newer than the selected commit.
OVERRIDES = {
    "data/sync_object_storage.py",
    "training/ray/scripts/sync-cluster-code.py",
    "training/ray/scripts/sync-cluster-data.sh",
    "training/ray/scripts/configure-storage-secrets.py",
    "training/ray/docker/README.md",
    "training/ray/docker/standalone-nodes.sh",
}

DEPLOY = '''import io,os,pathlib,sys,tarfile,tempfile
root=pathlib.Path('/home/cc/spectrum-usage');assert root.is_dir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read())) as tar:
    for item in tar:
        if not item.isfile(): continue
        path=root/item.name;path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.sync-',delete=False) as stream:
            temporary=pathlib.Path(stream.name);stream.write(tar.extractfile(item).read())
        os.chmod(temporary,item.mode);os.utime(temporary,(item.mtime,item.mtime));temporary.replace(path)
'''

VERIFY = '''import hashlib,json,pathlib,sys
expected=json.load(sys.stdin);root=pathlib.Path('/home/cc/spectrum-usage')
bad=[name for name,digest in expected.items() if not(root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
print(json.dumps({'verified_files':len(expected),'mismatches':bad}));sys.exit(bool(bad))
'''


def build_archive(revision):
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision], text=True
    ).splitlines()
    paths = ["training", "models", "tests", "data/sync_object_storage.py", "data/locations"]
    paths += [name for name in tree if name.startswith("evaluation/") and not name.startswith("evaluation/results/")]
    paths += [
        name for name in tree
        if "/" not in name and Path(name).suffix in {".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".txt"}
    ]
    original = subprocess.check_output(["git", "archive", "--format=tar.gz", revision, *paths])
    expected = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(original)) as source, \
         tarfile.open(fileobj=buffer, mode="w:gz") as destination:
        for entry in source:
            if entry.isfile():
                value = (
                    Path(entry.name).read_bytes()
                    if entry.name in OVERRIDES
                    else source.extractfile(entry).read()
                )
                entry.size = len(value)
                expected[entry.name] = hashlib.sha256(value).hexdigest()
                destination.addfile(entry, io.BytesIO(value))
            else:
                if entry.issym() or entry.islnk():
                    raise ValueError(f"Unexpected source link: {entry.name}")
                destination.addfile(entry)
    return buffer.getvalue(), expected


def sync(target, revision, archive, expected):
    cluster, host, options = target

    def ssh(command, **kwargs):
        return subprocess.run(
            ["ssh", *options, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", f"cc@{host}", command],
            capture_output=True,
            check=True,
            **kwargs,
        )

    before = ssh("docker ps --no-trunc --format '{{.Names}} {{.ID}} {{.CreatedAt}}'", text=True).stdout
    ssh("python3 -c " + shlex.quote(DEPLOY), input=archive)
    checked = json.loads(
        ssh("python3 -c " + shlex.quote(VERIFY), input=json.dumps(expected).encode()).stdout
    )
    after = ssh("docker ps --no-trunc --format '{{.Names}} {{.ID}} {{.CreatedAt}}'", text=True).stdout
    if sorted(before.splitlines()) != sorted(after.splitlines()):
        raise RuntimeError(f"Container identities changed on {host}")
    ssh("tee /home/cc/spectrum-source-revision >/dev/null", input=(revision + "\n").encode())
    result = {"cluster": cluster, "host": host, "revision": revision, **checked, "containers_unchanged": True}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="origin/integrate")
    parser.add_argument("--cluster", action="append", choices=sorted(TARGETS), default=[])
    args = parser.parse_args()
    subprocess.run(["git", "fetch", "origin"], check=True)
    revision = subprocess.check_output(["git", "rev-parse", args.revision], text=True).strip()
    clusters = args.cluster or DEFAULT_CLUSTERS
    targets = [
        (cluster, host, options)
        for cluster in clusters
        for host, options in TARGETS[cluster]
    ]
    archive, expected = build_archive(revision)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda target: sync(target, revision, archive, expected), targets))
    print(f"Synced {revision} to {len(targets)} hosts")


if __name__ == "__main__":
    main()
