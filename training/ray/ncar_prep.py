"""Step 0 of the NCAR transfer batch: build the converted 15-minute target files and the large-model runtime on ``ray-ncar-1``.

``convert``  builds, with the existing converters (``transfer_external.convert`` and ``powder15.ensure_powder15``, unchanged), every converted
             15-minute file the transfer batch reads: AERPAW/ARA/COSMOS (``transfer_external_intervals.csv``), the June-July POWDER runs
             (``transfer_july_intervals.csv``), the 07-03 800-1000/5725-5925 MHz files (``transfer_crossband_intervals.csv``) and the six
             receivers' October runs (the test-run file also holds week 2). Only rows with ``use`` true are built; existing files are reused.
             Every file gets a fingerprint (SHA-256, rows, first/last timestamp, column checksum) so it can be compared with another cluster's copy.
``fingerprint``  the same fingerprint for files that already exist (``--paths``), no conversion; runs anywhere with pandas.
``setup``    installs momentfm (no dependencies) next to the isolated Transformers in ``/workspace/pylibs`` (``tune_direct.setup_timeran_node``),
             caches GPT-2, BERT and MOMENT-1-small under ``/workspace/hf-cache`` and the TimeRAN checkpoint, removes any duplicate NumPy from
             ``pylibs``, then checks in a fresh offline process (``HF_HUB_OFFLINE=1``) that every backbone loads from the cache.

    python -m training.ray.ncar_prep convert --output-key transfer-ncar-2026-11/step0/conversions.json
    python -m training.ray.ncar_prep setup --output-key transfer-ncar-2026-11/step0/setup.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

INTERVAL_TABLES = {"external": "evaluation/analysis/transfer_external_intervals.csv",
                   "july": "evaluation/analysis/transfer_july_intervals.csv",
                   "crossband": "evaluation/analysis/transfer_crossband_intervals.csv"}
SIX = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1", "guesthouse-nuc1", "law73-nuc1")
BACKBONES = ("openai-community/gpt2", "gpt2", "google-bert/bert-base-uncased", "bert-base-uncased")


def fingerprint(path: str | Path) -> dict[str, Any]:
    """SHA-256 plus format-independent facts: rows, first/last timestamp, and a checksum of the values (sum per column, then summed)."""
    import numpy as np
    import pandas as pd

    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)
    values = frame.drop(columns=["timestamp_utc"]).to_numpy(dtype=np.float64)
    return {"path": str(path), "sha256": digest, "rows": int(len(frame)), "columns": int(values.shape[1]),
            "first": str(frame["timestamp_utc"].iloc[0]), "last": str(frame["timestamp_utc"].iloc[-1]),
            "value_sum": float(np.nansum(values)), "nan_count": int(np.isnan(values).sum())}


def convert_one(group: str, target: dict[str, Any], repo_root: str) -> dict[str, Any]:
    os.chdir(repo_root)
    sys.path.insert(0, repo_root)
    from training.ray import transfer_external as te

    if group == "crossband":
        from training.ray import transfer_crossband as tc

        tc.register_bands()
    row = {"group": group, **{k: target[k] for k in ("platform", "site", "run", "band")}}
    try:
        te.convert(repo_root, target)
        out = te.converted_path(repo_root, target)
        row.update(ok=True, relpath=str(out.relative_to(repo_root)), **fingerprint(out))
    except Exception as error:
        row.update(ok=False, error=f"{type(error).__name__}: {error}"[:400])
    return row


def powder15_one(node: str, band: str, repo_root: str) -> list[dict[str, Any]]:
    os.chdir(repo_root)
    sys.path.insert(0, repo_root)
    from training.ray.powder15 import OUT_NAME, OUT_ROOT, TEST_RUN, TRAIN_RUN, ensure_powder15

    ensure_powder15(band, repo_root, (node,))
    rows = []
    for run in (TRAIN_RUN, TEST_RUN):
        out = Path(repo_root) / OUT_ROOT / node / run / band / OUT_NAME
        rows.append({"group": "powder15", "platform": "powder", "site": node, "run": run, "band": band, "ok": True,
                     "relpath": str(out.relative_to(repo_root)), **fingerprint(out)})
    return rows


def put_json(key: str, payload: Any) -> None:
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    try:
        client.head_object(Bucket=bucket, Key=key)
        raise SystemExit(f"refusing to overwrite existing s3://{bucket}/{key}")
    except client.exceptions.ClientError:
        pass
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload, indent=1, default=str).encode())
    print(f"wrote s3://{bucket}/{key}", flush=True)


def cmd_convert(a: argparse.Namespace) -> None:
    import pandas as pd
    import ray

    ray.init(address=a.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=1, memory=6 * 2**30)
    refs = []
    for group, table in INTERVAL_TABLES.items():
        for t in pd.read_csv(Path(a.repo_root) / table).to_dict("records"):
            if t["use"]:
                refs.append(remote(convert_one).remote(group, t, a.repo_root))
    p15 = [remote(powder15_one).remote(n, b, a.repo_root) for n in SIX for b in ("600_800", "2400_2600")]
    rows = ray.get(refs) + [r for chunk in ray.get(p15) for r in chunk]
    failed = [r for r in rows if not r["ok"]]
    print(f"[ncar-prep] {len(rows)} files, {len(failed)} failed", flush=True)
    for r in failed:
        print(json.dumps(r), flush=True)
    if a.output_key:
        put_json(a.output_key, rows)


def cmd_fingerprint(a: argparse.Namespace) -> None:
    rows = []
    for p in a.paths:
        try:
            rows.append({"relpath": p, **fingerprint(Path(a.repo_root) / p)})
        except Exception as error:
            rows.append({"relpath": p, "error": f"{type(error).__name__}: {error}"})
    print(json.dumps(rows))


def offline_check() -> dict[str, Any]:
    """Fresh process, network loads disabled: every backbone and MOMENT-1-small must come from the cache."""
    code = f"""
import json, sys
sys.path.insert(0, '/workspace/pylibs')
out = {{}}
from transformers import AutoConfig, AutoModel, AutoTokenizer
for name in {list(BACKBONES)!r}:
    try:
        m = AutoModel.from_pretrained(name); AutoTokenizer.from_pretrained(name)
        out[name] = sum(p.numel() for p in m.parameters())
    except Exception as e:
        out[name] = 'FAIL ' + type(e).__name__ + ': ' + str(e)[:200]
try:
    from momentfm import MOMENTPipeline
    m = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-small', model_kwargs={{'task_name': 'reconstruction'}})
    out['AutonLab/MOMENT-1-small'] = sum(p.numel() for p in m.parameters())
except Exception as e:
    out['AutonLab/MOMENT-1-small'] = 'FAIL ' + type(e).__name__ + ': ' + str(e)[:200]
import numpy, transformers, torch
out['numpy_file'] = numpy.__file__; out['numpy'] = numpy.__version__; out['transformers'] = transformers.__version__; out['torch'] = torch.__version__
try:
    import fsspec; out['fsspec'] = fsspec.__version__ + ' ' + fsspec.__file__
except Exception as e:
    out['fsspec'] = str(e)
print('RESULT ' + json.dumps(out))
"""
    env = {**os.environ, "HF_HOME": "/workspace/hf-cache", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=1800)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    return json.loads(line[7:]) if line else {"error": proc.stderr[-1500:]}


def setup_on_node(repo_root: str, bucket: str) -> dict[str, Any]:
    import shutil
    import socket

    os.chdir(repo_root)
    sys.path.insert(0, repo_root)
    from training.ray.llm_powder import HF_HOME, PYLIBS, use_isolated_libs
    from training.ray.tune_direct import setup_timeran_node

    row: dict[str, Any] = {"host": socket.gethostname(), "timeran": setup_timeran_node(bucket)}
    removed = []
    for leftover in list(Path(PYLIBS).glob("numpy*")):              # keep the image's NumPy 2.1.3
        shutil.rmtree(leftover, ignore_errors=True)
        removed.append(leftover.name)
    row["removed_from_pylibs"] = removed
    use_isolated_libs()
    os.environ["HF_HOME"] = HF_HOME
    from transformers import AutoModel, AutoTokenizer

    for name in BACKBONES:                                           # download into the cache
        AutoModel.from_pretrained(name)
        AutoTokenizer.from_pretrained(name)
    row["offline"] = offline_check()
    row["ok"] = bool(row["timeran"].get("ok")) and not any(str(v).startswith("FAIL") for v in row["offline"].values()) \
        and "error" not in row["offline"]
    return row


def node_strategy(node: str | None):
    """Scheduling strategy pinning a task to the live Ray node (GPU or CPU-only) whose IP or hostname contains ``node`` (None: no pinning)."""
    if not node:
        return None
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    live = [n for n in ray.nodes() if n.get("Alive")]                # GPU hosts and the x86 CPU workers (cpu-3/4, 11-12 23:20 UTC)
    hits = [n for n in live if node in n.get("NodeManagerAddress", "") or node in n.get("NodeManagerHostname", "") or node == n["NodeID"]]
    if len(hits) != 1:
        raise SystemExit(f"--node {node!r} matches {len(hits)} nodes: {[(n['NodeManagerAddress'], n.get('NodeManagerHostname')) for n in live]}")
    return NodeAffinitySchedulingStrategy(hits[0]["NodeID"], soft=False)


def cmd_setup(a: argparse.Namespace) -> None:
    import ray

    ray.init(address=a.address, ignore_reinit_error=True)
    bucket = os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    task = ray.remote(num_cpus=4, num_gpus=0)(setup_on_node)
    strategy = node_strategy(a.node)
    row = ray.get((task.options(scheduling_strategy=strategy) if strategy else task).remote(a.repo_root, bucket))
    print(json.dumps(row, indent=1, default=str), flush=True)
    if a.output_key:
        put_json(a.output_key, row)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("convert", "setup", "fingerprint"):
        s = sub.add_parser(name)
        s.add_argument("--repo-root", default="/workspace/spectrum-usage")
        s.add_argument("--address", default="auto")
        s.add_argument("--output-key", default=None)
        s.add_argument("--node", default=None, help="pin the setup to this node (IP or hostname substring); default: any node")
        if name == "fingerprint":
            s.add_argument("--paths", nargs="+", required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    {"convert": cmd_convert, "setup": cmd_setup, "fingerprint": cmd_fingerprint}[a.command](a)


if __name__ == "__main__":
    main()
