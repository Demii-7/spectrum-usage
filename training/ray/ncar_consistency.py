"""Step 0c of the NCAR transfer batch: rescore stored cells on NCAR (PyTorch 2.6, ARM) and compare with the stored values (PyTorch 2.5.1, x86).

``transfer`` part: one stored ``transfer_all``/``transfer15`` cell per model (POWDER guesthouse, October test week, seed 40 unless the
model lacks it), rescored with the unchanged ``transfer_all.job`` into ``transfer-ncar-2026-11/step0/consistency/`` and compared per horizon
on MAE, MSE, RMSE and MAPE (linear). Rolled-out 1D/2D models are compared in regime A (stored by ``transfer15``: ``transfer15-oct2026``) and
regime C (stored by ``transfer_all``: ``transfer-all-oct2026``). Pass: |MAE difference| <= 1e-3 dB.

    python -m training.ray.ncar_consistency transfer --out-key transfer-ncar-2026-11/step0/consistency_transfer.csv
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Any

PREFIX = "transfer-ncar-2026-11/step0/consistency"
TARGET = {"platform": "powder", "site": "guesthouse", "run": "20261025T1355Z", "band": "600_800", "n_origins": 625}
# (model, rep, source, regime, seed)
CELLS = [("revinmlp", "2d", "direct", "A", 40), ("residuallinearar2d", "2d", "direct", "A", 40),
         ("autoformer_csa", "2d", "rolled", "A", 40), ("autoformer_csa", "2d", "rolled", "C", 40),
         ("lstmattn", "2d", "rolled", "A", 40), ("lstmattn", "2d", "rolled", "C", 40),
         ("residuallinearar2d", "2d", "rolled", "A", 40),
         ("ofa", "2d", "direct", "A", 40), ("timellm", "2d", "direct", "A", 40), ("moment_lp", "2d", "direct", "A", 40),
         ("timeran_lp", "2d", "direct", "A", 40)]
METRICS = {"mae_db": "mae", "mse_db2": "mse", "rmse_db": "rmse", "mape_linear_pct": "mape_linear"}


def stored_key(model: str, rep: str, source: str, regime: str, seed: int, horizon: int | None) -> str:
    t = TARGET
    if source == "rolled" and regime == "A":
        return f"transfer15-oct2026/{t['band']}/{t['site']}/{rep}/{model}/seed-{seed}/summary.json"
    h = f"/h{horizon}" if horizon else ""
    return f"transfer-all-oct2026/{regime}/{t['platform']}/{t['site']}/{t['run']}/{t['band']}/{rep}/{model}{h}/seed-{seed}/summary.json"


def cmd_transfer(a: argparse.Namespace) -> None:
    import pandas as pd
    import ray

    from training.ray import transfer_all as ta
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    cells = [c for c in CELLS if not a.models or c[0] in a.models]
    specs = []
    for model, rep, source, regime, seed in cells:
        kind = "p15tuned" if source == "rolled" else "direct"
        entries = [e for e in discover_sources(client, bucket, [kind], [model]) if e["band"] == TARGET["band"] and e["rep"] == rep
                   and int(e["seed"]) == seed and (source == "rolled" or (e["resolution"] == "15min" and int(e["horizon"]) in a.horizons))]
        entries = sorted(entries, key=lambda e: int(e.get("horizon") or 0))
        if not entries:
            print(f"[consistency] no checkpoint for {model} {rep} {source} seed {seed}", flush=True)
            continue
        specs.append(({"target": TARGET, "entries": entries, "regimes": [regime]}, (model, rep, source, regime, seed)))
    shared = {"repo_root": a.repo_root, "bucket": bucket, "prefix": f"{PREFIX}/{source_tag(a)}", "code_revision": a.code_revision or "ncar",
              "threads": 4, "save_arrays": False}
    ray.init(address=a.address, ignore_reinit_error=True)
    refs = []
    for spec, cell in specs:
        gpu = ta.GPU_FRACTION.get(cell[0], 0.0)
        refs.append((ray.remote(num_cpus=4, num_gpus=gpu, memory=int(24 * 2 ** 30), max_retries=0)(ta.job).remote(spec, shared), spec, cell))
    rows = []
    for ref, spec, (model, rep, source, regime, seed) in refs:
        record = ray.get(ref)
        print(f"[consistency] {model} {source} {regime} ok={record.get('ok')} {record.get('error', '')} ({record.get('wall_seconds')}s)", flush=True)
        for e in spec["entries"]:
            horizon = int(e["horizon"]) if source == "direct" else None
            new_key = ta.output_base(shared["prefix"], regime, TARGET, rep, model, seed, horizon) + "/summary.json"
            try:
                new = json.loads(client.get_object(Bucket=bucket, Key=new_key)["Body"].read())["metrics"]
                old = json.loads(client.get_object(Bucket=bucket, Key=stored_key(model, rep, source, regime, seed, horizon))["Body"].read())["metrics"]
            except Exception as error:
                rows.append({"model": model, "source": source, "cell": f"{regime} {ta.target_key(TARGET)} seed {seed} h{horizon}", "metric": "load",
                             "stored": None, "ncar": None, "diff": None, "error": f"{type(error).__name__}: {error}"[:200]})
                continue
            for h in sorted(new, key=int):
                if h not in old:
                    continue
                for m, short in METRICS.items():
                    if m in new[h] and m in old[h]:
                        rows.append({"model": model, "source": source,
                                     "cell": f"{regime} {ta.target_key(TARGET)} seed {seed}" + (f" trained h{horizon}" if horizon else "") + f" t+{h}",
                                     "metric": short, "stored": old[h][m], "ncar": new[h][m], "diff": new[h][m] - old[h][m]})
    table = pd.DataFrame(rows)
    print(table.to_string(), flush=True)
    if a.out_key:
        client.put_object(Bucket=bucket, Key=a.out_key, Body=table.to_csv(index=False).encode())


def source_tag(a) -> str:
    return a.tag


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("transfer")
    t.add_argument("--models", nargs="+")
    t.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24, 48])
    t.add_argument("--tag", default="run1", help="sub-prefix for this attempt (never reused)")
    t.add_argument("--out-key", default=None)
    t.add_argument("--code-revision", default=None)
    t.add_argument("--address", default="auto")
    t.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    os.chdir(a.repo_root)
    {"transfer": cmd_transfer}[a.command](a)


if __name__ == "__main__":
    main()
