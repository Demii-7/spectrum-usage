"""Rescore checkpoints on one common test range (metrics only), reusing the benchmark's checkpoint discovery and config rebuilding.

Why: some result sets were scored on different windows. The tuned 15-minute finals used the preliminary range (to 2026-10-30T18:00Z)
and the 2400-2600 runs mix 10-30 and 11-03 ends. This job re-evaluates every checkpoint of the chosen sources on
2026-10-27T20:00Z to 2026-11-03T20:00Z and writes one record per checkpoint with all five metrics (MAE, MSE, RMSE, MAPE linear and dBm).

Sources (see ``inference_benchmark``): ``oct`` (October 1-minute 600-800; restrict with ``--models``), ``band`` (2400-2600 runs),
``p15`` and ``p15tuned`` (15-minute), ``direct`` (the OFA, Time-LLM, PatchTST, DLinear and residual-head finals of ``tune_direct``).
Every record also gets a per-bin table: the evaluator's per-frequency metrics (all five metrics per bin and horizon) for the test
split are uploaded as ``<output-prefix>/per-bin/<same path>.csv`` and linked from the record as ``per_bin_key``. Stride: 1 everywhere except the 1-minute 1D LSTMs, which score one bin at a time (about
800 windows per time step) and are thinned by ``--thin-1d-lstm`` (default 10; disclosed in every record).

    python -m training.ray.rescore --sources p15tuned
    python -m training.ray.rescore --sources band
    python -m training.ray.rescore --sources oct --models convlstm residualconvlstm dswinlstm_i vanillalstm1d residualvanillalstm
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

RANGE = ("2026-10-27T20:00:00Z", "2026-11-03T20:00:00Z")
LINEAR = {"linearar1d", "linearar2d", "linearar4d", "residuallinearar1d", "residuallinearar2d", "residuallinearar4d"}
LSTM_1D = {"vanillalstm1d", "residualvanillalstm"}


def stride_for(entry: dict[str, Any], thin_1d_lstm: int = 10, thin_direct_1min: int = 1) -> int:
    """1 everywhere; the 1-minute 1D LSTMs are thinned (one bin per window makes stride 1 about 30 million windows).

    The 1-minute ``direct`` finals were first scored on every 15th test window (``tune_direct`` thinned them); the default here, 1,
    re-scores them on every window from their saved weights. ``thin_direct_1min`` restores the old thinning if ever needed.
    """
    one_minute = entry["source"] in ("oct", "band") or entry.get("resolution") == "1min"
    if entry["source"] == "direct" and entry.get("resolution") == "1min":
        return thin_direct_1min
    return thin_1d_lstm if (one_minute and entry["rep"] == "1d" and entry["model"] in LSTM_1D) else 1


def memory_gb(entry: dict[str, Any]) -> float:
    """Memory to reserve for one task, so Ray never co-schedules more than a node holds (an unreserved batch of 4D and 1-minute 1D
    tasks took a 250 GB node past the 95% kill threshold on 2026-11-08 and killed every job driver on it). 4D 1-minute windows alone
    are about 48 GB at stride 1; the 1-minute 1D windows about 8 GB per 30 million series, plus predictions."""
    one_minute = entry["source"] in ("oct", "band") or entry.get("resolution") == "1min"
    if entry["rep"] == "4d" and one_minute:
        return 110.0
    if entry["rep"] == "2d" and one_minute:
        return 12.0                                  # about 40,000 windows x 60 steps x 200 bins at stride 1, plus forecasts
    if entry["rep"] == "1d" and one_minute:
        return 45.0 if entry["model"] in LINEAR else 20.0
    return 6.0


def record_key(prefix: str, entry: dict[str, Any]) -> str:
    horizon = f"/h{entry['horizon']}" if entry.get("horizon") else ""
    source = entry["prefix"].replace("/", "_") if entry["source"] in ("oct",) else entry["source"]
    resolution = f"/{entry['resolution']}" if entry.get("resolution") else ""
    return (f"{prefix.strip('/')}/{source}{resolution}/{entry.get('band') or ''}/{entry['rep']}/{entry['model']}{horizon}/seed-{entry['seed']}.json")


def per_bin_key(prefix: str, entry: dict[str, Any]) -> str:
    """Where the per-bin table of ``record_key``'s record goes."""
    key = record_key(prefix, entry)
    root = prefix.strip("/")
    return f"{root}/per-bin/{key[len(root) + 1:].removesuffix('.json')}.csv"


def rescore_entry(entry: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: download one checkpoint, evaluate it on the common range, return a record."""
    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    from training.ray.llm_powder import use_isolated_libs

    use_isolated_libs()                  # transformers lives in /workspace/pylibs on workers (OFA / Time-LLM checkpoints need it)
    import pandas as pd

    from training.common import evaluation_integrated as ei
    from training.ray.final_eval import summarize_metrics
    from training.ray.inference_benchmark import build_config
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    record: dict[str, Any] = {k: entry.get(k) for k in ("source", "prefix", "band", "rep", "model", "horizon", "seed", "pt_key")}
    stride = stride_for(entry, args["thin_1d_lstm"], args["thin_direct_1min"])
    record.update(test_start=args["test_start"], test_end=args["test_end"], test_stride=stride)
    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="rescore-") as scratch_dir:
            scratch = Path(scratch_dir)
            run_args = {"repo_root": args["repo_root"], "test_start": args["test_start"], "test_end": args["test_end"],
                        "device": args["device"], "strides": {k: 1 for k in ("oct", "band", "p15", "p15tuned", "direct")}}
            config, checkpoint = build_config(entry, run_args, client, bucket, scratch)
            name = entry.get("pipeline", entry["model"])
            config["data"]["split"]["ranges"]["test"] = {"start": args["test_start"], "end": args["test_end"]}
            config[name].setdefault("train", {})["test_stride"] = int(stride)
            if args.get("horizons") and int(config[name]["model"].get("prediction_horizon", 1)) == 1:
                # one-step models are rolled out, so any horizon can be scored (direct models only have the one they were trained for)
                config["windowing"]["horizons"] = sorted({int(h) for h in args["horizons"]})
                record["horizons"] = config["windowing"]["horizons"]
            out_dir = scratch / "eval"
            out_dir.mkdir()
            ei.evaluate_one_model(config, name, out_dir, checkpoint, skip_plots=True)
            rows = pd.read_csv(out_dir / "aggregate_metrics.csv").to_dict("records")
            record.update(summarize_metrics(rows))
            from training.ray.final_eval import per_bin_test_rows

            table = per_bin_test_rows(pd.read_csv(out_dir / "per_frequency_metrics.csv"),
                                      {"source": entry["source"], "band": entry.get("band"), "rep": entry["rep"], "model": entry["model"],
                                       "trained_horizon": entry.get("horizon"), "seed": entry["seed"], "test_stride": stride})
            buffer = io.StringIO()
            table.to_csv(buffer, index=False)
            record["per_bin_key"] = per_bin_key(args["output_prefix"], entry)
            client.put_object(Bucket=bucket, Key=record["per_bin_key"], Body=buffer.getvalue().encode())
            record["n_test_windows"] = int(max((r.get("n_targets") or 0) for r in rows if str(r["split"]).endswith("test")) or 0)
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-800:])
    record["wall_seconds"] = round(time.time() - started, 1)
    try:
        client.put_object(Bucket=bucket, Key=record_key(args["output_prefix"], entry), Body=json.dumps(record, default=str).encode())
    except Exception:
        pass
    return record


def numpy_major(_=None) -> int:
    import numpy

    return int(numpy.__version__.split(".")[0])


def usable_nodes(ray) -> list[dict[str, Any]]:
    """Alive nodes that can run tasks, keeping only those whose numpy major version is the most common one.

    Checkpoints are pickled with the writer's numpy; a worker with an older numpy (seen on one A100 host: 1.25 vs 2.1) cannot load
    checkpoints written under numpy 2 (``No module named 'numpy._core'``), so tasks are pinned to the consistent nodes.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("CPU", 0) >= 4]
    probes = {n["NodeID"]: ray.remote(num_cpus=1, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False),
                                      max_retries=0)(numpy_major).remote() for n in nodes}
    majors = {node_id: ray.get(ref) for node_id, ref in probes.items()}
    common = max(set(majors.values()), key=list(majors.values()).count)
    keep = [n for n in nodes if majors[n["NodeID"]] == common]
    dropped = len(nodes) - len(keep)
    if dropped:
        print(f"[rescore] avoiding {dropped} node(s) with a different numpy major version (using {common})", flush=True)
    return keep


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True, choices=["oct", "band", "p15", "p15tuned", "direct"])
    p.add_argument("--models", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--test-start", default=RANGE[0])
    p.add_argument("--test-end", default=RANGE[1])
    p.add_argument("--thin-1d-lstm", type=int, default=10)
    p.add_argument("--horizons", type=int, nargs="+", help="override the windowing horizons for one-step (rolled-out) models, "
                   "for example 1 4 12 24 48 to add t+15 minutes at 15-minute resolution; direct models keep their own")
    p.add_argument("--thin-direct-1min", type=int, default=1, help="test-window stride for the 1-minute direct finals (1 = every window)")
    p.add_argument("--resolution", choices=("1min", "15min"), help="only entries of this resolution (direct source)")
    p.add_argument("--gpu-fraction", type=float, default=0.25)
    p.add_argument("--device", default=None)
    p.add_argument("--skip-done", action="store_true", help="skip checkpoints that already have an ok record")
    p.add_argument("--no-pin", action="store_true",
                   help="let Ray place tasks anywhere with free resources (node pinning, even soft, waits on a busy pinned node; "
                        "only needed when a node has an incompatible numpy)")
    p.add_argument("--max-parallel", type=int, default=6, help="most tasks in flight at once (memory safety)")
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="rescored")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    entries = discover_sources(client, bucket, args.sources, args.models)
    if args.seeds:
        entries = [e for e in entries if e["seed"] in args.seeds]
    if args.resolution:
        entries = [e for e in entries if e.get("resolution") == args.resolution]
    if args.skip_done:
        def done(entry):
            try:
                return bool(json.loads(client.get_object(Bucket=bucket, Key=record_key(args.output_prefix, entry))["Body"].read()).get("ok"))
            except Exception:
                return False
        entries = [e for e in entries if not done(e)]
    print(f"[rescore] {len(entries)} checkpoints from {args.sources}", flush=True)
    ray.init(address=args.address, ignore_reinit_error=True)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "test_start": args.test_start, "test_end": args.test_end,
                   "thin_1d_lstm": args.thin_1d_lstm, "thin_direct_1min": args.thin_direct_1min, "device": args.device, "output_prefix": args.output_prefix,
                   "horizons": args.horizons}
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    nodes = usable_nodes(ray)
    gpu_nodes = [n for n in nodes if n.get("Resources", {}).get("GPU", 0) > 0] or nodes
    def submit(index: int, entry: dict[str, Any]):
        gpus = 0 if entry["model"] in LINEAR else args.gpu_fraction
        pool = gpu_nodes if gpus else nodes
        options = {"num_cpus": 4, "num_gpus": gpus, "memory": int(memory_gb(entry) * 2 ** 30), "max_retries": 1}
        if not args.no_pin:
            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(node_id=pool[index % len(pool)]["NodeID"], soft=True)
        return ray.remote(**options)(rescore_entry).remote(entry, worker_args)

    # heaviest first, at most --max-parallel in flight, so memory use stays bounded and the long tasks do not end up last
    order = sorted(enumerate(entries), key=lambda ie: -memory_gb(ie[1]))
    queue = list(order)
    rows: list[dict[str, Any]] = []
    pending: list[Any] = [submit(*queue.pop(0)) for _ in range(min(args.max_parallel, len(queue)))]
    remaining = len(queue) + len(pending)
    while pending:
        done_refs, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done_refs[0])
        rows.append(r)
        remaining -= 1
        if queue:
            pending.append(submit(*queue.pop(0)))
        print(f"[rescore] {r['source']} {r.get('band')} {r['rep']}/{r['model']} h={r.get('horizon')} seed{r['seed']} ok={r['ok']} "
              f"mae={r.get('test_mean_horizon_mae_db')} {r.get('error', '')} ({r['wall_seconds']}s) [{remaining} left]", flush=True)
    import pandas as pd

    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    key = f"{args.output_prefix.strip('/')}/summary_{'-'.join(args.sources)}_{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}.csv"
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
    print(f"[rescore] wrote s3://{bucket}/{key} ({len(rows)} rows, {sum(not r['ok'] for r in rows)} failed)", flush=True)


if __name__ == "__main__":
    main()
