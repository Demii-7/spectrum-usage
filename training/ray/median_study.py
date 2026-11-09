"""CPU-only study of the validation-tuned median baseline and of median-anchored / MAE-loss ResidualLinearAR on POWDER.

Part A (``sweep``): a parameter-free baseline sweep. For each (resolution, band, statistic) the statistic (median or mean) of the newest
K window steps is scored on the VALIDATION range and on the final TEST range for every K in a grid, through the repository's own
data pipeline and evaluator, so the windows, normalization, pooling over the four core receivers, and the autoregressive roll-out of
one-step models are exactly those of the model tables (``lookbackmean2d`` with ``statistic`` / ``window``). K is selected on validation
only: one K for all horizons (the roll-out convention of our tables) and one K per horizon (a post-hoc pick from the same sweep).
The per-bin median is the same for 1D and 2D inputs, so the sweep uses the 2D pipeline.

Part B (``variants``): ResidualLinearAR 1D and 2D with the winner's own hyperparameters (the October 1-minute seed-42 winners; the tuned
15-minute finals), retrained here under one protocol so the comparison is like for like:

    ref         anchor = lookback mean, MSE loss            (the model as in our tables, retrained)
    mae         anchor = lookback mean, MAE loss            (isolates the loss)
    median      anchor = median of the newest K steps, MSE  (isolates the anchor; K from part A, validation-selected)
    median_mae  both

Disclosed deviations: the 1-minute 1D models score every ``--thin-1d-1min``-th test window (default 10; a stride-1 1D evaluation needs
about 45 GB of host memory) and every variant, the reference included, uses the same stride. Hyperparameters are inherited, not re-tuned.

All records, weights and per-bin tables go to the durable bucket under ``<prefix>/``. Tasks are CPU only and reserve memory.

    python -m training.ray.median_study --max-parallel 4
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

BANDS = ("600_800", "2400_2600")
GRID = {"1min": (5, 10, 15, 20, 30, 45, 60), "15min": (1, 2, 3, 4, 6, 8, 12, 24, 48, 96)}
HORIZONS = {"1min": (1, 15, 60), "15min": (4, 12, 24, 48)}
VALIDATION = {"start": "2026-10-25T13:55:00Z", "end": "2026-10-27T13:55:00Z"}
TEST = {"start": "2026-10-27T20:00:00Z", "end": "2026-11-03T20:00:00Z"}
VARIANTS = {"ref": ("mean", "mse"), "mae": ("mean", "mae"), "median": ("median", "mse"), "median_mae": ("median", "mae")}
SEEDS = (40, 41, 42)
MODELS = {"1d": "residuallinearar1d", "2d": "residuallinearar2d"}


def memory_gb(resolution: str, rep: str) -> float:
    """Host memory reserved per task (the pipeline keeps all windows of a range in memory)."""
    if resolution == "15min":
        return 8.0
    return 30.0 if rep == "1d" else 14.0


def pick_k(rows: list[dict[str, Any]], statistic: str, horizons, per_horizon: bool = False):
    """K with the lowest validation MAE: one K (mean over horizons) or a dict horizon -> K."""
    table = [r for r in rows if r["statistic"] == statistic and r["range"] == "validation"]
    if not table:
        raise ValueError(f"no validation rows for {statistic}")
    if per_horizon:
        return {h: min(table, key=lambda r: (r[f"mae_t{h}"], r["k"]))["k"] for h in horizons}
    return min(table, key=lambda r: (sum(r[f"mae_t{h}"] for h in horizons) / len(horizons), r["k"]))["k"]


def sweep_config(resolution: str, band: str, root: str) -> tuple[dict[str, Any], str]:
    """Pipeline config of the parameter-free lookback model for this resolution and band (range-agnostic)."""
    import copy

    import yaml

    from training.ray.final_eval import config_file_for
    from training.ray.fixed_train import swap_band

    model = "lookbackmean2d"
    config = yaml.safe_load(Path(config_file_for(model, "2d")).read_text())
    config["training"] = {**config.get("training", {}), "models": [model], "device": "cpu"}
    config["training"].pop("model_name", None)
    if resolution == "15min":
        from training.ray.powder15 import apply_powder15, set_lookback_and_horizon

        config = apply_powder15(config, band, root, list(HORIZONS[resolution]))
        config = set_lookback_and_horizon(config, model, None)
    else:
        config = swap_band(config, band)
        config["windowing"] = {"lookback": 60, "horizons": list(HORIZONS[resolution])}
    config[model]["model"].update(prediction_horizon=1, input_sequence_length=int(config["windowing"]["lookback"]))
    config[model]["train"] = {**config[model].get("train", {}), "test_stride": 1}
    return copy.deepcopy(config), model


def sweep_job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one (resolution, band, statistic) sweep over K on validation and test."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    import copy

    import pandas as pd

    from training.common.evaluation_integrated import evaluate_one_model
    from training.common.train_integrated import train_one_model
    from training.ray.final_eval import summarize_metrics
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    resolution, band, statistic = spec["resolution"], spec["band"], spec["statistic"]
    record: dict[str, Any] = {"part": "sweep", "resolution": resolution, "band": band, "statistic": statistic}
    started, rows = time.time(), []
    try:
        if resolution == "15min":
            from training.ray.powder15 import ensure_powder15

            ensure_powder15(band, root)
        config, model = sweep_config(resolution, band, root)
        lookback = int(config["windowing"]["lookback"])
        with tempfile.TemporaryDirectory(prefix="median-sweep-") as scratch:
            scratch = Path(scratch)
            config["data"]["split"]["ranges"]["test"] = dict(TEST)
            train_one_model(config, model, scratch / "run")
            checkpoint = next((scratch / "run" / "checkpoints").glob("*.pt"))
            horizons = HORIZONS[resolution]
            for k in sorted(set(GRID[resolution]) | {lookback}):
                for range_name, window in (("validation", VALIDATION), ("test", TEST)):
                    trial = copy.deepcopy(config)
                    trial["data"]["split"]["ranges"]["test"] = dict(window)
                    trial[model]["model"].update(statistic=statistic, window=int(k))
                    out = scratch / f"eval-{statistic}-{k}-{range_name}"
                    out.mkdir()
                    evaluate_one_model(trial, model, out, checkpoint, skip_plots=True)
                    summary = summarize_metrics(pd.read_csv(out / "aggregate_metrics.csv").to_dict("records"))
                    # every metric (MAE, MSE, RMSE, both MAPEs; per horizon and mean) is kept, not only the MAE used for selection
                    rows.append({"statistic": statistic, "k": int(k), "range": range_name,
                                 **{f"mae_t{h}": summary[f"test_mae_db_t{h}"] for h in horizons},
                                 "mae_mean": summary["test_mean_horizon_mae_db"], "metrics": summary})
                    if range_name == "test":
                        from training.ray.final_eval import per_bin_test_rows

                        table = per_bin_test_rows(pd.read_csv(out / "per_frequency_metrics.csv"),
                                                  {"resolution": resolution, "band": band, "statistic": statistic, "k": int(k)})
                        buffer = io.StringIO()
                        table.to_csv(buffer, index=False)
                        client.put_object(Bucket=bucket, Key=f"{args['prefix'].strip('/')}/sweep/{resolution}/{band}/per-bin/"
                                                             f"{statistic}_k{int(k)}.csv", Body=buffer.getvalue().encode())
        record.update(rows=rows, lookback=lookback, ok=True)
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1000:], rows=rows)
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{args['prefix'].strip('/')}/sweep/{resolution}/{band}/{statistic}.json",
                      Body=json.dumps(record, default=str).encode())
    return record


def variant_job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: retrain one ResidualLinearAR variant on one (resolution, band, representation) and score it on the test range."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    import pandas as pd

    from training.common.evaluation_integrated import evaluate_one_model
    from training.common.train_integrated import train_one_model
    from training.ray.artifacts import save_run
    from training.ray.final_eval import per_bin_test_rows, summarize_metrics
    from training.ray.inference_benchmark import build_config
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    entry, variant, seed = spec["entry"], spec["variant"], int(spec["seed"])
    resolution, band, rep = spec["resolution"], entry["band"], entry["rep"]
    anchor, loss = VARIANTS[variant]
    base = f"{args['prefix'].strip('/')}/variants/{resolution}/{band}/{rep}/{variant}"
    record: dict[str, Any] = {"part": "variants", "resolution": resolution, "band": band, "rep": rep, "variant": variant, "seed": seed,
                              "anchor": anchor, "loss": loss, "k": spec["k"] if anchor == "median" else None,
                              "source": entry["source"], "hyperparameters": "inherited from the existing winner"}
    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="median-variant-") as scratch:
            scratch = Path(scratch)
            run_args = {"repo_root": root, "test_start": TEST["start"], "test_end": TEST["end"], "device": "cpu",
                        "strides": {key: 1 for key in ("oct", "band", "p15", "p15tuned", "direct")}}
            config, _ = build_config(entry, run_args, client, bucket, scratch)
            name = entry.get("pipeline", entry["model"])
            stride = args["thin_1d_1min"] if (resolution == "1min" and rep == "1d") else 1
            config["data"]["split"]["ranges"]["test"] = dict(TEST)
            config["training"] = {**config.get("training", {}), "models": [name], "device": "cpu"}
            config[name]["train"].update(seed=seed, loss=loss, test_stride=int(stride))
            model_cfg = config[name]["model"]
            model_cfg.pop("baseline", None)
            if anchor == "median":
                model_cfg.update(baseline="median", baseline_window=int(spec["k"]))
            record.update(test_stride=int(stride), model_config=model_cfg, train_config=config[name]["train"])
            run_dir, eval_dir = scratch / "run", scratch / "eval"
            train_one_model(config, name, run_dir)
            checkpoint = next((run_dir / "checkpoints").glob("*.pt"))
            eval_dir.mkdir()
            evaluate_one_model(config, name, eval_dir, checkpoint, skip_plots=True)
            rows = pd.read_csv(eval_dir / "aggregate_metrics.csv").to_dict("records")
            record.update(summarize_metrics(rows))
            table = per_bin_test_rows(pd.read_csv(eval_dir / "per_frequency_metrics.csv"),
                                      {"resolution": resolution, "band": band, "rep": rep, "variant": variant, "seed": seed})
            buffer = io.StringIO()
            table.to_csv(buffer, index=False)
            record["per_bin_key"] = f"{base}/per-bin/seed-{seed}.csv"
            client.put_object(Bucket=bucket, Key=record["per_bin_key"], Body=buffer.getvalue().encode())
            record.update(save_run(client, bucket, f"{base}/weights/seed-{seed}", run_dir, config,
                                   {"result_kind": "median_study", "variant": variant, "seed": seed, "k": record["k"]}))
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/seed-{seed}.json", Body=json.dumps(record, default=str).encode())
    return record


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolutions", nargs="+", default=["1min", "15min"], choices=("1min", "15min"))
    p.add_argument("--bands", nargs="+", default=list(BANDS), choices=BANDS)
    p.add_argument("--reps", nargs="+", default=["2d", "1d"], choices=("1d", "2d"))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--seeds-1d", type=int, nargs="+", default=[42], help="1D runs take 10-60+ minutes each; one seed (linear models show no seed spread)")
    p.add_argument("--thin-1d-1min", type=int, default=10, help="test-window stride for the 1-minute 1D models (memory)")
    p.add_argument("--max-parallel", type=int, default=4, help="most tasks in flight at once (host memory)")
    p.add_argument("--skip-sweep", action="store_true", help="reuse finished sweep records instead of recomputing")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--sweep-only", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="median-study")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.inference_benchmark import discover_sources, pick_checkpoints
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    ray.init(address=args.address, ignore_reinit_error=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "thin_1d_1min": args.thin_1d_1min}

    def submit(function, spec, gb):
        # No per-task node pinning: a hard pin made one task wait for a busy node while others were free. Memory is reserved instead.
        return ray.remote(num_cpus=4, num_gpus=0, memory=int(gb * 2**30), max_retries=1)(function).remote(spec, shared)

    def run_queue(items, label):
        queue, pending, results = list(items), [], []
        while queue or pending:
            while queue and len(pending) < args.max_parallel:
                function, spec, gb = queue.pop(0)
                pending.append(submit(function, spec, gb))
            done, pending = ray.wait(pending, num_returns=1)
            try:
                row = ray.get(done[0])
            except Exception as error:
                row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
            results.append(row)
            print(f"[median-study] {label} {row.get('resolution')} {row.get('band')} {row.get('rep', '')} {row.get('statistic', '')}"
                  f"{row.get('variant', '')} seed{row.get('seed', '')} ok={row.get('ok')} mae={row.get('test_mean_horizon_mae_db', '')} "
                  f"{row.get('error', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
        return results

    def load(key):
        try:
            return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
        except Exception:
            return None

    sweep_items, sweeps = [], {}
    for resolution in args.resolutions:
        for band in args.bands:
            for statistic in ("median", "mean"):
                key = f"{args.prefix.strip('/')}/sweep/{resolution}/{band}/{statistic}.json"
                done = load(key) if (args.skip_sweep or args.skip_done) else None
                if done and done.get("ok"):
                    sweeps[(resolution, band, statistic)] = done
                else:
                    sweep_items.append((sweep_job, {"resolution": resolution, "band": band, "statistic": statistic},
                                        memory_gb(resolution, "2d")))
    print(f"[median-study] {len(sweep_items)} sweeps to run", flush=True)
    for record in run_queue(sweep_items, "sweep"):
        if record.get("ok"):
            sweeps[(record["resolution"], record["band"], record["statistic"])] = record
    selected = {}
    for (resolution, band, statistic), record in sorted(sweeps.items()):
        if statistic == "median":
            horizons = HORIZONS[resolution]
            selected[(resolution, band)] = {"k": pick_k(record["rows"], "median", horizons),
                                            "k_per_horizon": pick_k(record["rows"], "median", horizons, per_horizon=True)}
            print(f"[median-study] selected K {resolution} {band}: {selected[(resolution, band)]}", flush=True)
    client.put_object(Bucket=bucket, Key=f"{args.prefix.strip('/')}/selected_k.json",
                      Body=json.dumps({f"{r}/{b}": v for (r, b), v in selected.items()}, default=str).encode())
    if args.sweep_only:
        return

    entries = pick_checkpoints(discover_sources(client, bucket, ["oct", "band", "p15tuned"], list(MODELS.values())))
    items = []
    for entry in entries:
        resolution = "15min" if entry["source"] == "p15tuned" else "1min"
        if resolution not in args.resolutions or entry["rep"] not in args.reps or entry["band"] not in args.bands:
            continue
        if (resolution, entry["band"]) not in selected:
            print(f"[median-study] no selected K for {resolution} {entry['band']}; skipping its variants", flush=True)
            continue
        for variant in args.variants:
            for seed in (args.seeds_1d if entry["rep"] == "1d" else args.seeds):
                spec = {"entry": entry, "variant": variant, "seed": seed, "resolution": resolution,
                        "k": selected[(resolution, entry["band"])]["k"]}
                base = f"{args.prefix.strip('/')}/variants/{resolution}/{entry['band']}/{entry['rep']}/{variant}/seed-{seed}.json"
                if args.skip_done and (load(base) or {}).get("ok"):
                    continue
                items.append((variant_job, spec, memory_gb(resolution, entry["rep"])))
    items.sort(key=lambda item: -item[2])                                   # heaviest first
    print(f"[median-study] {len(items)} variant runs", flush=True)
    results = run_queue(items, "variant")
    import pandas as pd

    buffer = io.StringIO()
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in results]).to_csv(buffer, index=False)
    key = f"{args.prefix.strip('/')}/summary_{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}.csv"
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
    print(f"[median-study] wrote s3://{bucket}/{key}; {sum(not r.get('ok') for r in results)} failed of {len(results)}", flush=True)


if __name__ == "__main__":
    main()
