"""Constant-offset response test on the 1-MINUTE checkpoints (evaluation only: forward passes, nothing trained or fitted, targets unused).

Same definition as ``offset_response.py`` (15 minutes): every input window of a frozen checkpoint is shifted by a constant ``c`` dB
(c = -20, -10, -5, +5, +10, +20) before the usual frozen pipeline (the checkpoint's own source normalization, the shared autoregressive
forecaster for one-step models), the forecast is compared with the forecast of the unshifted windows on the SAME origins, and

    shift(c, h) = mean over origins and bins of  forecast_c - forecast_0        (dB, at horizon h)
    gain(c, h)  = shift(c, h) / c

1 = the level is carried exactly, 0 = ignored. The 1-minute models are evaluated through the benchmark's own evaluation path
(``evaluation_integrated``: ``load_chunk``, ``build_model``, ``load_checkpoint_into_model``, ``make_window_starts``, ``forecast``), so
the checkpoint, its normalization and the window layout are exactly the ones used for the 1-minute results; the shift is applied to
the normalized windows as ``c / std`` per frequency bin (identical to shifting the raw dB values before normalization, because
normalization is per-bin affine).

* data: the October 1-minute test week (2026-10-27 20:00 to 11-03 20:00 UTC), the four core receivers pooled in ONE job per checkpoint (the
  1-minute chunk is the four receivers' files concatenated, with segment boundaries): the gain is the mean over all origins of all four
  receivers (a per-receiver split is not recorded);
* origins are thinned (disclosed in every record): test_stride 20 for 2D models (about 500 origins per receiver of about 9,900) and 100
  for 1D models (one series per receiver and bin, about 99 windows per series); every offset uses the identical origins;
* models: (a) the rolled-out one-step 1D/2D checkpoints of ``checkpoints-oct2026``, ``checkpoints-oct2026-r2`` (600-800) and
  ``oct2026-band-runs`` (2400-2600), horizons 1, 15, 60 minutes, lookback 60; (b) the 1-minute direct finals
  (``direct-tuning/1min/<band>/<model>/weights/h<h>/seed-N``) at their trained horizon.

Output: ``<prefix>/<band>/<rep>/<model>[__r2]/[h<h>/]seed-N/summary.json`` (one row per offset and horizon); the table script is
``evaluation/analysis/offset_response_table.py --prefix offset-response-1min-oct2026/``.

    python -m training.ray.offset_response_1min --bands 600_800 2400_2600 --skip-done
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

OFFSETS = (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0)
STRIDE = {"2d": 20, "1d": 100}
RANGE = ("2026-10-27T20:00:00Z", "2026-11-03T20:00:00Z")
ONE_STEP_MODELS = ("linearar1d", "residuallinearar1d", "vanillalstm1d", "residualvanillalstm", "linearar2d", "residuallinearar2d", "vanillalstm", "lstmattn",
                   "temporalconvnet", "autoformer_csa")
DIRECT_MODELS = ("revinmlp", "patchtst", "nlinear", "dlinear", "residuallinearar2d", "linearar2d", "residualnonlinear2d", "deepspred2d")
NEURAL = {"vanillalstm1d", "residualvanillalstm", "vanillalstm", "lstmattn", "temporalconvnet", "autoformer_csa"}
INTRINSIC = {"revinmlp": "full", "patchtst": "full", "nlinear": "last value subtracted", "dlinear": "decomposition (moving-average trend)",
             "residuallinearar1d": "partial", "residuallinearar2d": "partial", "residualvanillalstm": "partial", "residualnonlinear2d": "partial",
             "deepspred2d": "none (encoder-decoder skip connections; inputs clipped at the rendered z range)"}


def resources_for(model: str, direct: bool) -> float:
    """GPU share of one task: the small neural 1-minute models use a small share (they are a few hundred MB), everything else runs on CPU."""
    if model == "deepspred2d":                                   # a 3D Swin on colour frames: a quarter GPU, as in its tuning
        return 0.25
    return 0.125 if (model in NEURAL and not direct) else 0.0


def forward_offsets(config: dict[str, Any], name: str, checkpoint: Path, offsets: tuple[float, ...], horizons_out=None) -> dict[str, Any]:
    """Run the frozen checkpoint on the reference windows and on every shifted copy; returns rows and bookkeeping."""
    import torch

    from training.common.data import chunk_specs, load_chunk
    from training.common.evaluation_integrated import _flattened_1d_normalization, calculate_errors_and_export_arrays, evaluation_parameters
    from training.common.forecasting import forecast
    from training.common.model_factory import build_model, load_checkpoint_into_model
    from training.common.runtime import device_for
    from training.common.windowing import make_window_batch_array, make_window_starts, raw_targets_to_model_layout, to_model_layout
    from training.ray.offset_response import response_rows

    parameters = evaluation_parameters(config, name)
    lookback, horizons, max_horizon = parameters["lookback"], parameters["horizons"], parameters["max_horizon"]
    collected: dict[float, dict[int, list[np.ndarray]]] = {}
    n_windows, n_values = 0, 0
    all_offsets = (0.0,) + tuple(offsets)
    for chunk in chunk_specs(config):
        data = load_chunk(config, chunk, val_fraction=parameters["val_fraction"])
        model = build_model(model_name=name, config=config, train_data=data.splits[data.train_split].model_input)
        device = device_for(config)
        model, _ = load_checkpoint_into_model(checkpoint_path=checkpoint, model=model, model_name=name, data_normalization=data.normalization,
                                              data_frequencies=data.frequencies, device=device)
        model.eval()
        split = data.splits[data.test_split]
        full_x = to_model_layout(split.model_input)
        flattened = str(config.get("data", {}).get("representation", "")).lower() == "1d" and len(data.frequencies) > 1
        starts = make_window_starts(n_timesteps=len(full_x), lookback=lookback, rollout_horizon=max_horizon, stride=parameters["test_stride"],
                                    segments=split.segments)
        if not len(starts):
            raise ValueError("no windows")
        if flattened:
            row_std = _flattened_1d_normalization(data.normalization, split.segments, np.arange(len(full_x)))[0]["std_dbm"].reshape(-1, 1)
        else:
            row_std = np.asarray(data.normalization["std_dbm"], dtype=np.float64).reshape(1, -1)
        target_normalization = {}
        for h in horizons:
            rows = starts + lookback + h - 1
            target_normalization[h] = (_flattened_1d_normalization(data.normalization, split.segments, rows)[0] if flattened else data.normalization,
                                       raw_targets_to_model_layout(split.raw_dbm, rows))
        for c in all_offsets:
            shifted = (full_x + (c / row_std)).astype(np.float32) if c else full_x
            windows = make_window_batch_array(full_x=shifted, start_rows=starts, lookback=lookback)
            parts = {h: [] for h in horizons}
            with torch.no_grad():
                for i in range(0, len(windows), parameters["batch_size"]):
                    x = torch.from_numpy(windows[i:i + parameters["batch_size"]]).float().to(device)
                    out = forecast(model=model, x=x, prediction_horizon=parameters["prediction_horizon"], rollout_horizon=max_horizon, targets=None)
                    for h in horizons:
                        parts[h].append(out[:, h - 1, ...].detach().cpu().numpy())
            for h in horizons:
                norm, target_raw = target_normalization[h]
                prediction, _, _, _ = calculate_errors_and_export_arrays(prediction_normalized=np.concatenate(parts[h], axis=0).astype(np.float32),
                                                                         target_raw_model_layout=target_raw, normalization=dict(norm))
                collected.setdefault(c, {}).setdefault(h, []).append(prediction)
        n_windows += int(len(starts))
    base = {h: np.concatenate(collected[0.0][h], axis=0) for h in horizons}
    rows = []
    for c in offsets:
        shifted = {h: np.concatenate(collected[c][h], axis=0) for h in horizons}
        rows.extend(response_rows(base, shifted, c, tuple(h for h in horizons if horizons_out is None or h in horizons_out)))
    n_values = int(sum(v.size for v in base.values()))
    return {"rows": rows, "n_windows": n_windows, "horizons": list(horizons), "n_values": n_values, "test_stride": parameters["test_stride"]}


def out_key(prefix: str, entry: dict[str, Any]) -> str:
    tag = "__r2" if str(entry.get("prefix", "")).endswith("-r2") else ""
    h = f"/h{entry['horizon']}" if entry.get("source") == "direct" else ""
    return f"{prefix.strip('/')}/{entry['band']}/{entry['rep']}/{entry['model']}{tag}{h}/seed-{entry['seed']}"


def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one checkpoint, every offset."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    entry = spec["entry"]
    from training.ray.inference_benchmark import build_config
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    direct = entry.get("source") == "direct"
    name = entry.get("pipeline", entry["model"])
    base = out_key(args["prefix"], entry)
    record = {"kind": "offset_response_1min", "band": entry["band"], "receiver": "pooled-4", "rep": entry["rep"], "model": entry["model"], "seed": int(entry["seed"]),
              "source": entry.get("source"), "trained_horizon": int(entry["horizon"]) if direct else None, "checkpoint_key": entry["pt_key"],
              "intrinsic_window_normalization": INTRINSIC.get(entry["model"], "none"), "offsets_db": list(OFFSETS), "test_range": list(RANGE),
              "origin_stride": STRIDE[entry["rep"]]}
    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="offset1m-") as scratch_dir:
            run_args = {"repo_root": root, "test_start": RANGE[0], "test_end": RANGE[1], "device": args.get("device"),
                        "strides": {k: STRIDE[entry["rep"]] for k in ("oct", "band", "p15", "p15tuned", "direct")}}
            config, checkpoint = build_config(entry, run_args, client, bucket, Path(scratch_dir))
            config["data"]["split"]["ranges"]["test"] = {"start": RANGE[0], "end": RANGE[1]}
            config[name].setdefault("train", {})["test_stride"] = STRIDE[entry["rep"]]
            config.setdefault("training", {})["models"] = [name]
            result = forward_offsets(config, name, Path(checkpoint), OFFSETS)
        record.update(rows=result["rows"], n_windows=result["n_windows"], n_values=result["n_values"], horizons=result["horizons"],
                      code_revision=args["code_revision"], ok=True)
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(record, default=str).encode())
    return record


def main(argv=None) -> None:
    import subprocess

    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=["600_800", "2400_2600"])
    p.add_argument("--source", choices=("rolled", "direct", "both"), default="both")
    p.add_argument("--models", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--direct-horizons", type=int, nargs="+", default=[1, 15, 60])
    p.add_argument("--memory-gb", type=float, default=10.0)
    p.add_argument("--max-parallel", type=int, default=16)
    p.add_argument("--limit", type=int, help="smoke tests: at most this many jobs, spread across models")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="offset-response-1min-oct2026")
    p.add_argument("--code-revision", default=None)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    revision = args.code_revision
    if revision is None:
        try:
            revision = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            revision = "unknown"
    entries = []
    if args.source in ("rolled", "both"):
        entries += [e for e in discover_sources(client, bucket, ["oct", "band"], args.models or list(ONE_STEP_MODELS))
                    if e["rep"] in ("1d", "2d") and e["band"] in args.bands and int(e["seed"]) in args.seeds and e["model"] in ONE_STEP_MODELS]
    if args.source in ("direct", "both"):
        entries += [e for e in discover_sources(client, bucket, ["direct"], args.models or list(DIRECT_MODELS))
                    if e.get("resolution") == "1min" and e["band"] in args.bands and int(e["seed"]) in args.seeds and int(e["horizon"]) in args.direct_horizons
                    and e["model"] in DIRECT_MODELS]
    jobs = [{"entry": e} for e in entries]
    if args.skip_done:
        def done(j):
            try:
                return bool(json.loads(client.get_object(Bucket=bucket, Key=out_key(args.prefix, j["entry"]) + "/summary.json")["Body"].read()).get("ok"))
            except Exception:
                return False
        jobs = [j for j in jobs if not done(j)]
    if args.limit:
        seen: dict[tuple, int] = {}
        kept = []
        for j in jobs:
            k = (j["entry"]["band"], j["entry"]["rep"], j["entry"]["model"], j["entry"].get("source"))
            if seen.get(k, 0) < 1:
                seen[k] = 1
                kept.append(j)
        jobs = kept[:args.limit]
    print(f"[offset-1min] {len(jobs)} jobs, revision {revision[:12]}", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "code_revision": revision}
    ray.init(address=args.address, ignore_reinit_error=True)
    queue, pending, ok = list(jobs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            j = queue.pop(0)
            gpu = resources_for(j["entry"]["model"], j["entry"].get("source") == "direct")
            pending.append(ray.remote(num_cpus=4, num_gpus=gpu, memory=int(args.memory_gb * 2 ** 30), max_retries=1)(job).remote(j, shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        g = [r["gain"] for r in (row.get("rows") or []) if r["offset_db"] == 10.0]
        print(f"[offset-1min] {row.get('band')} {row.get('rep')} {row.get('model')} h{row.get('trained_horizon')} seed{row.get('seed')} ok={row.get('ok')} "
              f"gain(+10 dB)={[round(x, 3) for x in g]} {row.get('error', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[offset-1min] done: {ok} ok", flush=True)


if __name__ == "__main__":
    main()
