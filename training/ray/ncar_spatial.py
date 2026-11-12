"""NCAR transfer batch, item 7 and step 0c (4D rows): spatial (4D) models at the physical receivers from saved weights, with per-window arrays.

Item 7 (``run``): the neural 4D models Residual ConvLSTM and DSwinLSTM-I, trained on core-four maps (variant ``core4-p2``, the tuning finals of
``tuning/powder15_4d/<band>/4d/<model>/weights/seed-N/``; the H100 finals-only copies under ``tuning-h100/`` are used only when the UC final is absent),
scored with ``training.common.spatial_ablation_eval.evaluate_condition`` exactly as ``training.ray.spatial15`` does:
  insample  ``geometry_correct``: the four core receivers, every one in the input map;
  absent    ``six_to_guesthouse-<r>``: guest house or law 73 left out of the input map (the held-out receiver);
  added     ``seven_to_guesthouse-<r>``: the held-out receiver added to the input map without retraining.
No leave-one-out (``loo-*``) checkpoints exist for the neural models (training one takes up to a day), so the held-out receivers are guest house
and law 73, the receivers the base model never saw. Weights that are not in S3 yet are polled for (``--poll-until``).

Step 0c (``consistency``): one stored ``spatial15-tuned-oct2026`` cell each of Residual LinearAR 4D and STS-PredNet (``loo-guesthouse``, seed 40,
600-800) rescored on NCAR, compared per (condition, receiver, horizon) on the evaluator's region metrics (count-weighted over all bins):
MAE, RMSE, MSE (= RMSE^2 pooled); the stored cells keep no forecasts, so MAPE has no stored value (NCAR value reported, diff empty).

Outputs (source bucket), per cell ``transfer-ncar-2026-11/item7/<band>/<variant>/<model>/seed-N/``:
  ``summary.json``                         provenance (weights key, sha256), peak GPU memory, and four metrics per (condition, receiver, horizon)
  ``<condition>/<receiver>/forecasts.npz``  ``training.ray.ncar_cells`` layout (origin = last observed step; dBm; columns = bins)
  ``<condition>/<receiver>/per_bin.csv``    four metrics per (horizon, bin), with the bin frequency
  ``<condition>/eval/{result.json, metadata.json, region_metrics.csv}``  the evaluator's own files

    python -m training.ray.ncar_spatial consistency --out-key transfer-ncar-2026-11/step0/consistency_4d.csv
    python -m training.ray.ncar_spatial run --models residualconvlstm dswinlstm_i --seeds 40 41 42 43 44 --skip-done --poll-until 2026-11-14T00:00:00Z
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

PREFIX = "transfer-ncar-2026-11/item7"
STORED_PREFIX = "spatial15-tuned-oct2026"
WEIGHT_PREFIXES = ("tuning", "tuning-h100")
COND_NAME = {"geometry_correct": "insample", "six_to_guesthouse": "absent", "seven_to_guesthouse": "added"}
CONSISTENCY_CELLS = (("600_800", "loo-guesthouse", "residuallinearar4d", 40), ("600_800", "loo-guesthouse", "stsprednet", 40))


def weights_key(client, bucket: str, band: str, variant: str, model: str, seed: int) -> str | None:
    """Saved weights of a cell: a trained variant's own ``weights/``, or (``core4-p2``) the tuning final of that seed."""
    if variant == "core4-p2":
        candidates = [f"{p}/powder15_4d/{band}/4d/{model}/weights/seed-{seed}/" for p in WEIGHT_PREFIXES]
    else:
        candidates = [f"{STORED_PREFIX}/{band}/{variant}/{model}/seed-{seed}/weights/"]
    for where in candidates:
        keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=where)
                for o in page.get("Contents", []) if o["Key"].endswith(".pt")]
        if keys:
            try:                                   # the sidecar is written after the weights; a cell is ready only with both
                client.head_object(Bucket=bucket, Key=keys[0] + ".meta.json")
                return keys[0]
            except Exception:
                continue
    return None


def labels_for(variant: dict[str, Any]) -> list[tuple[str, str | None, str]]:
    """(evaluator condition, held-out node or None, label) in the order ``spatial15.physical_scores`` runs them."""
    from training.ray.spatial15 import SIX

    held_out = [n for n in SIX if n not in variant["nodes"]]
    out = [("geometry_correct", held_out[0] if held_out else None, "geometry_correct")]
    for held in held_out:
        for cond in ("six_to_guesthouse", "seven_to_guesthouse"):
            out.append((cond, held, f"{cond}-{SIX[held].replace(' ', '')}"))
    return out


def receiver_mats(npz, normalization) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Per receiver, ``ncar_cells`` matrices (origin epoch seconds, pred/target dBm (N, F)) from the evaluator's ``receiver_forecasts.npz``."""
    import pandas as pd

    from training.ray.spatial15 import HORIZONS

    pred = npz["predictions_normalized"].astype(np.float64)            # (N, H, R, F)
    if normalization is not None:
        mean = np.asarray(normalization["mean_dbm"], dtype=np.float64).squeeze()
        std = np.asarray(normalization["std_dbm"], dtype=np.float64).squeeze()
        pred = pred * std + mean
    target = npz["targets_dbm"].astype(np.float64)
    origins = pd.to_datetime(np.asarray(npz["origin_timestamps"]).astype(str), utc=True).asi8 // 10 ** 9
    order = np.argsort(origins, kind="stable")
    out = {}
    for r, name in enumerate(np.asarray(npz["receivers"]).astype(str)):
        out[str(name)] = {int(h): {"origin": origins[order].astype(np.int64), "pred": pred[order, i, r, :].astype(np.float32),
                                   "target": target[order, i, r, :].astype(np.float32)} for i, h in enumerate(HORIZONS)}
    return out


def aggregate(mats: dict[int, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Four metrics (and MAPE dBm) per horizon, pooled over windows and bins (the transfer15 definitions)."""
    out = {}
    for h, m in mats.items():
        e = m["pred"].astype(np.float64) - m["target"].astype(np.float64)
        e32, t32 = e.astype(np.float32), m["target"].astype(np.float32)
        lin = np.abs(np.power(np.float32(10.0), e32 / np.float32(10.0)) - np.float32(1.0)) * np.float32(100.0)
        dbm = np.abs(e32) / np.maximum(np.abs(t32), np.float32(1e-6)) * np.float32(100.0)
        out[str(h)] = {"n_windows": int(e.shape[0]), "mae_db": float(np.abs(e).mean()), "mse_db2": float((e ** 2).mean()),
                       "rmse_db": float(np.sqrt((e ** 2).mean())), "mape_linear_pct": float(lin.mean(dtype=np.float64)),
                       "mape_dbm_pct": float(dbm.mean(dtype=np.float64)), "bias_db": float(e.mean())}
    return out


def cell_job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one (band, variant, model, seed) cell, every condition, arrays and metrics to S3."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    import torch
    import yaml

    from training.common.spatial_ablation_eval import _checkpoint_metadata, evaluate_condition
    from training.ray import ncar_cells
    from training.ray.powder15 import ensure_powder15
    from training.ray.spatial15 import (CPU_MODELS, SIX, experiment_yaml, grid_for, model_config_for, six_receiver_frame, variants)
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    band, variant_name, model, seed = spec["band"], spec["variant"], spec["model"], int(spec["seed"])
    base = f"{args['prefix'].strip('/')}/{band}/{variant_name}/{model}/seed-{seed}"
    record: dict[str, Any] = {"kind": "ncar_spatial", "band": band, "variant": variant_name, "model": model, "seed": seed,
                              **ncar_cells.tags(model, "4d"), "code_revision": args.get("code_revision")}
    started = time.time()
    device = "cpu" if model in CPU_MODELS else "cuda"
    try:
        variant = next(v for v in variants(["s2", "s3"]) if v["name"] == variant_name)
        grid = grid_for(variant, six_receiver_frame(root))
        ensure_powder15(band, root, tuple(SIX))
        key = weights_key(client, bucket, band, variant_name, model, seed)
        if key is None:
            raise FileNotFoundError("no saved weights for this cell")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory(prefix="ncar-spatial-") as scratch_dir:
            scratch = Path(scratch_dir)
            checkpoint = scratch / Path(key).name
            client.download_file(bucket, key, str(checkpoint))
            meta = json.loads(client.get_object(Bucket=bucket, Key=key + ".meta.json")["Body"].read())
            record.update(weights_key=key, weights_sha256=meta.get("sha256"))
            _, normalization = _checkpoint_metadata(checkpoint)
            model_config = scratch / "models.yaml"
            model_config.write_text(yaml.safe_dump(model_config_for(meta["config"], model)))
            metrics: dict[str, Any] = {}
            for condition, held, label in labels_for(variant):
                exp_path = Path(root) / "training/configs" / f".ncar-spatial-{os.getpid()}-{variant_name}-{model}-{seed}.yaml"
                exp_path.write_text(yaml.safe_dump(experiment_yaml(band, variant["nodes"], held, grid)))   # as spatial15.physical_scores
                out = scratch / "cond" / label
                try:
                    summary = evaluate_condition(experiment_path=exp_path, model_config_path=model_config, checkpoint_path=checkpoint,
                                                 model_name=model, seed=seed, condition=condition, output_directory=out,
                                                 batch_size=args["batch_size"], device=device)
                finally:
                    exp_path.unlink(missing_ok=True)
                cond = COND_NAME[condition]
                for name in ("result.json", "metadata.json", "region_metrics.csv"):
                    if (out / name).exists():
                        client.upload_file(str(out / name), bucket, f"{base}/{label}/eval/{name}")
                with np.load(out / "receiver_forecasts.npz", allow_pickle=True) as npz:
                    by_receiver = receiver_mats(npz, normalization)
                for receiver, mats in by_receiver.items():
                    rkey = receiver.replace(" ", "")
                    if cond != "insample" and rkey != label.split("-")[-1]:
                        continue                                     # absent/added: the held-out receiver only
                    ncar_cells.save_cell(client, bucket, f"{base}/{cond}/{rkey}", mats, ncar_cells.band_frequencies(band))
                    metrics.setdefault(cond, {})[rkey] = aggregate(mats)
                metrics.setdefault("_evaluator", {})[label] = {k: summary.get(k) for k in summary if k.startswith(("mae_db", "mean_horizon", "n_origins"))}
        record["metrics"] = metrics
        if device == "cuda":
            record["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    record["wall_seconds"] = round(time.time() - started, 1)
    if record["ok"] or not args.get("quiet_missing") or "no saved weights" not in record.get("error", ""):
        client.put_object(Bucket=bucket, Key=f"{base}/summary.json" if record["ok"] else f"{base}/errors/{int(time.time())}.json",
                          Body=json.dumps(record, default=str, indent=1).encode())
    return record


def _done(client, bucket: str, prefix: str, spec: dict[str, Any]) -> bool:
    key = f"{prefix.strip('/')}/{spec['band']}/{spec['variant']}/{spec['model']}/seed-{spec['seed']}/summary.json"
    try:
        return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
    except Exception:
        return False


def run_specs(a: argparse.Namespace, specs: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    import ray

    shared = {"repo_root": a.repo_root, "bucket": a.bucket, "prefix": prefix, "batch_size": a.batch_size, "code_revision": a.code_revision,
              "quiet_missing": True}
    from training.ray.spatial15 import CPU_MODELS

    cpu = ray.remote(num_cpus=4, num_gpus=0, memory=int(a.memory_gb * 2 ** 30), max_retries=1)(cell_job)
    gpu = ray.remote(num_cpus=4, num_gpus=a.gpu_fraction, memory=int(a.memory_gb * 2 ** 30), max_retries=1)(cell_job)
    queue, pending, rows = list(specs), [], []
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            s = queue.pop(0)
            pending.append((cpu if s["model"] in CPU_MODELS else gpu).remote(s, shared))
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        rows.append(r)
        shown = {c: {rc: round(float(np.mean([m["mae_db"] for m in hm.values()])), 3) for rc, hm in v.items()}
                 for c, v in (r.get("metrics") or {}).items() if not c.startswith("_")}
        print(f"[ncar-spatial] {r['band']} {r['variant']} {r['model']} seed{r['seed']} ok={r['ok']} peak={r.get('peak_gpu_memory_gb')}GB "
              f"{shown} {r.get('error', '')} ({r['wall_seconds']}s) [{len(queue) + len(pending)} left]", flush=True)
    return rows


def cmd_run(a: argparse.Namespace) -> None:
    import datetime as dt

    import ray

    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    ray.init(address=a.address, ignore_reinit_error=True)
    deadline = dt.datetime.fromisoformat(a.poll_until.replace("Z", "+00:00")) if a.poll_until else None
    while True:
        wanted = [{"band": b, "variant": "core4-p2", "model": m, "seed": s} for b in a.bands for m in a.models for s in a.seeds]
        todo = [s for s in wanted if not (a.skip_done and _done(client, a.bucket, a.prefix, s))]
        ready = [s for s in todo if weights_key(client, a.bucket, s["band"], s["variant"], s["model"], s["seed"])]
        waiting = [s for s in todo if s not in ready]
        print(f"[ncar-spatial] {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ}: {len(wanted)} cells, {len(wanted) - len(todo)} done, "
              f"{len(ready)} ready, {len(waiting)} waiting for weights: "
              f"{[(s['band'], s['model'], s['seed']) for s in waiting]}", flush=True)
        if ready:
            run_specs(a, ready, a.prefix)
            continue
        if not waiting or deadline is None or dt.datetime.now(dt.timezone.utc) >= deadline:
            break
        time.sleep(a.poll_seconds)
    print("[ncar-spatial] finished", flush=True)


def cmd_consistency(a: argparse.Namespace) -> None:
    import io

    import pandas as pd
    import ray

    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    ray.init(address=a.address, ignore_reinit_error=True)
    prefix = f"{a.prefix.rsplit('/', 1)[0]}/step0/consistency-4d/{a.tag}"
    specs = [{"band": b, "variant": v, "model": m, "seed": s} for b, v, m, s in CONSISTENCY_CELLS]
    records = run_specs(a, specs, prefix)

    def pooled(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.assign(sq=frame.rmse_db ** 2 * frame["count"], ab=frame.mae_db * frame["count"])
        g = frame.groupby(["receiver", "horizon"]).agg(n=("count", "sum"), ab=("ab", "sum"), sq=("sq", "sum")).reset_index()
        return g.assign(mae=g.ab / g.n, mse=g.sq / g.n, rmse=np.sqrt(g.sq / g.n))

    rows = []
    for (band, variant, model, seed), rec in zip(CONSISTENCY_CELLS, records):
        if not rec.get("ok"):
            rows.append({"model": model, "cell": f"{band} {variant} seed {seed}", "metric": "run", "stored": None, "ncar": None, "diff": None,
                         "error": rec.get("error")})
            continue
        from training.ray.spatial15 import variants as all_variants

        v = next(x for x in all_variants(["s2", "s3"]) if x["name"] == variant)
        for condition, held, label in labels_for(v):
            new = pooled(pd.read_csv(io.BytesIO(client.get_object(Bucket=a.bucket, Key=f"{prefix}/{band}/{variant}/{model}/seed-{seed}/"
                                                                          f"{label}/eval/region_metrics.csv")["Body"].read())))
            old = pooled(pd.read_csv(io.BytesIO(client.get_object(Bucket=a.bucket, Key=f"{STORED_PREFIX}/{band}/{variant}/{model}/seed-{seed}/"
                                                                          f"eval/{label}/region_metrics.csv")["Body"].read())))
            both = new.merge(old, on=["receiver", "horizon"], suffixes=("_ncar", "_stored"))
            for _, r in both.iterrows():
                cell = f"{band} {variant} seed {seed} {COND_NAME[condition]} {r.receiver} t+{int(r.horizon)}"
                for m in ("mae", "mse", "rmse"):
                    rows.append({"model": model, "cell": cell, "metric": m, "stored": r[f"{m}_stored"], "ncar": r[f"{m}_ncar"],
                                 "diff": r[f"{m}_ncar"] - r[f"{m}_stored"]})
                rkey = str(r.receiver).replace(" ", "")
                ncar_mape = rec["metrics"].get(COND_NAME[condition], {}).get(rkey, {}).get(str(int(r.horizon)), {}).get("mape_linear_pct")
                rows.append({"model": model, "cell": cell, "metric": "mape_linear", "stored": None, "ncar": ncar_mape, "diff": None,
                             "note": "stored cell keeps no forecasts or MAPE"})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(table.to_string(), flush=True)
    mae = table[table.metric == "mae"]["diff"].abs()
    print(f"[ncar-spatial] consistency: max |MAE diff| {mae.max():.3g} dB over {len(mae)} cells; pass (<= 1e-3): {bool((mae <= 1e-3).all())}",
          flush=True)
    if a.out_key:
        try:
            client.head_object(Bucket=a.bucket, Key=a.out_key)
            raise SystemExit(f"refusing to overwrite {a.out_key}")
        except client.exceptions.ClientError:
            pass
        client.put_object(Bucket=a.bucket, Key=a.out_key, Body=table.to_csv(index=False).encode())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "consistency"):
        s = sub.add_parser(name)
        s.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
        s.add_argument("--models", nargs="+", default=["residualconvlstm", "dswinlstm_i"])
        s.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
        s.add_argument("--batch-size", type=int, default=8)
        s.add_argument("--gpu-fraction", type=float, default=0.25)
        s.add_argument("--memory-gb", type=float, default=16.0)
        s.add_argument("--max-parallel", type=int, default=2)
        s.add_argument("--skip-done", action="store_true")
        s.add_argument("--poll-until", default=None, help="keep polling S3 for missing weights until this UTC time")
        s.add_argument("--poll-seconds", type=int, default=600)
        s.add_argument("--tag", default="run1", help="consistency: sub-prefix of this attempt (never reused)")
        s.add_argument("--out-key", default=None)
        s.add_argument("--prefix", default=PREFIX)
        s.add_argument("--bucket", default=None)
        s.add_argument("--code-revision", default=None)
        s.add_argument("--address", default="auto")
        s.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    os.chdir(a.repo_root)
    a.bucket = a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    {"run": cmd_run, "consistency": cmd_consistency}[a.command](a)


if __name__ == "__main__":
    main()
