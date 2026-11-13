"""Transfer of the 15-minute 4D (map) models, from saved weights, to the paper's other evaluation sets.

The 4D models (LinearAR 4D, Residual LinearAR 4D, Residual ConvLSTM, STS-PredNet, DSwinLSTM-I; base map ``core4-p2``) were scored only on the
October test week and at guesthouse/law73 (``ncar_spatial``). This scores the same frozen weights, map built from the four core receivers, at
those four receivers (nearest grid cell, as in the paper's spatial section) on:

  week2            later interval, 2026-11-03 20:00 to 2026-11-10 20:00 (the October test run's files)
  july-<run>       separate collection: the core four in the 06-28 and 07-03 runs, on the interval all four share
  xband-<band>     new frequency: 800_1000 and 5725_5925 of the 07-03 run, bin by bin (bin k of the new band in place of bin k of the
                   training band: the converted file is copied with the training band's column labels), with the weights of either band

Conditions: ``frozen`` and ``centered`` (``evaluate_condition(center_windows="own")``, regime B of the transfer tables).
Outputs: ``transfer-ncar-2026-11/item12-4d/<set>/from-<band>/<model>/seed-N/{summary.json, <condition>/<receiver>/{forecasts.npz, per_bin.csv}}``.

    python -m training.ray.transfer_4d run --skip-done
    python -m training.ray.transfer_4d run --origin-limit 4 --prefix transfer-ncar-2026-11/item12-4d-smoke --models linearar4d --seeds 40   # smoke
    python -m training.ray.transfer_4d table
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

import numpy as np

PREFIX = "transfer-ncar-2026-11/item12-4d"
MODELS = ("linearar4d", "residuallinearar4d", "residualconvlstm", "stsprednet", "dswinlstm_i")
BANDS = ("600_800", "2400_2600")
WEEK2 = {"start": "2026-11-03T20:00:00Z", "end": "2026-11-10T20:00:00Z"}
JULY_RUNS = {"20260628": "july-20260628", "20260703": "july-20260703"}
REGIMES = {"frozen": None, "centered": "own", "refit": None}
DEFAULT_REGIMES = ("frozen", "centered")
STD_FLOOR_FRACTION = 0.25                                  # as training.common.transfer_regimes (regime C)


def summary_name(regimes) -> str:
    return "summary.json" if tuple(regimes) == DEFAULT_REGIMES else f"summary-{'-'.join(regimes)}.json"


def refit_checkpoint(checkpoint: Path, exp: dict[str, Any], root: str, out: Path) -> dict[str, Any] | None:
    """A copy of ``checkpoint`` whose per-bin normalization is re-estimated on the target: mean and standard deviation over the map cells and
    the 96 blocks (24 h) up to the first scored origin, standard deviation floored at 0.25 x the source's; weights unchanged (regime C)."""
    import pandas as pd
    import torch

    from training.common import spatial_ablation_eval as sae

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    norm = ckpt.get("normalization")
    if norm is None:
        return None
    rootp = Path(root)
    names = list(exp["baseline_receivers"])
    coords = sae._coordinates(sae._resolve(exp["locations"], rootp), str(exp.get("collection_key", "endpoints")), names)
    series, _, _ = sae._condition_data(exp, "geometry_correct", np.asarray(ckpt["frequencies"], dtype=np.float32), coords, rootp)
    g = exp.get("grid", {})
    grid = sae.derive_frozen_grid([coords[n][0] for n in names], [coords[n][1] for n in names], height=int(g.get("height", 10)),
                                  width=int(g.get("width", 10)), power=float(g.get("power", 2.0)), width_meters=g.get("width_meters"),
                                  height_meters=g.get("height_meters"), origin_lon=g.get("origin_lon"), origin_lat=g.get("origin_lat"),
                                  power_domain=str(g.get("power_domain", "db")), method=str(g.get("method", "idw")), variogram=g.get("variogram"))
    start = pd.Timestamp(exp["geometry_timing"]["start"])
    mask = np.asarray((series.timestamps > start - pd.Timedelta(hours=24)) & (series.timestamps <= start))
    if mask.sum() < 48:
        raise RuntimeError(f"only {int(mask.sum())} calibration rows before {start}")
    maps = sae.build_idw_maps(series.values_dbm[mask], series.longitudes, series.latitudes, grid)
    flat = np.asarray(maps, dtype=np.float64).reshape(-1, maps.shape[-1])
    src_mean, src_std = np.asarray(norm["mean_dbm"], dtype=np.float64), np.asarray(norm["std_dbm"], dtype=np.float64)
    mean = np.nanmean(flat, axis=0)
    std = np.maximum(np.nanstd(flat, axis=0), STD_FLOOR_FRACTION * src_std.squeeze())
    new = {**norm, "mean_dbm": mean.reshape(src_mean.shape).astype(np.float32), "std_dbm": std.reshape(src_std.shape).astype(np.float32)}
    ckpt["normalization"] = new
    torch.save(ckpt, out)
    return {"normalization": new, "calibration_rows": int(mask.sum()), "mean_shift_db": float(np.mean(mean - src_mean.squeeze())),
            "std_ratio": float(np.mean(std / src_std.squeeze()))}


def heuristics(root: str, files: dict[str, str], npz, ks: dict[str, dict[int, int]], band: str) -> dict[str, dict[str, dict]]:
    """Persistence, 24 h lookback mean, same time yesterday and the tuned lookback median at each receiver, on exactly the evaluator's origins.
    Returns heuristic -> receiver -> ncar_cells matrices. Windows whose targets do not match the evaluator's are dropped (counted)."""
    import pandas as pd

    from training.ray.ncar_baselines import LOOKBACK, baseline_forecasts
    from training.ray.spatial15 import HORIZONS

    origins = pd.to_datetime(np.asarray(npz["origin_timestamps"]).astype(str), utc=True)
    targets = npz["targets_dbm"].astype(np.float64)                         # (N, H, R, F)
    out: dict[str, dict[str, dict]] = {}
    for r, name in enumerate(np.asarray(npz["receivers"]).astype(str)):
        frame = pd.read_csv(Path(root) / files[str(name)])
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame = frame.set_index("timestamp_utc").sort_index()
        frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="15min")).interpolate(limit=16, limit_area="inside")
        data = frame.to_numpy(dtype=np.float32)
        last = frame.index.get_indexer(origins)                             # the origin is the last observed step
        ok = (last >= LOOKBACK - 1) & (last + max(HORIZONS) < len(data))
        for i, h in enumerate(HORIZONS):
            ok &= np.nanmax(np.abs(data[np.clip(last + h, 0, len(data) - 1)] - targets[:, i, r, :]), axis=1) < 1e-2
        starts = last[ok] - (LOOKBACK - 1)
        preds = baseline_forecasts(data, starts, {"tuned_median": ks[band]})
        for hname, by_h in preds.items():
            out.setdefault(hname, {})[str(name)] = {int(h): {"origin": (origins[ok].asi8 // 10 ** 9).astype(np.int64), "pred": by_h[h].astype(np.float32),
                                                              "target": targets[ok, i, r, :].astype(np.float32)}
                                                    for i, h in enumerate(HORIZONS) if h in by_h}
        out.setdefault("_dropped", {})[str(name)] = int((~ok).sum())
    return out


def set_names() -> list[str]:
    return ["week2", *JULY_RUNS.values(), "xband-800_1000", "xband-5725_5925"]


def prepare(root: str, name: str, band: str) -> tuple[dict[str, str], dict[str, str], str]:
    """(receiver name -> 15-minute file, scoring interval, target band) of a set for weights trained on ``band``; converts on this node."""
    import pandas as pd

    from training.ray import transfer_all as ta
    from training.ray import transfer_crossband as tc
    from training.ray import transfer_external as te
    from training.ray.powder15 import OUT_NAME, OUT_ROOT, TEST_RUN, ensure_powder15
    from training.ray.spatial15 import CORE, SIX

    if name == "week2":
        ensure_powder15(band, root, tuple(CORE))
        return {SIX[n]: f"{OUT_ROOT}/{n}/{TEST_RUN}/{band}/{OUT_NAME}" for n in CORE}, dict(WEEK2), band
    if name.startswith("july-"):
        run = name.split("-")[1]
        rows = [t for t in ta.targets(root, ["powder"], [band], ta.JULY_INTERVALS, core=False) if t["site"] in CORE and t["run"].startswith(run)]
        tband = band
    else:
        tband = name.split("-", 1)[1]
        rows = [t for t in tc.targets(root, [tband]) if t["site"] in CORE]
    if {t["site"] for t in rows} != set(CORE):
        raise RuntimeError(f"{name}/{band}: usable targets only at {[t['site'] for t in rows]}")
    files = {}
    for t in rows:
        te.convert(root, t)
        path = te.converted_path(root, t)
        if tband != band:                                  # positional cross-band: relabel the columns with the training band's bins
            ensure_powder15(band, root, tuple(CORE))
            labels = list(pd.read_csv(Path(root) / OUT_ROOT / CORE[0] / TEST_RUN / band / OUT_NAME, nrows=0).columns)
            out = Path(root) / "data/ext15x" / t["site"] / t["run"] / f"{tband}-as-{band}" / "power_15min.csv"
            if not out.exists():
                frame = pd.read_csv(path)
                if len(frame.columns) != len(labels):
                    raise RuntimeError(f"{path}: {len(frame.columns)} columns, expected {len(labels)}")
                frame.columns = labels
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(f".{os.getpid()}.tmp")
                frame.to_csv(tmp, index=False)
                os.replace(tmp, out)
            path = out
        files[SIX[t["site"]]] = str(Path(path).relative_to(root)) if str(path).startswith(str(root)) else str(path)
    start = max(pd.Timestamp(t["target_start"]) for t in rows)
    end = min(pd.Timestamp(t["target_end"]) for t in rows)
    return files, {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")}, tband


def cell_job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    import torch
    import yaml

    from training.common.spatial_ablation_eval import _checkpoint_metadata, evaluate_condition
    from training.ray import ncar_cells
    from training.ray.ncar_spatial import aggregate, receiver_mats, weights_key
    from training.ray.spatial15 import CORE, CPU_MODELS, experiment_yaml, grid_for, model_config_for, six_receiver_frame, variants
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    name, band, model, seed = spec["set"], spec["band"], spec["model"], int(spec["seed"])
    base = f"{args['prefix'].strip('/')}/{name}/from-{band}/{model}/seed-{seed}"
    record: dict[str, Any] = {"kind": "transfer_4d", "set": name, "source_band": band, "model": model, "seed": seed, "variant": "core4-p2",
                              "readout": "nearest_cell", "code_revision": args.get("code_revision"), "origin_limit": args.get("origin_limit")}
    started = time.time()
    device = "cpu" if model in CPU_MODELS else "cuda"
    try:
        regimes = ["frozen"] if model == "lookbackmean4d" else list(args.get("regimes") or DEFAULT_REGIMES)
        key = None if model == "lookbackmean4d" else weights_key(client, bucket, band, "core4-p2", model, seed)
        if key is None and model != "lookbackmean4d":
            raise FileNotFoundError("no saved weights for this cell")
        files, timing, tband = prepare(root, name, band)
        record.update(target_band=tband, timing=timing, files=files, weights_key=key)
        variant = next(v for v in variants(["s2"]) if v["name"] == "core4-p2")
        exp = experiment_yaml(band, list(CORE), None, grid_for(variant, six_receiver_frame(root)))
        exp["geometry_timing"], exp["geometry_files"] = timing, files
        with tempfile.TemporaryDirectory(prefix="transfer4d-") as scratch_dir:
            scratch = Path(scratch_dir)
            model_config = scratch / "models.yaml"
            if model == "lookbackmean4d":
                checkpoint, normalization = None, None
                model_config.write_text(yaml.safe_dump({"windowing": {"lookback": 96, "horizons": [4, 12, 24, 48]}}))
            else:
                checkpoint = scratch / Path(key).name
                client.download_file(bucket, key, str(checkpoint))
                meta = json.loads(client.get_object(Bucket=bucket, Key=key + ".meta.json")["Body"].read())
                _, normalization = _checkpoint_metadata(checkpoint)
                model_config.write_text(yaml.safe_dump(model_config_for(meta["config"], model)))
            source_checkpoint, source_normalization = checkpoint, normalization
            metrics: dict[str, Any] = {}
            for regime in regimes:
                center = REGIMES[regime]
                checkpoint, normalization = source_checkpoint, source_normalization
                if regime == "refit":
                    refit = refit_checkpoint(source_checkpoint, exp, root, scratch / "refit.pt")
                    if refit is None:
                        continue
                    checkpoint, normalization = scratch / "refit.pt", refit.pop("normalization")
                    record["refit"] = refit
                exp_path = Path(root) / "training/configs" / f".transfer4d-{os.getpid()}-{name}-{band}-{model}-{seed}.yaml"
                exp_path.write_text(yaml.safe_dump(exp))
                out = scratch / regime
                try:
                    evaluate_condition(experiment_path=exp_path, model_config_path=model_config, checkpoint_path=checkpoint, model_name=model,
                                       seed=seed, condition="geometry_correct", output_directory=out, batch_size=args["batch_size"],
                                       device=device, center_windows=center, readout="nearest_cell", origin_limit=args.get("origin_limit"))
                finally:
                    exp_path.unlink(missing_ok=True)
                with np.load(out / "receiver_forecasts.npz", allow_pickle=True) as npz:
                    by_receiver = receiver_mats(npz, normalization)
                for receiver, mats in by_receiver.items():
                    rkey = receiver.replace(" ", "")
                    ncar_cells.save_cell(client, bucket, f"{base}/{regime}/{rkey}", mats, ncar_cells.band_frequencies(tband))
                    metrics.setdefault(regime, {})[rkey] = aggregate(mats)
                if model == "lookbackmean4d":          # the per-receiver heuristics on exactly these origins
                    from training.ray.ncar_baselines import selected_k

                    with np.load(out / "receiver_forecasts.npz", allow_pickle=True) as npz:
                        heur = heuristics(root, files, npz, selected_k(client, bucket), band)
                    record["heuristic_windows_dropped"] = heur.pop("_dropped")
                    for hname, by_r in heur.items():
                        for receiver, mats in by_r.items():
                            rkey = receiver.replace(" ", "")
                            ncar_cells.save_cell(client, bucket, f"{base}/heuristic-{hname}/{rkey}", mats, ncar_cells.band_frequencies(tband))
                            metrics.setdefault(f"heuristic-{hname}", {})[rkey] = aggregate(mats)
        record["metrics"] = metrics
        if device == "cuda":
            record["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-2500:])
    record["wall_seconds"] = round(time.time() - started, 1)
    if record["ok"] or "no saved weights" not in record.get("error", ""):
        client.put_object(Bucket=bucket, Key=f"{base}/{summary_name(args.get('regimes') or DEFAULT_REGIMES)}" if record["ok"] else f"{base}/errors/{int(time.time())}.json",
                          Body=json.dumps(record, default=str, indent=1).encode())
    return record


def cmd_run(a: argparse.Namespace) -> None:
    import ray

    from training.ray.spatial15 import CPU_MODELS
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()

    def done(s):
        try:
            return bool(json.loads(client.get_object(Bucket=a.bucket, Key=f"{a.prefix}/{s['set']}/from-{s['band']}/{s['model']}/seed-{s['seed']}/{summary_name(a.regimes)}")["Body"].read()).get("ok"))
        except Exception:
            return False

    specs = [{"set": n, "band": b, "model": m, "seed": s} for n in a.sets for b in a.bands for m in a.models
             for s in ([42] if m == "lookbackmean4d" else a.seeds)]
    if a.skip_done:
        specs = [s for s in specs if not done(s)]
    print(f"[transfer4d] {len(specs)} cells", flush=True)
    ray.init(address=a.address, ignore_reinit_error=True)
    shared = {"repo_root": a.repo_root, "bucket": a.bucket, "prefix": a.prefix, "batch_size": a.batch_size, "code_revision": a.code_revision,
              "origin_limit": a.origin_limit or None, "regimes": a.regimes}
    cpu = ray.remote(num_cpus=4, num_gpus=0, memory=int(a.memory_gb * 2 ** 30), max_retries=1)(cell_job)
    gpu = ray.remote(num_cpus=4, num_gpus=a.gpu_fraction, memory=int(a.memory_gb * 2 ** 30), max_retries=1)(cell_job)
    queue, pending, ok, failed, missing = list(specs), [], 0, 0, 0
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            s = queue.pop(0)
            pending.append((cpu if s["model"] in CPU_MODELS else gpu).remote(s, shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            r = ray.get(finished[0])
        except Exception as error:
            r = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:300]}"}
        if "no saved weights" in str(r.get("error", "")):
            missing += 1
            continue
        ok += bool(r.get("ok"))
        failed += not r.get("ok")
        shown = {c: round(float(np.mean([m["mae_db"] for rc in v.values() for h, m in rc.items() if h in ("4", "12", "24")])), 3)
                 for c, v in (r.get("metrics") or {}).items()}
        print(f"[transfer4d] {r.get('set')} from-{r.get('source_band')} {r.get('model')} seed{r.get('seed')} ok={r.get('ok')} {shown} "
              f"{r.get('error', '')} {(r.get('trace') or '')[-600:] if not r.get('ok') else ''} ({r.get('wall_seconds')}s) "
              f"[{len(queue) + len(pending)} left]", flush=True)
    print(f"[transfer4d] done: {ok} ok, {failed} failed, {missing} without weights", flush=True)


def cmd_table(a: argparse.Namespace) -> None:
    import pandas as pd

    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    rows = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix=a.prefix + "/"):
        for o in page.get("Contents", []):
            if not (o["Key"].rsplit("/", 1)[-1].startswith("summary") and o["Key"].endswith(".json")) or "/tables/" in o["Key"]:
                continue
            d = json.loads(client.get_object(Bucket=a.bucket, Key=o["Key"])["Body"].read())
            for regime, by_r in d["metrics"].items():
                for receiver, by_h in by_r.items():
                    for h, m in by_h.items():
                        rows.append({"set": d["set"], "source_band": d["source_band"], "target_band": d["target_band"], "model": d["model"],
                                     "seed": d["seed"], "condition": regime, "receiver": receiver, "horizon": int(h),
                                     "first_origin_target_start": d["timing"]["start"], "end": d["timing"]["end"], **m})
    out = pd.DataFrame(rows)
    key = f"{a.prefix}/tables/{a.stamp}/transfer_4d.csv"
    client.put_object(Bucket=a.bucket, Key=key, Body=out.to_csv(index=False).encode())
    print(f"[transfer4d] wrote s3://{a.bucket}/{key} ({len(out)} rows)", flush=True)
    pd.set_option("display.width", 250)
    g = out[out.horizon.isin([4, 12, 24])]
    seeds = g.groupby(["set", "source_band", "model", "condition"]).seed.nunique().rename("seeds")
    print(g.groupby(["set", "source_band", "model", "condition"]).agg(mae_db=("mae_db", "mean"), n_windows=("n_windows", "max")).join(seeds).round(3).to_string())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "table"):
        s = sub.add_parser(name)
        s.add_argument("--prefix", default=PREFIX)
        s.add_argument("--bucket", default=os.environ.get("SPECTRUM_S3_BUCKET", "spectrum"))
        s.add_argument("--repo-root", default="/workspace/spectrum-usage")
        s.add_argument("--stamp", default="v1")
        if name == "run":
            s.add_argument("--sets", nargs="+", default=set_names(), choices=set_names())
            s.add_argument("--bands", nargs="+", default=list(BANDS))
            s.add_argument("--models", nargs="+", default=list(MODELS), help="lookbackmean4d also writes the per-receiver heuristics")
            s.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES), choices=list(REGIMES))
            s.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
            s.add_argument("--origin-limit", type=int, default=0, help="score only the first N origins (smoke tests)")
            s.add_argument("--gpu-fraction", type=float, default=0.25)
            s.add_argument("--max-parallel", type=int, default=16)
            s.add_argument("--memory-gb", type=float, default=16.0)
            s.add_argument("--batch-size", type=int, default=8)
            s.add_argument("--skip-done", action="store_true")
            s.add_argument("--code-revision", default=None)
            s.add_argument("--address", default="auto")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    os.chdir(a.repo_root)
    {"run": cmd_run, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
