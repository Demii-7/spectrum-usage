"""Periodicity-crossed frozen-input frequency ablation (evaluation only; nothing is retrained or fitted).

Extends ``training.ray.frequency_ablation`` (same models, checkpoints, origins, receivers, regions; the target region's own bins are never touched,
only the target region's bins are scored) with a CLASS of the other bins and a KIND of disturbance:

    disturbed set  all | P_only | N_only | noise_only      (class of each out-of-region bin, per receiver, from evaluation/results/bin_periodicity_classes.csv)
    disturbance    shift96 (24 h) | shift192 (48 h) | shift144 (36 h, 12 h out of phase) | shift210 (52.5 h) | level (each disturbed bin replaced by its
                   mean over the input window)

For rolled-out models the disturbance applies to the initial input window only. For every (receiver, condition) the output keeps

    cls|<recv>|<cond>|<P|N|M>   per-origin, per-horizon SUM of the absolute errors over the target bins of that class (target bins = the bins of every
                                non-noise region in which the condition disturbs at least one bin; each region is its own forward pass), float32 (origins, horizons)
    clsr|...                    the same sums of the REAL-input errors over exactly the same regions; cnt|<recv>|<cond>: number of target bins (P, N, middle)
    bin|<recv>|<cond>           per-target-bin mean absolute error over the main origins (lookback inside the test week), (horizons, 200); ``binnw`` the same
                                over the origins that contain no wrapped rows (start >= 210)
    rmae|<recv>|<region>|<cond> region-mean per-origin errors for the conditions real, all_shift96, all_shift210 only (reproduces the existing ablation)

and the real-input per-bin errors over all 200 bins (``binmae_main``, ``binmae_all``) for the identity check. ``--probe`` instead compares the recomputed real
per-bin errors with the stored rescored per-bin table for several origin-set definitions (to explain the small identity differences of the first run).

    python -m training.ray.frequency_ablation_classes --models vanillalstm revinmlp --seeds 42 --skip-done
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from training.ray.frequency_ablation import DIRECT, HORIZONS, PLANS, RECEIVERS, ROLLED, regions_of

CLASS_FILE = "evaluation/results/bin_periodicity_classes.csv"
CLASS_SETS = ("all", "P_only", "N_only", "noise_only")
DISTURBANCES = ("shift96", "shift192", "shift144", "shift210", "level")
REGION_CONDITIONS = ("real", "all_shift96", "all_shift210")      # the conditions kept per region (the existing ablation's numbers)
MAIN_MIN_START = 96                                              # lookback inside the test week
NOWRAP_MIN_START = 210
DEFAULT_PREFIX = "frequency-ablation-2026-11-classes"


def out_base(prefix: str, entry: dict[str, Any]) -> str:
    base = f"{prefix.strip('/')}/{entry['model']}/{entry['band']}/seed-{entry['seed']}"
    return base + (f"/h{entry['horizon']}" if entry.get("source") == "direct" else "")


CLASS_KEY = "analysis-inputs/bin_periodicity_classes.csv"      # copy of evaluation/results/bin_periodicity_classes.csv (code sync does not ship evaluation/results)


def class_table(root: str, band: str, client=None, bucket: str = "spectrum") -> dict[str, np.ndarray]:
    """receiver -> array (200,) of classes P / N / middle / noise (local file if present, else the S3 copy)."""
    import pandas as pd

    local = Path(root) / CLASS_FILE
    if local.exists():
        frame = pd.read_csv(local)
    else:
        frame = pd.read_csv(io.BytesIO(client.get_object(Bucket=bucket, Key=CLASS_KEY)["Body"].read()))
    frame = frame[frame["band"] == band]
    out = {}
    for receiver, part in frame.groupby("receiver"):
        out[receiver] = part.sort_values("bin")["class"].to_numpy(dtype=object)
    return out


def conditions() -> list[tuple[str, str, str]]:
    return [("real", "real", "real")] + [(f"{s}_{d}", s, d) for s in CLASS_SETS for d in DISTURBANCES]


def stored_per_bin(client, bucket: str, entry: dict[str, Any]):
    import pandas as pd

    direct = entry.get("source") == "direct"
    key = (f"rescored/per-bin/direct/15min/{entry['band']}/2d/{entry['model']}/h{entry['horizon']}/seed-{entry['seed']}.csv" if direct
           else f"rescored/per-bin/p15tuned/{entry['band']}/2d/{entry['model']}/seed-{entry['seed']}.csv")
    try:
        return pd.read_csv(io.BytesIO(client.get_object(Bucket=bucket, Key=key)["Body"].read()))
    except Exception:
        return None


def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one checkpoint, four receivers; conditions mode or probe mode."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    entry = spec["entry"]
    band, label, seed = entry["band"], entry["model"], int(entry["seed"])
    name = entry.get("pipeline", label)
    direct = entry.get("source") == "direct"
    horizons = [int(entry["horizon"])] if direct else list(HORIZONS)
    probe = bool(args.get("probe"))
    base = out_base(args["prefix"] + ("/_probe" if probe else ""), entry)
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    record: dict[str, Any] = {"kind": "frequency_ablation_classes" + ("_probe" if probe else ""), "band": band, "model": label, "seed": seed,
                              "source": entry.get("source"), "trained_horizon": horizons[0] if direct else None, "horizons": horizons,
                              "checkpoint_key": entry["pt_key"], "code_revision": args.get("code_revision")}
    started = time.time()
    try:
        import pandas as pd
        import torch

        torch.set_num_threads(max(1, int(args.get("threads", 2))))
        from training.common import transfer_2d as module
        from training.common.evaluation_integrated import calculate_errors_and_export_arrays, evaluation_parameters
        from training.common.forecasting import forecast
        from training.common.frequency_shift import (band_frequencies, class_bin_sums, disturb_series, disturbed_mask, level_windows, region_mask)
        from training.common.model_factory import build_model
        from training.common.windowing import make_window_batch_array
        from training.ray import transfer15 as t15
        from training.ray.powder15 import SPATIAL_NODES, ensure_powder15

        ensure_powder15(band, root, SPATIAL_NODES)
        meta = json.loads(client.get_object(Bucket=bucket, Key=entry["meta_key"])["Body"].read())
        config = meta["config"]
        config["windowing"] = {"lookback": t15.LOOKBACK, "horizons": horizons}
        config[name].setdefault("train", {})["test_stride"] = int(args.get("stride", 1))
        regions = regions_of(root, band)
        if args.get("max_regions"):
            regions = regions[:int(args["max_regions"])]
        receivers = list(args.get("receivers") or RECEIVERS)
        frequencies = band_frequencies(band)
        masks = {r["band_id"]: region_mask(frequencies, r["start_mhz"], r["end_mhz"]) for r in regions}
        classes = class_table(root, band, client, bucket)
        cond_list = [c for c in conditions() if not args.get("only_conditions") or c[0] in args["only_conditions"] or c[0] == "real"]
        collected: dict[str, np.ndarray] = {}
        with tempfile.TemporaryDirectory(prefix="freq-cls-") as scratch_dir:
            path = Path(scratch_dir) / Path(entry["pt_key"]).name
            client.download_file(bucket, entry["pt_key"], str(path))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            normalization = checkpoint["normalization"]
            parameters = evaluation_parameters(config, name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = None

            def predict(source, data, starts, level=None):
                """Frozen forecasts (dB) at ``starts`` from ``data``; ``level`` = bins replaced by their window mean (after windowing)."""
                normalized = module.apply_source_normalization(data, normalization)
                nonlocal model
                if model is None:
                    model = build_model(name, config, normalized)
                    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    model.to(device).eval()
                    torch.manual_seed(seed)
                windows = make_window_batch_array(full_x=normalized, start_rows=starts, lookback=parameters["lookback"])
                if level is not None and level.any():
                    windows = level_windows(windows, level)
                parts = {h: [] for h in parameters["horizons"]}
                with torch.no_grad():
                    for offset in range(0, len(windows), parameters["batch_size"]):
                        values = torch.from_numpy(windows[offset:offset + parameters["batch_size"]]).float().to(device)
                        predicted = forecast(model=model, x=values, prediction_horizon=parameters["prediction_horizon"],
                                             rollout_horizon=parameters["max_horizon"], targets=None)
                        for h in parts:
                            parts[h].append(predicted[:, h - 1].cpu().numpy())
                out = {}
                n_bins = data.shape[1]
                for h, chunk in parts.items():
                    rows = starts + parameters["lookback"] + h - 1
                    valid = rows < len(source.data)                      # origins added by the probe may lack the target row of a long horizon
                    pred_db = np.full((len(starts), n_bins), np.nan)
                    target_db = np.full((len(starts), n_bins), np.nan)
                    if valid.any():
                        p_db, t_db, _, _ = calculate_errors_and_export_arrays(
                            prediction_normalized=np.concatenate(chunk)[valid], target_raw_model_layout=source.data[rows[valid]],
                            normalization=dict(normalization))
                        pred_db[valid], target_db[valid] = np.asarray(p_db, dtype=np.float64), np.asarray(t_db, dtype=np.float64)
                    out[h] = (pred_db, target_db)
                return out

            probe_rows: dict[str, Any] = {}
            for receiver in receivers:
                task = t15.target_task(band, receiver)
                source = module.load_target(task, [t15.target_file(root, band, receiver)], impute=False, max_missing_gap=0)
                starts = module.target_window_starts(source.timestamps, source.segments, lookback=parameters["lookback"],
                                                     max_horizon=max(parameters["max_horizon"], 48), stride=parameters["test_stride"],
                                                     target_start=task.target_start, target_end=task.target_end)
                if not len(starts):
                    raise ValueError("no origins")
                data = np.asarray(source.data, dtype=np.float32)
                starts = np.asarray(starts)
                collected[f"times|{receiver}"] = np.asarray((source.timestamps[starts + parameters["lookback"] - 1]).values, dtype="datetime64[ns]").astype(str)
                collected[f"starts|{receiver}"] = starts.astype(np.int64)
                real = predict(source, data, starts)
                err_real = {h: np.abs(real[h][0] - real[h][1]) for h in real}
                main = starts >= MAIN_MIN_START
                nowrap = starts >= NOWRAP_MIN_START
                collected[f"binmae_main|{receiver}"] = np.stack([err_real[h][main].mean(axis=0) for h in horizons])
                collected[f"binmae_all|{receiver}"] = np.stack([err_real[h].mean(axis=0) for h in horizons])
                collected[f"allbin|{receiver}"] = np.stack([err_real[h].mean(axis=1) for h in horizons], axis=1).astype(np.float32)     # per-origin MAE over all 200 bins
                if probe:
                    # origin-set variants for the identity check: (A) main, (B) all, (C) per-horizon-valid origins (windows whose own horizon target exists),
                    # (D) main plus the origin whose last target row is the end of the test range
                    variants: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
                    variants["A_main"] = {h: (err_real[h][main].sum(axis=0), np.full(200, main.sum())) for h in horizons}
                    variants["B_all"] = {h: (err_real[h].sum(axis=0), np.full(200, len(starts))) for h in horizons}
                    probe_rows[receiver] = {"n_origins": int(len(starts)), "n_main": int(main.sum())}
                    ext = {}
                    for h in horizons:
                        sh = module.target_window_starts(source.timestamps, source.segments, lookback=parameters["lookback"], max_horizon=h,
                                                         stride=parameters["test_stride"], target_start=task.target_start, target_end=task.target_end)
                        sh = np.asarray(sh)
                        ext[h] = sh
                        probe_rows[receiver][f"n_valid_h{h}"] = int(len(sh))
                    extra_starts = np.unique(np.concatenate([ext[h] for h in horizons]))
                    if len(extra_starts):
                        real_ext = predict(source, data, extra_starts)
                        variants["C_valid_per_horizon"] = {}
                        for h in horizons:
                            sel = np.isin(extra_starts, ext[h]) & (extra_starts >= MAIN_MIN_START)
                            e = np.abs(real_ext[h][0] - real_ext[h][1])[sel]
                            variants["C_valid_per_horizon"][h] = (e.sum(axis=0), np.full(200, int(sel.sum())))
                            probe_rows[receiver][f"n_C_h{h}"] = int(sel.sum())
                    import pandas as pd  # noqa: F811

                    d_starts = np.asarray(module.target_window_starts(
                        source.timestamps, source.segments, lookback=parameters["lookback"], max_horizon=max(parameters["max_horizon"], 48),
                        stride=parameters["test_stride"], target_start=task.target_start, target_end=task.target_end + pd.Timedelta(minutes=15)))
                    probe_rows[receiver]["n_D_end_inclusive"] = int(len(d_starts))
                    probe_rows[receiver]["last_timestamp"] = str(source.timestamps[-1])
                    probe_rows[receiver]["target_end"] = str(task.target_end)
                    if len(d_starts):
                        real_d = predict(source, data, d_starts)
                        sel = d_starts >= MAIN_MIN_START
                        variants["D_end_inclusive"] = {h: (np.abs(real_d[h][0] - real_d[h][1])[sel].sum(axis=0), np.full(200, int(sel.sum()))) for h in horizons}
                    for vname, per_h in variants.items():
                        for h, (s, n) in per_h.items():
                            collected[f"probe|{vname}|{receiver}|h{h}|sum"] = np.asarray(s)
                            collected[f"probe|{vname}|{receiver}|h{h}|n"] = np.asarray(n)
                    continue
                # --- conditions
                cls_arrays = {c: {k: np.zeros((len(starts), len(horizons)), dtype=np.float32) for k in ("P", "N", "M")} for c, _, _ in cond_list}
                clsr_arrays = {c: {k: np.zeros((len(starts), len(horizons)), dtype=np.float32) for k in ("P", "N", "M")} for c, _, _ in cond_list}
                cnt_arrays = {c: {k: 0 for k in ("P", "N", "M")} for c, _, _ in cond_list}       # target bins of the regions that contribute to the condition
                bin_main = {c: np.full((len(horizons), 200), np.nan, dtype=np.float32) for c, _, _ in cond_list}
                bin_nw = {c: np.full((len(horizons), 200), np.nan, dtype=np.float32) for c, _, _ in cond_list}
                skipped: dict[str, list[str]] = {}
                targets = {h: real[h][1] for h in real}
                for region_id, rmask in masks.items():
                    for cname, cset, dist in cond_list:
                        if cname == "real":
                            errs = err_real
                        else:
                            dmask = disturbed_mask(rmask, classes[receiver], cset)
                            if not dmask.any():
                                skipped.setdefault(cname, []).append(region_id)
                                continue
                            if dist == "level":
                                shifted = predict(source, data, starts, level=dmask)
                            else:
                                shifted = predict(source, disturb_series(data, dmask, dist), starts)
                            errs = {h: np.abs(shifted[h][0] - targets[h]) for h in shifted}
                        for key, short in (("P", "P"), ("N", "N"), ("middle", "M")):
                            cnt_arrays[cname][short] += int((rmask & (classes[receiver] == key)).sum())
                        for j, h in enumerate(horizons):
                            sums = class_bin_sums(errs[h], rmask, classes[receiver])
                            real_sums = class_bin_sums(err_real[h], rmask, classes[receiver])
                            for key, short in (("P", "P"), ("N", "N"), ("middle", "M")):
                                cls_arrays[cname][short][:, j] += sums[key].astype(np.float32)
                                clsr_arrays[cname][short][:, j] += real_sums[key].astype(np.float32)
                            bin_main[cname][j, rmask] = errs[h][main][:, rmask].mean(axis=0)
                            bin_nw[cname][j, rmask] = errs[h][nowrap][:, rmask].mean(axis=0) if nowrap.any() else np.nan
                        if cname in REGION_CONDITIONS:
                            collected[f"rmae|{receiver}|{region_id}|{cname}"] = np.stack(
                                [errs[h][:, rmask].mean(axis=1) for h in horizons], axis=1).astype(np.float32)
                for cname, _, _ in cond_list:
                    for short in ("P", "N", "M"):
                        collected[f"cls|{receiver}|{cname}|{short}"] = cls_arrays[cname][short]
                        collected[f"clsr|{receiver}|{cname}|{short}"] = clsr_arrays[cname][short]
                    collected[f"cnt|{receiver}|{cname}"] = np.asarray([cnt_arrays[cname][k] for k in ("P", "N", "M")], dtype=np.int32)
                    collected[f"bin|{receiver}|{cname}"] = bin_main[cname]
                    collected[f"binnw|{receiver}|{cname}"] = bin_nw[cname]
                if skipped:
                    record.setdefault("skipped_empty_sets", {})[receiver] = {k: len(v) for k, v in skipped.items()}
            buffer = io.BytesIO()
            np.savez_compressed(buffer, horizons=np.asarray(horizons), regions=np.asarray(list(masks)), receivers=np.asarray(receivers),
                                conditions=np.asarray([c[0] for c in cond_list]), **collected)
            client.put_object(Bucket=bucket, Key=f"{base}/per_origin.npz", Body=buffer.getvalue())
        if probe:
            stored = stored_per_bin(client, bucket, entry)
            report: dict[str, Any] = {"receivers": probe_rows}
            if stored is not None:
                import pandas as pd  # noqa: F811

                for vname in ("A_main", "B_all", "C_valid_per_horizon", "D_end_inclusive"):
                    rows = []
                    for h in horizons:
                        key = f"probe|{vname}|{receivers[0]}|h{h}|sum"
                        if key not in collected:
                            continue
                        total = sum(collected[f"probe|{vname}|{r}|h{h}|sum"] for r in receivers)
                        count = sum(collected[f"probe|{vname}|{r}|h{h}|n"] for r in receivers)
                        mine = total / count
                        st = stored[stored["horizon"] == h].sort_values("frequency_mhz")["mae_db"].to_numpy()
                        if len(st) == len(mine):
                            d = np.abs(st - mine)
                            rows.append({"horizon": h, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "n_windows_total": int(count[0])})
                    report[vname] = rows
                report["stored_n_test_windows"] = None
            record["probe"] = report
        record.update(n_origins=int(len(collected[f"times|{receivers[0]}"])), ok=True)
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(record, default=str).encode())
    return record


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800"], choices=list(PLANS))
    p.add_argument("--models", nargs="+", default=["revinmlp", "vanillalstm"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS), help="trained horizons (direct source: one checkpoint each)")
    p.add_argument("--receivers", nargs="+", default=list(RECEIVERS), choices=list(RECEIVERS))
    p.add_argument("--max-regions", type=int, default=0, help="smoke tests only (use another --prefix)")
    p.add_argument("--only-conditions", nargs="+", help="smoke tests only: restrict the conditions (real is always kept)")
    p.add_argument("--probe", action="store_true", help="identity probe: compare real per-bin errors with the stored table for origin-set variants")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--gpu-fraction", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--memory-gb", type=float, default=8.0)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    try:
        revision = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        revision = "unknown"
    rolled = [m for m in args.models if m in ROLLED]
    direct = [m for m in args.models if m in DIRECT]
    unknown = [m for m in args.models if m not in ROLLED and m not in DIRECT]
    if unknown:
        raise SystemExit(f"unknown models {unknown}")
    entries = []
    if rolled:
        entries += [e for e in discover_sources(client, bucket, ["p15tuned"], rolled)
                    if e["band"] in args.bands and e["rep"] == "2d" and int(e["seed"]) in args.seeds]
    if direct:
        entries += [e for e in discover_sources(client, bucket, ["direct"], direct)
                    if e["resolution"] == "15min" and e["band"] in args.bands and int(e["seed"]) in args.seeds and int(e["horizon"]) in args.horizons]
    jobs = [{"entry": e} for e in entries]
    prefix = args.prefix + ("/_probe" if args.probe else "")
    if args.skip_done:
        def done(j):
            try:
                return bool(json.loads(client.get_object(Bucket=bucket, Key=out_base(prefix, j["entry"]) + "/summary.json")["Body"].read()).get("ok"))
            except Exception:
                return False
        jobs = [j for j in jobs if not done(j)]
    print(f"[freq-classes] {len(jobs)} jobs (of {len(entries)} checkpoints), revision {revision[:12]}{' PROBE' if args.probe else ''}", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "code_revision": revision, "threads": args.threads,
              "stride": args.stride, "max_regions": args.max_regions, "receivers": args.receivers, "probe": args.probe,
              "only_conditions": args.only_conditions}
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=2, max_retries=1)(job)
    queue, pending, ok = list(jobs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            pending.append(remote.options(num_gpus=args.gpu_fraction, memory=int(args.memory_gb * 2 ** 30)).remote(queue.pop(0), shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        extra = ""
        if row.get("probe"):
            pr = row["probe"]
            extra = " PROBE " + json.dumps({k: pr[k] for k in pr if k.startswith(("A_", "B_", "C_", "D_")) or k == "receivers"}, default=str)[:600]
        print(f"[freq-classes] {row.get('band')} {row.get('model')} seed{row.get('seed')} h{row.get('trained_horizon')} ok={row.get('ok')} "
              f"{row.get('error', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]{extra}", flush=True)
    print(f"[freq-classes] done: {ok} ok", flush=True)


if __name__ == "__main__":
    main()
