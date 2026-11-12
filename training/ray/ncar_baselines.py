"""Item 1 of the NCAR transfer batch: simple baselines on every transfer target, on the windows every model cell uses.

Windows: ``transfer_all.prepare_target`` (task and converted file), ``transfer_2d.load_target(task, [file], impute=False, max_missing_gap=0)``
and ``target_window_starts(lookback=96, max_horizon=48, stride=1, target_start, target_end)``, i.e. ``transfer15.evaluate_frozen``'s origins.
Horizons 1, 4, 12, 24, 48 (15-minute steps).

Baselines (all per bin, in dB):
  persistence           the last observed value;
  lookbackmean24h       the mean of the 96-step (24 h) window, constant over horizons (as the stored transfer baselines);
  same_time_yesterday   the value 96 steps (24 h) before the TARGET time (row ``origin + h - 96``, inside the lookback window);
  tuned_median          the lookback median with the per-horizon window K selected on POWDER validation for the 15-minute task
                        (``median-study/selected_k.json``, ``15min/<band>`` ``k_per_horizon``; ``evaluation/results/baselines_all_horizons.csv``
                        rows ``median, per_horizon``), rolled out step by step exactly as the pipeline's one-step ``lookbackmean2d``
                        (statistic ``median``, lower median for an even count, window slides with the predictions appended).
                        Same-band targets use their band's K; the cross-band targets (800-1000, 5725-5925 MHz) have no own selection
                        and get both source bands' K (``tuned_median_k600_800``, ``tuned_median_k2400_2600``).

Outputs: ``transfer-ncar-2026-11/item1/<target_key>/<baseline>/{summary.json, forecasts.npz, per_bin.csv}`` (``ncar_cells`` formats).

    python -m training.ray.ncar_baselines run --skip-done
    python -m training.ray.ncar_baselines table      # transfer_baselines.csv, index and the verification against stored baselines
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREFIX = "transfer-ncar-2026-11/item1"
LOOKBACK, HORIZONS = 96, (1, 4, 12, 24, 48)
SELECTED_K_KEY = "median-study/selected_k.json"
CORE4_NODES = {"humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1"}
METRICS = ("mae_db", "mse_db2", "rmse_db", "mape_linear_pct", "mape_dbm_pct")


# ----------------------------------------------------------------------------------------- targets
def all_targets(root: str) -> list[dict[str, Any]]:
    """Every target with its group name; the crossband table's bands are registered by ``transfer_crossband`` at import."""
    from training.ray import ncar_cells as nc
    from training.ray import transfer_all as ta
    from training.ray import transfer_crossband as tc  # noqa: F401  (registers 800_1000 and 5725_5925)

    bands = ["600_800", "2400_2600"]
    out = []
    for t in ta.targets(root, ["powder"], bands):                                       # October guesthouse, law73
        out.append({**t, "group": "heldout_oct"})
    for t in ta.targets(root, ["cosmos", "aerpaw", "ara"], bands, ta.EXTERNAL_INTERVALS, core=False):
        out.append({**t, "group": t["platform"]})
    for t in ta.targets(root, ["powder"], bands, ta.JULY_INTERVALS, core=False):
        out.append({**t, "group": "july_core4" if t["site"] in CORE4_NODES else "july_other"})
    for t in ta.targets(root, ["powder"], list(tc.CROSS_BANDS), tc.INTERVALS_CSV, core=False):
        out.append({**t, "group": f"crossband_{t['band']}"})
    out += [{**t, "group": "core4_oct"} for t in nc.core_targets(bands, "oct")]
    out += [{**t, "group": "core4_week2"} for t in nc.core_targets(bands, "week2")]
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in t.items()} for t in out]


def receiver_of(t: dict[str, Any]) -> str:
    return t["site"].replace("-nuc1", "")


# ----------------------------------------------------------------------------------------- forecasts
def rolled_median(windows: np.ndarray, k: int, steps: int) -> np.ndarray:
    """(N, steps, F) rolled-out lookback median over the newest ``k`` steps (torch lower median, as ``LookbackMeanForecaster``)."""
    import torch

    buf = torch.from_numpy(np.ascontiguousarray(windows[:, -k:], dtype=np.float32))
    out = []
    for _ in range(steps):
        nxt = buf.median(dim=1, keepdim=True).values
        out.append(nxt)
        buf = torch.cat([buf[:, 1:], nxt], dim=1)
    return torch.cat(out, dim=1).numpy()


def baseline_forecasts(data: np.ndarray, starts: np.ndarray, k_sets: dict[str, dict[int, int]]) -> dict[str, dict[int, np.ndarray]]:
    """name -> horizon -> (N, F) predictions."""
    idx = starts[:, None] + np.arange(LOOKBACK)[None, :]
    windows = data[idx]                                                     # (N, 96, F)
    out: dict[str, dict[int, np.ndarray]] = {
        "persistence": {h: windows[:, -1] for h in HORIZONS},
        "lookbackmean24h": {h: windows.mean(axis=1) for h in HORIZONS},
        "same_time_yesterday": {h: data[starts + LOOKBACK + h - 1 - 96] for h in HORIZONS}}
    cache: dict[int, np.ndarray] = {}
    for name, k_per_h in k_sets.items():
        out[name] = {}
        for h in HORIZONS:
            if h not in k_per_h:              # the validation selection covers 4/12/24/48 only (median_study SELECT_HORIZONS)
                continue
            k = int(k_per_h[h])
            if k not in cache:
                cache[k] = rolled_median(windows, k, max(HORIZONS))
            out[name][h] = cache[k][:, h - 1]
    return out


def selected_k(client, bucket: str) -> dict[str, dict[int, int]]:
    raw = json.loads(client.get_object(Bucket=bucket, Key=SELECTED_K_KEY)["Body"].read())
    return {band: {int(h): int(k) for h, k in raw[f"15min/{band}"]["k_per_horizon"].items()} for band in ("600_800", "2400_2600")}


# ------------------------------------------------------------------------------------------- worker
def cell_base(t: dict[str, Any], name: str) -> str:
    from training.ray import transfer_all as ta

    return f"{PREFIX}/{ta.target_key(t)}/{name}"


def job(t: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray import transfer_crossband as tc

    tc.register_bands()
    from training.common import transfer_2d
    from training.ray import ncar_cells as nc
    from training.ray import transfer15 as t15
    from training.ray import transfer_all as ta
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    record: dict[str, Any] = {"target": ta.target_key(t), "group": t["group"]}
    started = time.time()
    try:
        import torch

        torch.set_num_threads(2)
        task, file = ta.prepare_target(root, t)
        source = transfer_2d.load_target(task, [file], impute=False, max_missing_gap=0)
        starts = transfer_2d.target_window_starts(source.timestamps, source.segments, lookback=LOOKBACK, max_horizon=max(HORIZONS), stride=1,
                                                  target_start=task.target_start, target_end=task.target_end)
        if not len(starts):
            raise ValueError("no origins")
        if t.get("n_origins") and len(starts) != int(t["n_origins"]):
            raise RuntimeError(f"origin count {len(starts)} differs from the frozen {t['n_origins']}")
        data = np.asarray(source.data, dtype=np.float32)
        ks = args["selected_k"]
        k_sets = {"tuned_median": ks[t["band"]]} if t["band"] in ks else {f"tuned_median_k{b}": ks[b] for b in ks}
        preds = baseline_forecasts(data, starts, k_sets)
        freqs = [float(f) for f in source.frequencies] if getattr(source, "frequencies", None) is not None else nc.band_frequencies(t["band"])
        record["n_windows"] = int(len(starts))
        record["mae"] = {}
        for name, by_h in preds.items():
            mats, predictions, targets, bins, times = {}, {}, {}, {}, {}
            for h in sorted(by_h):
                rows = starts + LOOKBACK + h - 1
                origin = t15.origin_times(source, rows, h)
                mats[h] = {"origin": origin.astype("datetime64[s]").astype(np.int64), "pred": by_h[h].astype(np.float32), "target": data[rows]}
                predictions[h], targets[h], bins[h], times[h] = by_h[h], data[rows], np.empty(0), origin
            scored = t15.score_all(predictions, targets, bins, times)
            base = cell_base(t, name)
            nc.save_cell(client, bucket, base, mats, freqs)
            summary = {"kind": "ncar_item1_baseline", "baseline": name, "target": ta.target_key(t), "group": t["group"],
                       "platform": t["platform"], "site": t["site"], "receiver": receiver_of(t), "run": t["run"], "band": t["band"],
                       "window": t.get("window") or ("oct" if t["group"] in ("heldout_oct", "core4_oct") else "frozen_interval"),
                       "target_start": str(task.target_start), "target_end": str(task.target_end), "n_windows": int(len(starts)),
                       "k_per_horizon": {str(h): int(v) for h, v in k_sets[name].items()} if name in k_sets else None,
                       "k_source": SELECTED_K_KEY if name in k_sets else None, "lookback": LOOKBACK, "horizons": sorted(by_h),
                       "code_revision": args["code_revision"], "metrics": scored["metrics"]}
            client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(summary, default=str, indent=1).encode())
            record["mae"][name] = {str(h): round(scored["metrics"][str(h)]["mae_db"], 4) for h in (4, 48)}
            del mats, predictions, targets
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
        try:
            client.put_object(Bucket=bucket, Key=f"{PREFIX}/errors/{ta.target_key(t)}.json", Body=json.dumps(record, default=str).encode())
        except Exception:
            pass
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


# -------------------------------------------------------------------------------------------- driver
def exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def cmd_run(a: argparse.Namespace) -> None:
    import ray

    from training.ray.storage import spectrum_s3_client

    bucket = a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    revision = a.code_revision
    if not revision:
        try:
            revision = subprocess.run(["git", "-C", a.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            revision = "unknown"
    tgts = all_targets(a.repo_root)
    if a.groups:
        tgts = [t for t in tgts if t["group"] in a.groups]
    if a.targets:
        from training.ray import transfer_all as ta

        tgts = [t for t in tgts if any(s in ta.target_key(t) for s in a.targets)]
    if a.skip_done:   # the last baseline written per target is the persistence... check all four names via the summary of lookbackmean24h and a median
        def done(t):
            names = ["persistence", "lookbackmean24h", "same_time_yesterday",
                     "tuned_median" if t["band"] in ("600_800", "2400_2600") else "tuned_median_k2400_2600"]
            return all(exists(client, bucket, f"{cell_base(t, n)}/summary.json") for n in names)
        tgts = [t for t in tgts if not done(t)]
    shared = {"repo_root": a.repo_root, "bucket": bucket, "code_revision": revision, "selected_k": selected_k(client, bucket)}
    print(f"[item1] {len(tgts)} targets, selected K {shared['selected_k']}, revision {revision[:12]}", flush=True)
    ray.init(address=a.address, ignore_reinit_error=True)
    queue, pending, ok, failed = list(tgts), {}, 0, 0
    remote = ray.remote(num_cpus=2, num_gpus=0, memory=int(a.memory_gb * 2 ** 30), max_retries=1)(job)
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            t = queue.pop(0)
            pending[remote.remote(t, shared)] = t
        finished, _ = ray.wait(list(pending), num_returns=1)
        t = pending.pop(finished[0])
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "target": str(t), "error": f"{type(error).__name__}: {str(error)[:300]}"}
        ok += bool(row.get("ok"))
        failed += not row.get("ok")
        print(f"[item1] {row.get('target')} ok={row.get('ok')} n={row.get('n_windows')} {row.get('mae')} {row.get('error', '')} "
              f"({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[item1] done: {ok} ok, {failed} failed", flush=True)


# --------------------------------------------------------------------------------------------- table
def cmd_table(a: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from training.ray.storage import spectrum_s3_client

    bucket = a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    keys = [o["Key"] for p in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{PREFIX}/")
            for o in p.get("Contents", []) if o["Key"].endswith("/summary.json") and "/errors/" not in o["Key"] and "/tables/" not in o["Key"]]

    def read(k):
        return k, json.loads(client.get_object(Bucket=bucket, Key=k)["Body"].read())

    with ThreadPoolExecutor(32) as pool:
        summaries = list(pool.map(read, keys))
    rows, index = [], []
    for k, d in summaries:
        base = k.rsplit("/", 1)[0]
        index.append({"group": d["group"], "target": d["target"], "receiver": d["receiver"], "band": d["band"], "baseline": d["baseline"],
                      "n_windows": d["n_windows"], "per_bin_key": f"{base}/per_bin.csv", "forecasts_key": f"{base}/forecasts.npz"})
        for h, m in d["metrics"].items():
            rows.append({"level": "receiver", "group": d["group"], "target": d["target"], "receiver": d["receiver"], "band": d["band"],
                         "baseline": d["baseline"], "horizon": int(h), "n_windows": d["n_windows"],
                         "k": (d.get("k_per_horizon") or {}).get(h), **{m_: m[m_] for m_ in METRICS}})
    recv = pd.DataFrame(rows).sort_values(["group", "band", "baseline", "horizon", "target"])
    grp = (recv.groupby(["group", "band", "baseline", "horizon"])
           .agg(n_receivers=("target", "nunique"), n_windows=("n_windows", "sum"), k=("k", "first"), **{m: (m, "mean") for m in METRICS})
           .reset_index().assign(level="group", target="", receiver="mean over receivers"))
    table = pd.concat([recv.assign(n_receivers=1), grp], ignore_index=True)[
        ["level", "group", "band", "baseline", "horizon", "target", "receiver", "n_receivers", "n_windows", "k", *METRICS]]
    # mean over horizons 4/12/24/48 as an extra row (horizon = "mean_4h")
    h4 = table[table.horizon.isin([4, 12, 24, 48])]
    m4 = (h4.groupby(["level", "group", "band", "baseline", "target", "receiver"], dropna=False)
          .agg(n_receivers=("n_receivers", "first"), n_windows=("n_windows", "first"), **{m: (m, "mean") for m in METRICS}).reset_index()
          .assign(horizon="mean_4h", k=None))
    table = pd.concat([table.astype({"horizon": object}), m4], ignore_index=True)
    out = Path(a.repo_root) / "runs" / "ncar-tx" if a.local_out is None else Path(a.local_out)
    payload = {"transfer_baselines.csv": table.to_csv(index=False),
               "transfer_baselines_per_bin_index.csv": pd.DataFrame(index).sort_values(["group", "target", "baseline"]).to_csv(index=False),
               "transfer_baselines_verification.csv": verify(client, bucket, summaries).to_csv(index=False)}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for name, text in payload.items():
        client.put_object(Bucket=bucket, Key=f"{PREFIX}/tables/{stamp}/{name}", Body=text.encode())
        print(f"[item1] wrote s3://{bucket}/{PREFIX}/tables/{stamp}/{name}", flush=True)
    pd.set_option("display.width", 250)
    g = table[(table.level == "group") & (table.horizon == "mean_4h")]
    print(g.pivot_table(index=["group", "band"], columns="baseline", values="mae_db").round(3).to_string())


def verify(client, bucket: str, summaries) -> pd.DataFrame:
    """Stored persistence / lookback mean (transfer15, transfer15-external, transfer15-july) and stored model origins, against ours."""
    rows = []
    for k, d in summaries:
        if d["baseline"] not in ("persistence", "lookbackmean24h"):
            continue
        stored_name = "persistence" if d["baseline"] == "persistence" else "lookbackmean"
        tk = d["target"]
        platform, site, run, band = tk.split("/")
        if d["group"] in ("heldout_oct", "core4_oct"):
            cands = [f"transfer15-oct2026/{band}/{site}/baseline/{stored_name}/summary.json"]
        elif d["group"] in ("aerpaw", "ara", "cosmos"):
            cands = [f"transfer15-external-oct2026/baseline/{tk}/{stored_name}/summary.json"]
        elif d["group"].startswith("july"):
            cands = [f"transfer15-july-2026/baseline/{tk}/{stored_name}/summary.json"]
        elif d["group"].startswith("crossband"):
            cands = [f"transfer-crossband-july-2026/baseline/{tk}/{stored_name}/summary.json"]
        else:
            cands = []
        for c in cands:
            try:
                s = json.loads(client.get_object(Bucket=bucket, Key=c)["Body"].read())
            except Exception:
                rows.append({"check": "baseline", "group": d["group"], "target": tk, "baseline": d["baseline"], "stored_key": c, "found": False})
                continue
            for h, m in s["metrics"].items():
                if h in d["metrics"]:
                    rows.append({"check": "baseline", "group": d["group"], "target": tk, "baseline": d["baseline"], "stored_key": c, "found": True,
                                 "horizon": int(h), "stored_mae": m["mae_db"], "ncar_mae": d["metrics"][h]["mae_db"],
                                 "diff": d["metrics"][h]["mae_db"] - m["mae_db"], "stored_n_origins": s.get("n_origins"), "ncar_n_windows": d["n_windows"]})
    # origin sets: our persistence origins against one stored model cell per group (revinmlp h4 seed 40, frozen)
    seen = set()
    for k, d in summaries:
        if d["baseline"] != "persistence" or d["group"] in seen:
            continue
        tk = d["target"]
        if d["group"] == "core4_week2":             # no stored transfer cell exists for week 2
            continue
        if d["group"].startswith("crossband"):
            cand = f"transfer-crossband-july-2026/from-600_800/A/{tk}/2d/revinmlp/h4/seed-40/per_origin.csv"
        elif d["group"].startswith("july"):
            cand = f"transfer-all-july-2026/A/{tk}/2d/revinmlp/h4/seed-40/per_origin.csv"
        else:
            cand = f"transfer-all-oct2026/A/{tk}/2d/revinmlp/h4/seed-40/per_origin.csv"
        try:
            po = pd.read_csv(io.BytesIO(client.get_object(Bucket=bucket, Key=cand)["Body"].read()))
        except Exception:
            rows.append({"check": "origins", "group": d["group"], "target": tk, "stored_key": cand, "found": False})
            continue
        from training.ray import ncar_cells as nc

        mats = nc.load_npz(client.get_object(Bucket=bucket, Key=k.rsplit("/", 1)[0] + "/forecasts.npz")["Body"].read())
        ours = pd.to_datetime(mats[4]["origin"], unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
        theirs = po[po.horizon == 4]["time"].astype(str)
        rows.append({"check": "origins", "group": d["group"], "target": tk, "stored_key": cand, "found": True, "horizon": 4,
                     "stored_n_origins": int(len(theirs)), "ncar_n_windows": int(len(ours)), "origins_equal": set(ours) == set(theirs)})
        seen.add(d["group"])
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--groups", nargs="+")
    r.add_argument("--targets", nargs="+")
    r.add_argument("--skip-done", action="store_true")
    r.add_argument("--max-parallel", type=int, default=8)
    r.add_argument("--memory-gb", type=float, default=6.0)
    r.add_argument("--code-revision", default=None)
    t = sub.add_parser("table")
    t.add_argument("--local-out", default=None)
    for s in (r, t):
        s.add_argument("--bucket", default=None)
        s.add_argument("--address", default="auto")
        s.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    os.chdir(a.repo_root)
    {"run": cmd_run, "table": cmd_table}[a.command](a)


if __name__ == "__main__":
    main()
