"""NCAR transfer batch, item 6: the tuned 15-minute TSS-LCD finals scored, without training, on shifted targets (``transfer-ncar-2026-11/item6``).

TSS-LCD is one joint model over 4 receiver positions x 200 bins (800 columns, receiver-major) with a positional encoding over the receiver
slots, so it is order-sensitive: a receiver swap is defined by an ordered assignment of receivers to positions. Reuses the networks, scaler,
sampler and constants of ``training.ray.tsslcd_authors`` (``--eval-only`` semantics: weights ``<source>/weights/savgol-off/seed-N``,
hyperparameters from the source record, ``torch.manual_seed(seed + 1000)`` before sampling, one diffusion sample per window, batches of 256).

Cases
  6a  receiver swap, October test week: guest house and law 73 replace 2 of the 4 core positions, all 12 ordered assignments (6 position
      pairs x 2 orders) plus the unswapped model; every position is scored (the remaining core receivers are the check against ``unswapped``).
  6b  the core four on the June-July runs (06-28 and 07-03), common times of all four; the per-receiver frozen origins (interval table) and the
      common origins are both recorded (windows lost to gaps and to the common range).
  6c  new frequency: the core four at 800-1000 and 5725-5925 MHz of the 07-03 run, checkpoints of both source bands, positional by bin.
Conditions
  frozen  the checkpoint's min-max scaler, fit on the source band's training partition of the core four (per position: a swapped-in receiver
          uses the scaler of the position it occupies);
  refit   min-max refit per column on the 96 blocks (24 h) ending at the first scored origin, with the range floored at 0.25 x the frozen
          range (as the std floor of transfer regime C); frozen weights.
Windows: the benchmark's (``transfer_2d.target_window_starts``): origin = last observed step; a window is kept when its first target is at or
after ``target_start`` and its last target (t+48) before ``target_end``; history 100 blocks (TSS-LCD needs a multiple of its 5 heads, so
100, not 96); no window crosses a gap (receivers joined on common blocks, gaps up to 9 blocks interpolated as in ``tsslcd_authors``).
NB: the stock TSS-LCD 15-minute pipeline's origins are one block later (first origin 20:00, not 19:45, on the October week);
``--origin-convention tsslcd`` reproduces them (used for the step-0c check only).

Outputs: per position ``transfer-ncar-2026-11/item6/<case>/<condition>/<target key>/2d/tsslcd/<variant>/seed-N/{forecasts.npz, per_bin.csv}``
(``ncar_cells`` format) and one record per run ``transfer-ncar-2026-11/item6/records/<case>/<condition>/<variant>/<source band>[-to-<target
band>]/seed-N.json`` (metrics per position and horizon, windows, provenance, peak GPU memory). Resumable (records with ok are skipped).

    python -m training.ray.ncar_tsslcd run --cases 6a 6b 6c --conditions frozen refit --gpu-fraction 0.1 --max-parallel 8
    python -m training.ray.ncar_tsslcd consistency --out-key transfer-ncar-2026-11/step0/consistency_tsslcd.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREFIX = "transfer-ncar-2026-11/item6"
WEIGHTS = {"600_800": "tsslcd-authors-15min-tuned", "2400_2600": "tsslcd-authors-15min-tuned-2400"}
SEEDS = (40, 41, 42, 43, 44)
CORE = ("humanities", "madsen", "moran", "sagepoint")
HELD = ("guesthouse", "law73")
NODE = {r: f"{r}-nuc1" for r in CORE + HELD}
TEST_RUN = "20261025T1355Z"
OCT = (pd.Timestamp("2026-10-27T20:00:00Z"), pd.Timestamp("2026-11-03T20:00:00Z"))
STEP = pd.Timedelta(minutes=15)
CONTEXT, FUTURE, HORIZONS = 100, 48, (1, 4, 12, 24, 48)
MAX_GAP = 9
REFIT_ROWS, REFIT_FLOOR = 96, 0.25
JULY_INTERVALS = "evaluation/analysis/transfer_july_intervals.csv"
CROSS_INTERVALS = "evaluation/analysis/transfer_crossband_intervals.csv"
CROSS_BANDS = ("800_1000", "5725_5925")


# ------------------------------------------------------------------------------------------------ specs
def assignments() -> list[tuple[str, tuple[str, ...]]]:
    """(variant, receivers by position): unswapped plus 12 ordered placements of guest house and law 73."""
    out = [("unswapped", CORE)]
    for i, j in itertools.permutations(range(4), 2):
        recv = list(CORE)
        recv[i], recv[j] = "guesthouse", "law73"
        out.append((f"gh{i}-law{j}", tuple(recv)))
    return out


def oct_position(receiver: str, band: str) -> dict[str, Any]:
    return {"receiver": receiver, "node": NODE[receiver], "file": f"data/powder15/{NODE[receiver]}/{TEST_RUN}/{band}/power_15min.csv",
            "target_key": f"powder/{receiver}/{TEST_RUN}/{band}", "role": "held-out" if receiver in HELD else "core"}


def ext_position(row: dict[str, Any]) -> dict[str, Any]:
    node = row["site"]
    return {"receiver": node.replace("-nuc1", ""), "node": node,
            "file": f"data/ext15/{row['platform']}/{node}/{row['run']}/{row['band']}/power_15min.csv",
            "target_key": f"{row['platform']}/{node}/{row['run']}/{row['band']}", "role": "core",
            "frozen_target_start": row["target_start"], "frozen_target_end": row["target_end"], "frozen_n_origins": int(row["n_origins"])}


def common_group(repo_root: str, table: str, band: str, day: str) -> dict[str, Any] | None:
    t = pd.read_csv(Path(repo_root) / table)
    rows = [r for r in t.to_dict("records") if r["use"] and r["band"] == band and r["run"][:8] == day and r["site"] in {NODE[c] for c in CORE}]
    if len({r["site"] for r in rows}) != 4:
        return None
    rows = sorted(rows, key=lambda r: CORE.index(r["site"].replace("-nuc1", "")))          # source position order
    start = max(pd.Timestamp(r["target_start"]) for r in rows)
    end = min(pd.Timestamp(r["target_end"]) for r in rows)
    return {"positions": [ext_position(r) for r in rows], "target_start": start.isoformat(), "target_end": end.isoformat()}


def build_specs(repo_root: str, cases: list[str], conditions: list[str], seeds: list[int], bands: list[str]) -> list[dict[str, Any]]:
    specs = []
    for cond in conditions:
        for seed in seeds:
            for band in bands:
                if "6a" in cases:
                    for variant, recv in assignments():
                        specs.append({"case": "6a", "condition": cond, "seed": seed, "source_band": band, "target_band": band, "variant": variant,
                                      "positions": [oct_position(r, band) for r in recv],
                                      "target_start": OCT[0].isoformat(), "target_end": OCT[1].isoformat()})
                if "6b" in cases:
                    for day, label in (("20260628", "june"), ("20260703", "july")):
                        g = common_group(repo_root, JULY_INTERVALS, band, day)
                        if g:
                            specs.append({"case": "6b", "condition": cond, "seed": seed, "source_band": band, "target_band": band,
                                          "variant": f"run-{day}", **g})
                if "6c" in cases:
                    for tb in CROSS_BANDS:
                        g = common_group(repo_root, CROSS_INTERVALS, tb, "20260703")
                        if g:
                            specs.append({"case": "6c", "condition": cond, "seed": seed, "source_band": band, "target_band": tb,
                                          "variant": f"from-{band}", **g})
    return specs


def record_key(spec: dict[str, Any]) -> str:
    bands = spec["source_band"] + ("" if spec["target_band"] == spec["source_band"] else f"-to-{spec['target_band']}")
    return f"{PREFIX}/records/{spec['case']}/{spec['condition']}/{spec['variant']}/{bands}/seed-{spec['seed']}.json"


def cell_base(spec: dict[str, Any], position: dict[str, Any]) -> str:
    return f"{PREFIX}/{spec['case']}/{spec['condition']}/{position['target_key']}/2d/tsslcd/{spec['variant']}/seed-{spec['seed']}"


# ------------------------------------------------------------------------------------------------- data
def load_positions(repo_root: str, files: list[str], start: pd.Timestamp, end: pd.Timestamp):
    """(values (T, 4*200) dB, grid, valid) on the 15-minute grid [start, end], receivers joined, gaps <= 9 blocks interpolated."""
    frames = []
    for i, f in enumerate(files):
        raw = pd.read_csv(Path(repo_root) / f)
        stamps = pd.to_datetime(raw.pop("timestamp_utc"), utc=True).dt.floor("15min")
        raw.index = stamps
        raw = raw[~raw.index.duplicated(keep="first")].sort_index()
        if raw.shape[1] != 200:
            raise ValueError(f"{f}: {raw.shape[1]} bins, expected 200")
        raw = raw.loc[(raw.index >= start) & (raw.index <= end)]
        raw.columns = [f"p{i}:{c}" for c in raw.columns]
        frames.append(raw)
    grid = pd.date_range(start, end, freq="15min")
    joined = pd.concat([fr.reindex(grid) for fr in frames], axis=1)
    both = joined.interpolate(method="linear", limit=MAX_GAP, limit_area="inside")
    values = both.to_numpy(dtype=np.float64)
    return values, grid, np.isfinite(values).all(axis=1)


def window_starts(grid: pd.DatetimeIndex, valid: np.ndarray, target_start: pd.Timestamp, target_end: pd.Timestamp) -> np.ndarray:
    from training.ray.tsslcd_authors import contiguous_segments

    length = CONTEXT + FUTURE
    starts = [np.arange(a, b - length + 1) for a, b in contiguous_segments(valid, length)]
    starts = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
    first, last = starts + CONTEXT, starts + CONTEXT + FUTURE - 1
    keep = (grid[first] >= target_start) & (grid[last] < target_end) if len(starts) else np.zeros(0, bool)
    return starts[np.asarray(keep)]


_SCALERS: dict[str, Any] = {}


def frozen_scaler(repo_root: str, band: str):
    """The finals' scaler: min-max per column on the core four's training partition (``tsslcd_authors.build_partition``)."""
    if band not in _SCALERS:
        from training.ray.powder15 import ensure_powder15
        from training.ray.tsslcd_authors import SITES, build_partition

        ensure_powder15(band, repo_root, tuple(SITES))
        _SCALERS[band] = build_partition(repo_root, "train", None, "off", band=band, profile="15min")[3]
    return _SCALERS[band]


def refit_scaler(frozen, values: np.ndarray, origin_row: int):
    from training.ray.tsslcd_authors import MinMax

    rows = values[origin_row - REFIT_ROWS + 1: origin_row + 1]
    s = MinMax()
    s.min = rows.min(axis=0)
    s.scale = np.maximum(rows.max(axis=0) - s.min, REFIT_FLOOR * frozen.scale)
    return s


# --------------------------------------------------------------------------------------------------- run
def run_spec(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root, bucket = args["repo_root"], args["bucket"]
    os.chdir(root)
    sys.path.insert(0, root)
    import tempfile

    import torch

    from training.ray import ncar_cells
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    record: dict[str, Any] = {k: v for k, v in spec.items() if k != "positions"}
    record.update(kind="ncar_tsslcd", positions=spec["positions"], model="tsslcd", code_revision=args.get("code_revision"),
                  origin_convention=args.get("origin_convention", "benchmark"))
    started = time.time()
    try:
        from models.tsslcd_authors.Context2CondNew import ContextTransformerAE
        from models.tsslcd_authors.F2Cnet import FutureAutoencoder
        from models.tsslcd_authors.NoiseNet import DiffusionModel, inference
        from training.ray.tsslcd_authors import BATCH, N_TIMESTEPS

        torch.set_num_threads(2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seed, band = int(spec["seed"]), spec["source_band"]
        wprefix = WEIGHTS[band]
        source = json.loads(client.get_object(Bucket=bucket, Key=f"{wprefix}/savgol-off/seed-{seed}.json")["Body"].read())
        latent = int(source["hparams"]["latent_dim"])
        ae_ctx = ContextTransformerAE(seq_len=CONTEXT, input_dim=800, latent_dim=latent, n_locations=4).to(device)
        ae_fut = FutureAutoencoder(FUTURE, 800, latent).to(device)
        diffusion = DiffusionModel(cond_dim=latent, latent_dim=latent, n_timestep=N_TIMESTEPS, device=device).to(device)
        with tempfile.TemporaryDirectory(prefix="ncar-tsslcd-") as scratch:
            for part, module in (("ae_ctx", ae_ctx), ("ae_fut", ae_fut), ("diffusion", diffusion)):
                local = Path(scratch) / f"{part}.pth"
                client.download_file(bucket, f"{wprefix}/weights/savgol-off/seed-{seed}/{part}.pth", str(local))
                module.load_state_dict(torch.load(local, map_location=device))
        for m in (ae_ctx, ae_fut, diffusion):
            m.eval()
        record.update(weights_prefix=f"{wprefix}/weights/savgol-off/seed-{seed}", hparams=source["hparams"])

        t0, t1 = pd.Timestamp(spec["target_start"]), pd.Timestamp(spec["target_end"])
        if args.get("origin_convention") == "tsslcd":          # the stock pipeline: origins one block later (step-0c check only)
            t0, t1 = t0 + STEP, t1 + STEP
        values, grid, valid = load_positions(root, [p["file"] for p in spec["positions"]], t0 - CONTEXT * STEP, t1 - STEP)
        starts = window_starts(grid, valid, t0, t1)
        idx = np.arange(len(grid))                                          # origins the range would give with no gap at all
        possible = int(((idx >= CONTEXT - 1) & (grid >= t0 - STEP) & (grid + FUTURE * STEP < t1)).sum())
        record.update(n_windows=int(len(starts)), windows_without_gaps=possible, windows_lost_to_gaps=possible - int(len(starts)),
                      target_start_used=t0.isoformat(), target_end_used=t1.isoformat())
        if not len(starts):
            raise ValueError("no windows")
        origin_rows = starts + CONTEXT - 1
        frozen = frozen_scaler(root, band)
        scaler = frozen if spec["condition"] == "frozen" else refit_scaler(frozen, values, int(origin_rows[0]))
        series = torch.as_tensor(scaler.transform(np.nan_to_num(values)), dtype=torch.float32, device=device)
        offsets = torch.arange(CONTEXT, device=device)
        torch.manual_seed(seed + 1000)
        preds = []
        ts = time.time()
        with torch.no_grad():
            for lo in range(0, len(starts), BATCH):
                idx = torch.as_tensor(starts[lo:lo + BATCH], device=device)
                preds.append(inference(ae_ctx, ae_fut, diffusion, series[idx[:, None] + offsets], device)[:, [h - 1 for h in HORIZONS]])
        record["inference_seconds"] = round(time.time() - ts, 1)
        pred = scaler.inverse(np.concatenate(preds))                                       # (N, horizons, 800) dB
        origins = np.asarray(grid[origin_rows].values, dtype="datetime64[s]").astype(np.int64)
        record["metrics"] = {}
        for p_i, pos in enumerate(spec["positions"]):
            cols = slice(200 * p_i, 200 * (p_i + 1))
            mats = {h: {"origin": origins, "pred": pred[:, j, cols].astype(np.float32),
                        "target": values[origin_rows + h][:, cols].astype(np.float32)} for j, h in enumerate(HORIZONS)}
            if args.get("save_arrays", True):
                table = ncar_cells.save_cell(client, bucket, cell_base(spec, pos), mats, ncar_cells.band_frequencies(spec["target_band"]))
            else:
                table = ncar_cells.per_bin_table(mats)
            per_h = {}
            for h, m in mats.items():
                e = m["pred"].astype(np.float64) - m["target"].astype(np.float64)
                e32 = e.astype(np.float32)
                per_h[str(h)] = {"n_windows": int(e.shape[0]), "mae_db": float(np.abs(e).mean()), "mse_db2": float((e ** 2).mean()),
                                 "rmse_db": float(np.sqrt((e ** 2).mean())),
                                 "mape_linear_pct": float((np.abs(np.power(np.float32(10.0), e32 / np.float32(10.0)) - 1) * 100).mean(dtype=np.float64)),
                                 "mape_dbm_pct": float((np.abs(e32) / np.maximum(np.abs(m["target"]), 1e-6) * 100).mean(dtype=np.float64)),
                                 "bias_db": float(e.mean())}
            record["metrics"][f"p{p_i}:{pos['receiver']}"] = per_h
        if device.type == "cuda":
            record["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3)
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    record["wall_seconds"] = round(time.time() - started, 1)
    if args.get("write_record", True):
        client.put_object(Bucket=bucket, Key=args.get("record_key") or record_key(spec), Body=json.dumps(record, default=str, indent=1).encode())
    return record


def _done(client, bucket: str, key: str) -> bool:
    try:
        return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
    except Exception:
        return False


def _revision(root: str) -> str:
    try:
        return Path("/home/cc/spectrum-source-revision").read_text().strip()
    except Exception:
        return os.environ.get("CODE_REVISION", "unknown")


def cmd_run(a: argparse.Namespace) -> None:
    import ray

    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    specs = build_specs(a.repo_root, a.cases, a.conditions, a.seeds, a.bands)
    if a.variants:
        specs = [s for s in specs if s["variant"] in a.variants]
    if a.limit:
        specs = specs[:a.limit]
    todo = [s for s in specs if not _done(client, bucket, record_key(s))]
    print(f"[ncar-tsslcd] {len(specs)} runs, {len(todo)} to do", flush=True)
    shared = {"repo_root": a.repo_root, "bucket": bucket, "code_revision": a.code_revision or _revision(a.repo_root)}
    ray.init(address=a.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=2, num_gpus=a.gpu_fraction, memory=int(16 * 2 ** 30), max_retries=1)(run_spec)
    queue, pending, ok, failed = list(todo), [], 0, 0
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            pending.append(remote.remote(queue.pop(0), shared))
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        ok, failed = ok + bool(r.get("ok")), failed + (not r.get("ok"))
        shown = {k.split(":")[1]: round(v["12"]["mae_db"], 3) for k, v in (r.get("metrics") or {}).items()}
        print(f"[ncar-tsslcd] {r['case']} {r['condition']} {r['variant']} {r['source_band']}->{r['target_band']} seed{r['seed']} ok={r.get('ok')} "
              f"n={r.get('n_windows')} mae_t12={shown} gpu={r.get('gpu_peak_allocated_gb')}GB {r.get('error', '')} ({r['wall_seconds']}s) "
              f"[{len(queue) + len(pending)} left]", flush=True)
    print(f"[ncar-tsslcd] done: {ok} ok, {failed} failed", flush=True)


def cmd_consistency(a: argparse.Namespace) -> None:
    """Step 0c: (i) the stock ``tsslcd_authors.job`` eval-only path and (ii) this module (stock origins) on the stored cell (tuned 600-800,
    seed 40, October test range), compared with the stored record (MAE, MSE per horizon; RMSE = sqrt(MSE); MAPE is not stored)."""
    import ray

    from training.ray import tsslcd_authors as ta
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    stored_key = f"{WEIGHTS['600_800']}/savgol-off/seed-40.json"
    stored = json.loads(client.get_object(Bucket=bucket, Key=stored_key)["Body"].read())
    stock_prefix = f"{PREFIX}/consistency/{a.tag}/stock"
    ray.init(address=a.address, ignore_reinit_error=True)
    stock_spec = {"savgol": "off", "seed": 40, "repo_root": a.repo_root, "bucket": bucket, "output_prefix": stock_prefix, "device": None,
                  "epochs": dict(ta.EPOCHS), "test_stride": 1, "eval_mask_ratio": 0.0, "resolution": "15min", "band": "600_800",
                  "hparams": stored["hparams"], "eval_smooth": [(1, 1, "centered")], "eval_only": True, "weights_prefix": WEIGHTS["600_800"],
                  "save_weights": False}
    spec = {"case": "check", "condition": "frozen", "seed": 40, "source_band": "600_800", "target_band": "600_800", "variant": "unswapped",
            "positions": [oct_position(r, "600_800") for r in CORE], "target_start": OCT[0].isoformat(), "target_end": OCT[1].isoformat()}
    shared = {"repo_root": a.repo_root, "bucket": bucket, "code_revision": _revision(a.repo_root), "origin_convention": "tsslcd",
              "save_arrays": False, "record_key": f"{PREFIX}/consistency/{a.tag}/ncar_tsslcd.json"}
    stock = ray.get(ray.remote(num_cpus=2, num_gpus=a.gpu_fraction, max_retries=0)(ta.job).remote(stock_spec))
    mine = ray.get(ray.remote(num_cpus=2, num_gpus=a.gpu_fraction, max_retries=0)(run_spec).remote(spec, shared))
    rows = []
    old = stored["metrics"]
    for label, new in (("tsslcd_authors --eval-only", stock), ("ncar_tsslcd (stock origins)", mine)):
        if not new.get("ok"):
            rows.append({"model": "tsslcd", "cell": label, "metric": "error", "stored": None, "ncar": None, "diff": None, "error": new.get("error")})
            continue
        for h in HORIZONS:
            if label.startswith("tsslcd_authors"):
                vals = {"mae": new["metrics"][f"raw_target__off__mae_db_t{h}"], "mse": new["metrics"][f"raw_target__off__mse_db2_t{h}"]}
                vals["rmse"] = float(np.sqrt(vals["mse"]))
                vals["mape_linear"] = None
            else:
                pooled = [new["metrics"][f"p{i}:{r}"][str(h)] for i, r in enumerate(CORE)]       # equal windows per position: pooled = mean
                vals = {"mae": float(np.mean([p["mae_db"] for p in pooled])), "mse": float(np.mean([p["mse_db2"] for p in pooled])),
                        "mape_linear": float(np.mean([p["mape_linear_pct"] for p in pooled]))}
                vals["rmse"] = float(np.sqrt(vals["mse"]))
            ref = {"mae": old[f"raw_target__off__mae_db_t{h}"], "mse": old[f"raw_target__off__mse_db2_t{h}"]}
            ref["rmse"], ref["mape_linear"] = float(np.sqrt(ref["mse"])), None
            for m in ("mae", "mse", "rmse", "mape_linear"):
                rows.append({"model": "tsslcd", "cell": f"{label}; tuned 600_800 seed 40, October test week, t+{h}, n={new.get('test_windows', new.get('n_windows'))}",
                             "metric": m, "stored": ref[m], "ncar": vals[m],
                             "diff": None if ref[m] is None or vals[m] is None else vals[m] - ref[m]})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(table.to_string(), flush=True)
    print(json.dumps({"stock_gpu_s": stock.get("inference_seconds"), "mine_gpu_peak_gb": mine.get("gpu_peak_allocated_gb"),
                      "mine_wall": mine.get("wall_seconds")}), flush=True)
    if a.out_key:
        try:
            client.head_object(Bucket=bucket, Key=a.out_key)
            raise SystemExit(f"refusing to overwrite {a.out_key}")
        except client.exceptions.ClientError:
            pass
        client.put_object(Bucket=bucket, Key=a.out_key, Body=table.to_csv(index=False).encode())
        print(f"wrote {a.out_key}", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cases", nargs="+", default=["6a", "6b", "6c"], choices=["6a", "6b", "6c"])
    r.add_argument("--conditions", nargs="+", default=["frozen", "refit"], choices=["frozen", "refit"])
    r.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    r.add_argument("--bands", nargs="+", default=list(WEIGHTS), choices=list(WEIGHTS), help="source (checkpoint) bands")
    r.add_argument("--variants", nargs="+", help="only these variants (smoke tests)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--gpu-fraction", type=float, default=0.1)
    r.add_argument("--max-parallel", type=int, default=8)
    c = sub.add_parser("consistency")
    c.add_argument("--tag", default="run1")
    c.add_argument("--out-key", default=None)
    c.add_argument("--gpu-fraction", type=float, default=0.1)
    for s in (r, c):
        s.add_argument("--code-revision", default=None)
        s.add_argument("--address", default="auto")
        s.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    os.chdir(a.repo_root)
    {"run": cmd_run, "consistency": cmd_consistency}[a.command](a)


if __name__ == "__main__":
    main()
