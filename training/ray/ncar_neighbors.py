"""NCAR transfer batch, item 8: RevIN-MLP with neighbours (``revinmlp_nbr``) at the held-out receivers guest house and law 73, frozen.

The ``revinmlp_nbr`` finals (``direct-tuning/15min/<band>/revinmlp_nbr``, seeds 40-42, one model per horizon 4/12/24/48) were trained with the four
core receivers side by side in slots humanities, madsen, moran, sage point (the order of the training files); each slot's forecast sees its own
series first and the other three in slot order. At a held-out target there are again three neighbours, as in training:
  ``nearest``   the three core receivers nearest to the target (``data/locations/powder.json``) keep their slots; the target takes the slot of the
                dropped (farthest) receiver;
  ``farthest``  the other choice: the three farthest keep their slots; the target takes the slot of the nearest one.
Weights and the checkpoint's normalization are frozen (the target uses the statistics of the slot it occupies; the model's own per-window RevIN
removes them again). Plain RevIN-MLP (``direct-tuning/15min/<band>/revinmlp``, same seeds and horizons) is scored on the same windows.

Windows: the transfer windows of the target (``transfer_2d.target_window_starts``, lookback 96, origins with a full 48-step future, October test
week), kept where all four core receivers and the target have every timestamp of the window and the forecast row (one window set per target and
band, shared by both neighbour choices and the plain model; the dropped count is in the record). Check rows (``insample-check``): the core four in
their trained slots on the origins of ``training.ray.neighbor_origin_errors`` (max horizon = trained horizon), to compare with
``origin-errors-oct2026``.

Outputs: ``transfer-ncar-2026-11/item8/<band>/<target>/<model>[/<choice>]/h<h>/seed-N/{forecasts.npz, per_bin.csv}`` (``ncar_cells`` layout)
and one record per (band, horizon, seed) ``transfer-ncar-2026-11/item8/<band>/records/h<h>-seed-N.json``.

    python -m training.ray.ncar_neighbors run --skip-done
    python -m training.ray.ncar_neighbors table --out evaluation/results/transfer_revinmlp_neighbors.csv     # anywhere with S3 access
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREFIX = "transfer-ncar-2026-11/item8"
SLOTS = ("humanities", "madsen", "moran", "sagepoint")        # training order of the neighbour model (data.files, concat: columns)
TARGETS = ("guesthouse", "law73")
NAMES = {"humanities": "humanities", "madsen": "madsen", "moran": "moran", "sagepoint": "sage point", "guesthouse": "guest house", "law73": "law 73"}
LOOKBACK, MAX_H, STEP = 96, 48, pd.Timedelta(minutes=15)


def distances(root: str, target: str) -> dict[str, float]:
    points = {e["name"]: (e["longitude"], e["latitude"]) for e in json.loads((Path(root) / "data/locations/powder.json").read_text())["endpoints"]}
    lo0, la0 = points[NAMES[target]]
    out = {}
    for r in SLOTS:
        lo, la = points[NAMES[r]]
        x = math.radians(lo - lo0) * math.cos(math.radians((la + la0) / 2)) * 6_371_000.0
        y = math.radians(la - la0) * 6_371_000.0
        out[r] = math.hypot(x, y)
    return out


def arrangements(root: str, target: str) -> dict[str, dict[str, Any]]:
    """{choice: {slot receiver -> source receiver, target slot, neighbours}}."""
    d = distances(root, target)
    order = sorted(SLOTS, key=d.get)
    out = {}
    for choice, dropped in (("nearest", order[-1]), ("farthest", order[0])):
        out[choice] = {"slot_of_target": dropped, "neighbours": [r for r in SLOTS if r != dropped],
                       "distances_m": {r: round(d[r], 1) for r in SLOTS if r != dropped}}
    return out


def load_frame(root: str, band: str, receiver: str):
    from training.ray.origin_errors import load_source

    task, source = load_source(root, "15min", band, receiver)
    return task, source, pd.DataFrame(np.asarray(source.data, dtype=np.float64), index=pd.DatetimeIndex(source.timestamps))


def window_origins(task, source, common: pd.DatetimeIndex, gap: np.ndarray, max_h: int) -> tuple[np.ndarray, int]:
    """Positions in ``common`` of the last input row of each kept transfer window of this receiver, and the dropped count."""
    from training.common import transfer_2d

    starts = transfer_2d.target_window_starts(source.timestamps, source.segments, lookback=LOOKBACK, max_horizon=max_h, stride=1,
                                              target_start=task.target_start, target_end=task.target_end)
    last_input = pd.DatetimeIndex(source.timestamps)[starts + LOOKBACK - 1]
    pos = common.get_indexer(last_input)
    ok = (pos >= LOOKBACK - 1) & (pos + max_h < len(common))
    ok &= np.where(ok, gap[np.clip(pos - (LOOKBACK - 1), 0, None)] == gap[np.clip(pos + max_h, None, len(common) - 1)], False)
    return pos[ok], int((~ok).sum())


def predict(model, windows: np.ndarray, horizon: int, device) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(windows), 256):
            out.append(model(torch.from_numpy(windows[i:i + 256]).to(device))[:, horizon - 1, :].cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def load_final(client, bucket: str, model: str, band: str, horizon: int, seed: int):
    import torch

    from training.common.model_factory import build_model
    from training.ray.tune_direct import final_key, group

    final = json.loads(client.get_object(Bucket=bucket, Key=final_key(group("direct-tuning", "15min", band, model), horizon, seed))["Body"].read())
    if not final.get("ok") or "weights_key" not in final:
        raise ValueError(f"{model} h{horizon} seed {seed}: final not ok / no weights")
    config = json.loads(client.get_object(Bucket=bucket, Key=final["meta_key"])["Body"].read())["config"]
    with tempfile.TemporaryDirectory(prefix="ncar-nbr-") as scratch:
        path = Path(scratch) / "m.pt"
        client.download_file(bucket, final["weights_key"], str(path))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    width = 4 * 200 if model == "revinmlp_nbr" else 200
    net = build_model("revinmlp", config, np.zeros((1, width), dtype=np.float32))
    net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    norm = checkpoint["normalization"]
    return net, np.asarray(norm["mean_dbm"], dtype=np.float64).reshape(-1), np.asarray(norm["std_dbm"], dtype=np.float64).reshape(-1), \
        {"weights_key": final["weights_key"], "sha256": final.get("sha256")}


def job(band: str, horizon: int, seed: int, args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    import torch

    from training.ray import ncar_cells
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = f"{args['prefix']}/{band}"
    record: dict[str, Any] = {"kind": "ncar_neighbors", "band": band, "horizon": horizon, "seed": seed, "results": [],
                              "code_revision": args.get("code_revision")}
    started = time.time()
    freqs = ncar_cells.band_frequencies(band)
    try:
        nbr, n_mean, n_std, n_prov = load_final(client, bucket, "revinmlp_nbr", band, horizon, seed)
        plain, p_mean, p_std, p_prov = load_final(client, bucket, "revinmlp", band, horizon, seed)
        nbr.to(device).eval()
        plain.to(device).eval()
        record.update(revinmlp_nbr=n_prov, revinmlp=p_prov)
        frames = {r: load_frame(root, band, r) for r in SLOTS + TARGETS}

        def emit(target: str, model: str, choice: str | None, mats, extra: dict[str, Any]) -> None:
            cell = f"{base}/{target}/{model}" + (f"/{choice}" if choice else "") + f"/h{horizon}/seed-{seed}"
            table = ncar_cells.save_cell(client, bucket, cell, mats, freqs)
            e = mats[horizon]["pred"].astype(np.float64) - mats[horizon]["target"].astype(np.float64)
            lin = np.abs(np.power(10.0, e / 10.0) - 1.0) * 100.0
            dbm = np.abs(e) / np.maximum(np.abs(mats[horizon]["target"].astype(np.float64)), 1e-6) * 100.0
            record["results"].append({"target": target, "model": model, "choice": choice, "key": cell, "n_windows": int(e.shape[0]),
                                      "mae_db": float(np.abs(e).mean()), "mse_db2": float((e ** 2).mean()), "rmse_db": float(np.sqrt((e ** 2).mean())),
                                      "mape_linear_pct": float(lin.mean()), "mape_dbm_pct": float(dbm.mean()), "bias_db": float(e.mean()),
                                      "n_bins": int(len(table) // max(1, table.horizon.nunique())), **extra})

        # --- held-out targets: one common window set (all four core receivers and the target present)
        for target in TARGETS:
            common = frames[target][2].index
            for r in SLOTS:
                common = common.intersection(frames[r][2].index)
            common = common.sort_values()
            gap = np.concatenate([[0], np.cumsum(np.diff(common.asi8) != STEP.value)])
            task, source, _ = frames[target]
            pos, dropped = window_origins(task, source, common, gap, MAX_H)
            data = {r: frames[r][2].loc[common].to_numpy(np.float64) for r in SLOTS + (target,)}
            origin = common[pos].asi8 // 10 ** 9
            truth = data[target][pos + horizon].astype(np.float32)
            win = lambda arr: np.stack([arr[p - LOOKBACK + 1:p + 1] for p in pos])          # noqa: E731
            # plain RevIN-MLP
            pw = ((win(data[target]) - p_mean) / p_std).astype(np.float32)
            pp = predict(plain, pw, horizon, device) * p_std + p_mean
            emit(target, "revinmlp", None, {horizon: {"origin": origin, "pred": pp.astype(np.float32), "target": truth}},
                 {"n_dropped_windows": dropped, "neighbours": []})
            for choice, arr in arrangements(root, target).items():
                slot_series = [data[target] if s == arr["slot_of_target"] else data[s] for s in SLOTS]
                stacked = np.concatenate(slot_series, axis=1)                               # (T, 800) receiver-major
                nw = ((win(stacked) - n_mean) / n_std).astype(np.float32)
                k = SLOTS.index(arr["slot_of_target"])
                out = (predict(nbr, nw, horizon, device) * n_std + n_mean).reshape(-1, 4, 200)[:, k, :]
                emit(target, "revinmlp_nbr", choice, {horizon: {"origin": origin, "pred": out.astype(np.float32), "target": truth}},
                     {"n_dropped_windows": dropped, **arr})
        # --- in-sample check: the core four in their own slots, origins of neighbor_origin_errors (max horizon = trained horizon)
        common = frames[SLOTS[0]][2].index
        for r in SLOTS[1:]:
            common = common.intersection(frames[r][2].index)
        common = common.sort_values()
        gap = np.concatenate([[0], np.cumsum(np.diff(common.asi8) != STEP.value)])
        stacked = np.concatenate([frames[r][2].loc[common].to_numpy(np.float64) for r in SLOTS], axis=1)
        normalized = ((stacked - n_mean) / n_std).astype(np.float32)
        for k, r in enumerate(SLOTS):
            task, source, _ = frames[r]
            pos, dropped = window_origins(task, source, common, gap, horizon)
            out = (predict(nbr, np.stack([normalized[p - LOOKBACK + 1:p + 1] for p in pos]), horizon, device) * n_std + n_mean).reshape(-1, 4, 200)[:, k, :]
            mats = {horizon: {"origin": common[pos].asi8 // 10 ** 9, "pred": out.astype(np.float32),
                              "target": stacked[pos + horizon, k * 200:(k + 1) * 200].astype(np.float32)}}
            emit(r, "revinmlp_nbr", "insample-check", mats, {"n_dropped_windows": dropped})
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/records/h{horizon}-seed-{seed}.json" if record["ok"] else
                      f"{base}/errors/h{horizon}-seed-{seed}-{int(time.time())}.json", Body=json.dumps(record, default=str, indent=1).encode())
    return record


def cmd_run(a: argparse.Namespace) -> None:
    import ray

    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()

    def done(b, h, s) -> bool:
        try:
            client.head_object(Bucket=a.bucket, Key=f"{a.prefix}/{b}/records/h{h}-seed-{s}.json")
            return True
        except Exception:
            return False

    jobs = [(b, h, s) for b in a.bands for h in a.horizons for s in a.seeds if not (a.skip_done and done(b, h, s))]
    print(f"[ncar-nbr] {len(jobs)} jobs", flush=True)
    shared = {"repo_root": a.repo_root, "bucket": a.bucket, "prefix": a.prefix, "code_revision": a.code_revision}
    ray.init(address=a.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=2, num_gpus=a.gpu_fraction, memory=8 * 2 ** 30, max_retries=1)(job)
    queue, pending = list(jobs), []
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            pending.append(remote.remote(*queue.pop(0), shared))
        finished, pending = ray.wait(pending, num_returns=1)
        r = ray.get(finished[0])
        shown = {f"{x['target']}/{x['model']}/{x['choice']}": round(x["mae_db"], 3) for x in r.get("results", [])}
        print(f"[ncar-nbr] {r['band']} h{r['horizon']} seed{r['seed']} ok={r['ok']} {shown or r.get('error')} ({r['wall_seconds']}s)", flush=True)


def cmd_table(a: argparse.Namespace) -> None:
    from training.ray import ncar_cells
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    rows = []
    for b in a.bands:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix=f"{a.prefix}/{b}/records/"):
            for o in page.get("Contents", []):
                r = json.loads(client.get_object(Bucket=a.bucket, Key=o["Key"])["Body"].read())
                for x in r["results"]:
                    rows.append({"band": r["band"], "horizon": r["horizon"], "seed": r["seed"], "receiver": x["target"], "model": x["model"],
                                 "neighbour_choice": x["choice"] or "none", "neighbours": " ".join(x.get("neighbours", [])),
                                 "target_slot": x.get("slot_of_target", ""), "n_windows": x["n_windows"],
                                 "n_dropped_windows": x.get("n_dropped_windows"), "mae": x["mae_db"], "mse": x["mse_db2"], "rmse": x["rmse_db"],
                                 "mape_linear_pct": x["mape_linear_pct"], "mape_dbm_pct": x["mape_dbm_pct"], "per_bin_key": f"{x['key']}/per_bin.csv"})
    seeds = pd.DataFrame(rows)
    seeds["level"] = "seed"
    keys = ["band", "horizon", "receiver", "model", "neighbour_choice", "neighbours", "target_slot"]
    metrics = ["mae", "mse", "rmse", "mape_linear_pct", "mape_dbm_pct"]
    g = seeds.groupby(keys)
    recv = g[metrics].mean().join(g[metrics].sem().add_suffix("_se_seeds")).join(g.seed.nunique().rename("n_seeds")) \
        .join(g.n_windows.first()).reset_index()
    recv["level"] = "receiver"
    held = seeds[seeds.neighbour_choice != "insample-check"]
    per_seed_group = held.groupby(["band", "horizon", "model", "neighbour_choice", "seed"])[metrics].mean().reset_index()
    gg = per_seed_group.groupby(["band", "horizon", "model", "neighbour_choice"])
    grp = gg[metrics].mean().join(gg[metrics].sem().add_suffix("_se_seeds")).join(gg.seed.nunique().rename("n_seeds")).reset_index()
    grp["receiver"], grp["level"] = "guesthouse+law73", "group"
    table = pd.concat([recv, grp, seeds], ignore_index=True)
    for col, val in (("spectral_context", None), ("spatial_context", None)):
        table[col] = [ncar_cells.tags(m)[col] for m in table.model]
    table["condition"] = "frozen"
    table.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    show = table[table.level != "seed"].pivot_table(index=["band", "receiver", "model", "neighbour_choice"], columns="horizon", values="mae")
    print(show.round(4).to_string())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "table"):
        s = sub.add_parser(name)
        s.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
        s.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24, 48])
        s.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
        s.add_argument("--gpu-fraction", type=float, default=0.0)
        s.add_argument("--max-parallel", type=int, default=4)
        s.add_argument("--skip-done", action="store_true")
        s.add_argument("--out", default="evaluation/results/transfer_revinmlp_neighbors.csv")
        s.add_argument("--prefix", default=PREFIX)
        s.add_argument("--bucket", default=None)
        s.add_argument("--code-revision", default=None)
        s.add_argument("--address", default="auto")
        s.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    a.bucket = a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    {"run": cmd_run, "table": cmd_table}[a.command](a)


if __name__ == "__main__":
    main()
