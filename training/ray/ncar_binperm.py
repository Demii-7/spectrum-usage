"""Item 11 of the NCAR transfer batch: bin-order control on the core October test week (frozen, regime A).

The checkpoint is scored with its 200 input bins presented in a different order: each bin's series AND its per-bin source normalization
(mean, std) move together to the new position, so a per-bin model with shared weights sees exactly the same numbers (its result must not
change; this is the control), while a model with position-specific weights across bins (spectral context) sees a scrambled spectrum.
Predictions are mapped back to the original bin before scoring, so per-bin metrics refer to the true frequency.

Orders: ``identity`` (pipeline check: must equal the ``cells/A/.../core4_oct`` cell), ``reversed``, ``random`` (one fixed permutation,
``numpy.random.default_rng(PERM_SEED).permutation(200)``, seed recorded in every summary).
Models: the spectral-context models and their per-bin partners of item 5 (LinearAR 1D/2D, Residual LinearAR 1D/2D, LSTM 1D/2D,
RevIN-MLP / +mixing, PatchTST / -joint) plus the other spectral-context models (Residual LSTM 2D, LSTMAttn, Autoformer-CSA on CPU).
Outputs ``transfer-ncar-2026-11/item11/<order>/powder/<receiver>/20261025T1355Z/<band>/<rep>/<model>[/h<h>]/seed-N/`` in the
``ncar_cells`` format.

    python -m training.ray.ncar_binperm run --skip-done
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

PREFIX = "transfer-ncar-2026-11/item11"
PERM_SEED = 20261113
N_BINS = 200
ORDERS = ("identity", "reversed", "random")
ROLLED = ("linearar1d", "linearar2d", "residuallinearar1d", "residuallinearar2d", "vanillalstm1d", "vanillalstm", "residualvanillalstm",
          "lstmattn", "autoformer_csa")
DIRECT = ("revinmlp", "revinmlp_mix", "patchtst", "patchtst_joint")
CPU_ONLY = ("autoformer_csa",)                       # device-sensitive (step 0c): scored on CPU like its cells


def permutation(order: str) -> np.ndarray:
    if order == "identity":
        return np.arange(N_BINS)
    if order == "reversed":
        return np.arange(N_BINS)[::-1].copy()
    if order == "random":
        return np.random.default_rng(PERM_SEED).permutation(N_BINS)
    raise ValueError(order)


def permuted_normalization(base: dict[str, Any], perm: np.ndarray) -> dict[str, Any]:
    """Position p gets the statistics of original bin perm[p] (the bin whose series now sits at p)."""
    out = dict(base)
    for key in ("mean_dbm", "std_dbm"):
        values = np.asarray(base[key])
        out[key] = values.reshape(-1)[perm].reshape(values.shape).astype(values.dtype)
    return out


def unpermute(mats: dict[int, dict[str, np.ndarray]], perm: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Column p of the scored matrices belongs to original bin perm[p]; put it back in column perm[p]."""
    out = {}
    for h, m in mats.items():
        pred, tgt = np.empty_like(m["pred"]), np.empty_like(m["target"])
        pred[:, perm], tgt[:, perm] = m["pred"], m["target"]
        out[h] = {"origin": m["origin"], "pred": pred, "target": tgt}
    return out


def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """One (model, band, seed) on one core receiver, every order, frozen (A)."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    t, entries = spec["target"], spec["entries"]
    label, seed = entries[0]["model"], int(entries[0]["seed"])
    record: dict[str, Any] = {"model": label, "seed": seed, "receiver": t["site"], "band": t["band"], "mae": {}}
    started = time.time()
    try:
        import torch

        torch.set_num_threads(int(args.get("threads", 2)))
        from training.common import data_sources, transfer_1d, transfer_2d
        from training.ray import ncar_cells
        from training.ray import transfer15 as t15
        from training.ray import transfer_all as ta

        task, file = ta.prepare_target(root, t)
        original_loader = data_sources.load_csv_sources
        for entry in entries:
            rep, direct = entry["rep"], entry.get("source") == "direct"
            name = entry.get("pipeline", label)
            horizons = [int(entry["horizon"])] if direct else list(ta.ROLLED_HORIZONS)
            meta = json.loads(client.get_object(Bucket=bucket, Key=entry["meta_key"])["Body"].read())
            config = meta["config"]
            config["windowing"] = {"lookback": ta.LOOKBACK, "horizons": horizons}
            config[name].setdefault("train", {})["test_stride"] = 1
            module = transfer_2d if rep == "2d" else transfer_1d
            with tempfile.TemporaryDirectory(prefix="binperm-") as scratch:
                checkpoint = Path(scratch) / Path(entry["pt_key"]).name
                client.download_file(bucket, entry["pt_key"], str(checkpoint))
                base = torch.load(checkpoint, map_location="cpu", weights_only=False)["normalization"]
                for order in spec["orders"]:
                    perm = permutation(order)

                    def loader(*a, **k):
                        src = original_loader(*a, **k)
                        if src.data.shape[1] != N_BINS:
                            raise RuntimeError(f"expected {N_BINS} bins, got {src.data.shape}")
                        return dataclasses.replace(src, data=np.ascontiguousarray(src.data[:, perm]))   # frequencies keep their labels

                    module.load_csv_sources = loader
                    try:
                        source = module.load_target(task, [file], impute=False, max_missing_gap=0)
                    finally:
                        module.load_csv_sources = original_loader
                    result = t15.evaluate_frozen(module, rep, task, source, name, checkpoint, config, seed, False,
                                                 permuted_normalization(base, perm), origin_horizon=max(ta.ROLLED_HORIZONS))
                    mats = unpermute(ncar_cells.matrices(result, rep), perm)
                    base_key = ta.output_base(f"{args['prefix']}/{order}", "A", t, rep, label, seed,
                                              int(entry["horizon"]) if direct else None).replace("/A/", "/", 1)
                    table = ncar_cells.save_cell(client, bucket, base_key, mats, ncar_cells.band_frequencies(t["band"]))
                    metrics = {str(h): {m: float(g[m].mean()) for m in ("mae_db", "mse_db2", "rmse_db", "mape_linear_pct", "mape_dbm_pct")}
                               for h, g in table.groupby("horizon")}
                    summary = {"kind": "bin_order_control", "order": order, "perm_seed": PERM_SEED if order == "random" else None,
                               "permutation_first10": perm[:10].tolist(), "model": label, "pipeline": name, "rep": rep, "seed": seed,
                               "trained_horizon": int(entry["horizon"]) if direct else None, "target": ta.target_key(t),
                               "checkpoint_key": entry["pt_key"], "n_windows": int(len(next(iter(mats.values()))["origin"])),
                               "regime": "A", "normalization": "source statistics moved with their bins", "metrics": metrics,
                               "code_revision": args["code_revision"], **ncar_cells.tags(label, rep)}
                    client.put_object(Bucket=bucket, Key=f"{base_key}/summary.json", Body=json.dumps(summary, indent=1).encode())
                    record["mae"].setdefault(order, {}).update({h: round(v["mae_db"], 5) for h, v in metrics.items()})
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+")
    r.add_argument("--orders", nargs="+", default=list(ORDERS), choices=list(ORDERS))
    r.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
    r.add_argument("--receivers", nargs="+", default=["humanities", "madsen", "moran", "sagepoint"])
    r.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the transfer seeds (TCN/Time-LLM rules of ncar_tx_cells)")
    r.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24, 48])
    r.add_argument("--gpu-fraction", type=float, default=0.02)
    r.add_argument("--max-parallel", type=int, default=8)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--skip-done", action="store_true")
    r.add_argument("--prefix", default=PREFIX)
    r.add_argument("--code-revision", default="unknown")
    r.add_argument("--address", default="auto")
    r.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    os.chdir(a.repo_root)
    import ray

    from training.ray import ncar_tx_cells as cells
    from training.ray import transfer_all as ta
    from training.ray.ncar_cells import core_targets
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    models = a.models or list(ROLLED + DIRECT)
    sets = []
    from training.ray.inference_benchmark import discover_sources

    rolled = [m for m in ROLLED if m in models]
    if rolled:
        sets += [[e] for e in discover_sources(client, bucket, ["p15tuned"], rolled) if e["band"] in a.bands
                 and int(e["seed"]) in (a.seeds or cells.SEEDS.get(e["model"], cells.DEFAULT_SEEDS))]
    direct = [m for m in DIRECT if m in models]
    if direct:
        by: dict[tuple, dict[int, dict]] = {}
        for e in discover_sources(client, bucket, ["direct"], direct):
            if e["resolution"] == "15min" and e["band"] in a.bands and int(e["horizon"]) in a.horizons \
                    and int(e["seed"]) in (a.seeds or cells.DEFAULT_SEEDS):
                by.setdefault((e["model"], e["band"], int(e["seed"])), {})[int(e["horizon"])] = e
        sets += [[v[h] for h in sorted(v)] for v in by.values()]
    targets = [t for t in core_targets(a.bands, "oct") if t["site"] in a.receivers]
    jobs = [{"target": t, "entries": g, "orders": a.orders} for g in sets for t in targets if t["band"] == g[0]["band"]]
    # spectral-context models first within each band, partners after (both needed for the comparison)

    def done(j) -> bool:
        e = j["entries"][-1]
        key = ta.output_base(f"{a.prefix}/{j['orders'][-1]}", "A", j["target"], e["rep"], e["model"], int(e["seed"]),
                             int(e["horizon"]) if e.get("source") == "direct" else None).replace("/A/", "/", 1)
        try:
            client.head_object(Bucket=bucket, Key=f"{key}/per_bin.csv")
            return True
        except Exception:
            return False

    if a.skip_done:
        jobs = [j for j in jobs if not done(j)]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"[binperm] {len(sets)} model groups, {len(jobs)} jobs, orders {a.orders}, perm seed {PERM_SEED}", flush=True)
    shared = {"repo_root": a.repo_root, "bucket": bucket, "prefix": a.prefix, "code_revision": a.code_revision, "threads": 2}
    ray.init(address=a.address, ignore_reinit_error=True)
    queue, pending, ok, failed = list(jobs), {}, 0, 0
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            j = queue.pop(0)
            gpu = 0.0 if j["entries"][0]["model"] in CPU_ONLY else a.gpu_fraction
            pending[ray.remote(num_cpus=2, num_gpus=gpu, memory=8 * 2 ** 30, max_retries=1)(job).remote(j, shared)] = j
        finished, _ = ray.wait(list(pending), num_returns=1)
        pending.pop(finished[0])
        row = ray.get(finished[0])
        ok += bool(row.get("ok"))
        failed += not row.get("ok")
        print(f"[binperm] {row['band']} {row['receiver']} {row['model']} seed{row['seed']} ok={row.get('ok')} {row.get('mae')} "
              f"{row.get('error', '')} ({row['wall_seconds']}s) [{len(queue) + len(pending)} left]", flush=True)
        if not row.get("ok"):
            print(row.get("trace", ""), flush=True)
    print(f"[binperm] done: {ok} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
