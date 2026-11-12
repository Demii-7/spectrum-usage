"""Per-origin errors of the RevIN-MLP-with-neighbours finals at the four physical receivers (15 minutes, October test week).

The ``revinmlp_nbr`` finals of ``training.ray.tune_direct`` (``neighbor_inputs``: every (receiver, bin) series also sees the same bin at the other
three core receivers) take the four receivers' spectra side by side. This job re-runs the saved finals on the receivers' test files with the weights and
the training normalization frozen, and writes, per receiver, trained horizon and seed, the per-origin MAE over the 200 bins in exactly the layout of
``training.ray.origin_errors`` (``time`` = timestamp of the last input row, ``horizon``, ``mae_db``), under
``<prefix>/15min/<band>/<receiver>/direct/revinmlp_nbr/h<h>/seed-<s>/per_origin.csv`` (default prefix ``origin-errors-oct2026``), so they pair on the
origin time with the plain ``revinmlp`` files and with ``map_construction_vs_per_receiver``.

Origins of receiver k = the origins ``origin_errors`` uses for that receiver (``transfer_2d.target_window_starts``); an origin is scored when the
other receivers have the same timestamps for the whole window and the forecast row (the count of dropped origins is in ``summary.json``).

    python -m training.ray.neighbor_origin_errors --seeds 40 41 42 --skip-done
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

import numpy as np
import pandas as pd

CORE = ("humanities", "madsen", "moran", "sagepoint")
LOOKBACK, STEP = 96, pd.Timedelta(minutes=15)


def score_one(band: str, horizon: int, seed: int, args: dict) -> dict:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    import torch

    from training.common import transfer_2d
    from training.common.model_factory import build_model
    from training.ray.origin_errors import load_source
    from training.ray.storage import spectrum_s3_client
    from training.ray.tune_direct import final_key, group

    client, bucket = spectrum_s3_client(), args["bucket"]
    base = group(args["direct_prefix"], "15min", band, "revinmlp_nbr")
    started = time.time()
    record = {"kind": "neighbor_origin_errors", "band": band, "horizon": horizon, "seed": seed}
    try:
        final = json.loads(client.get_object(Bucket=bucket, Key=final_key(base, horizon, seed))["Body"].read())
        if not final.get("ok") or "weights_key" not in final:
            raise ValueError(f"final not ok / no weights: {final.get('error')}")
        sidecar = json.loads(client.get_object(Bucket=bucket, Key=final["meta_key"])["Body"].read())
        config = sidecar["config"]
        with tempfile.TemporaryDirectory(prefix="nbr-origin-") as scratch:
            path = Path(scratch) / "m.pt"
            client.download_file(bucket, final["weights_key"], str(path))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        norm = checkpoint["normalization"]
        mean = np.asarray(norm["mean_dbm"], dtype=np.float64).reshape(-1)
        std = np.asarray(norm["std_dbm"], dtype=np.float64).reshape(-1)
        model = build_model("revinmlp", config, np.zeros((1, 4 * 200), dtype=np.float32))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        sources = {r: load_source(root, "15min", band, r) for r in CORE}
        common = None
        for r in CORE:
            stamps = pd.DatetimeIndex(sources[r][1].timestamps)
            common = stamps if common is None else common.intersection(stamps)
        common = common.sort_values()
        arrays = np.stack([pd.DataFrame(sources[r][1].data, index=pd.DatetimeIndex(sources[r][1].timestamps)).loc[common].to_numpy(np.float64)
                           for r in CORE], axis=1)                          # (T, 4, 200)
        gap = np.concatenate([[0], np.cumsum(np.diff(common.asi8) != STEP.value)])
        normalized = ((arrays.reshape(len(common), -1) - mean) / std).astype(np.float32)
        kept, dropped, outputs = {}, {}, {}
        for k, r in enumerate(CORE):
            task, source = sources[r]
            starts = transfer_2d.target_window_starts(source.timestamps, source.segments, lookback=LOOKBACK, max_horizon=horizon, stride=1,
                                                      target_start=task.target_start, target_end=task.target_end)
            last_input = pd.DatetimeIndex(source.timestamps)[starts + LOOKBACK - 1]
            pos = common.get_indexer(last_input)
            ok = (pos >= LOOKBACK - 1) & (pos + horizon < len(common))
            ok &= np.where(ok, gap[np.clip(pos - (LOOKBACK - 1), 0, None)] == gap[np.clip(pos + horizon, None, len(common) - 1)], False)
            dropped[r], kept[r] = int((~ok).sum()), int(ok.sum())
            pos = pos[ok]
            windows = np.stack([normalized[p - LOOKBACK + 1:p + 1] for p in pos])
            preds = []
            with torch.no_grad():
                for i in range(0, len(windows), 256):
                    out = model(torch.from_numpy(windows[i:i + 256]).to(device))[:, horizon - 1, :].cpu().numpy().astype(np.float64)
                    preds.append(out)
            predicted = (np.concatenate(preds) * std + mean).reshape(-1, 4, 200)[:, k, :]
            target = arrays[pos + horizon, k, :]
            frame = pd.DataFrame({"time": np.asarray(common[pos].strftime("%Y-%m-%dT%H:%M:%SZ")), "horizon": horizon,
                                  "mae_db": np.abs(predicted - target).mean(axis=1)})
            outputs[r] = frame
            buf = io.StringIO()
            frame.to_csv(buf, index=False)
            out_base = f"{args['prefix']}/15min/{band}/{r}/direct/revinmlp_nbr/h{horizon}/seed-{seed}"
            client.put_object(Bucket=bucket, Key=f"{out_base}/per_origin.csv", Body=buf.getvalue().encode())
            client.put_object(Bucket=bucket, Key=f"{out_base}/summary.json", Body=json.dumps(
                {"ok": True, "band": band, "receiver": r, "horizon": horizon, "seed": seed, "n_origins": int(len(frame)),
                 "n_dropped": dropped[r], "mae_db": float(frame.mae_db.mean())}).encode())
        record.update(ok=True, kept=kept, dropped=dropped, mae={r: float(f.mae_db.mean()) for r, f in outputs.items()})
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1000:])
    record["wall_seconds"] = round(time.time() - started, 1)
    try:
        client.put_object(Bucket=bucket, Key=f"{args['prefix']}/15min/{band}/neighbor_job-h{horizon}-seed{seed}.json", Body=json.dumps(record, default=str).encode())
    except Exception:
        pass
    return record


def main() -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
    p.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24, 48])
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--prefix", default="origin-errors-oct2026")
    p.add_argument("--direct-prefix", default="direct-tuning")
    p.add_argument("--bucket", default=None)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args()
    sys.path.insert(0, a.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()

    def done(band, h, s):
        try:
            return bool(json.loads(client.get_object(Bucket=bucket, Key=f"{a.prefix}/15min/{band}/neighbor_job-h{h}-seed{s}.json")["Body"].read()).get("ok"))
        except Exception:
            return False

    jobs = [(b, h, s) for b in a.bands for h in a.horizons for s in a.seeds if not (a.skip_done and done(b, h, s))]
    print(f"[nbr-origin-errors] {len(jobs)} jobs", flush=True)
    args = {"repo_root": a.repo_root, "bucket": bucket, "prefix": a.prefix, "direct_prefix": a.direct_prefix}
    ray.init(address=a.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=2, num_gpus=0, memory=8 * 2 ** 30, max_retries=1)(score_one)
    queue, pending = list(jobs), []
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            pending.append(remote.remote(*queue.pop(0), args))
        done_refs, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done_refs[0])
        print(f"[nbr-origin-errors] {r['band']} h{r['horizon']} seed{r['seed']} ok={r['ok']} {r.get('mae') or r.get('error')} ({r['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
