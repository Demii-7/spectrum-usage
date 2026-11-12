"""Spectral-context ablation for TSS-LCD at 15 minutes (does the model use frequency bins outside the region it is forecasting?).

The other models' ablations train both conditions with the loss restricted to the region's bins. TSS-LCD trains two autoencoders and a latent diffusion
model over all 800 columns (4 receivers x 200 bins), so there is no per-bin loss to restrict; the contrast is therefore different:

* ``full_context``: the existing full-band final of the authors' default hyperparameters (savgol off, S3 ``tsslcd-authors-15min-default`` for
  600-800 and ``-default-2400`` for 2400-2600), re-scored in eval-only mode (same data, scaler, sampler seed and scoring code) to write the
  per-column errors; every region's MAE is read off the same run.
* ``region_only``: a new TSS-LCD trained and scored with every bin outside the region replaced by noise-floor noise at all four receivers
  (a Gaussian at each receiver's noise floor, estimated on the training rows of the plan's noise-floor regions, as in
  ``training.common.data.apply_2d_spectral_mask``). The mask is applied to the raw dBm values of all partitions BEFORE the min-max scaler is
  fit (the scaler sees the masked series), to the inputs AND to the targets the model is trained to generate (option (a): the closest
  analogue of the other models' region-only condition; keeping real targets, the ``input_only`` variant, badly degraded RevIN-MLP, CAMPAIGN 39.35).
  The loss still spans all 800 columns (the model has no per-bin loss); only the region's bins are scored, against the raw measurement.

delta = region_only - full_context (dB; > 0: the other bins help). Records go to ``<prefix>/<band>/full_context/savgol-off/seed-N.json`` and
``<prefix>/<band>/region_only/<region>/savgol-off/seed-N.json`` (the layout of ``tsslcd_authors.job``); finished cells are skipped.

    python -m training.ray.tsslcd_spectral_ablation --seeds 42 [--bands 600_800 2400_2600] [--smoke]
    # more seeds later (finished cells are skipped): --seeds 40 41 43 44
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLANS = {"600_800": "evaluation/analysis/plan_regions_600_800.csv", "2400_2600": "evaluation/analysis/plan_regions_2400_2600.csv"}
FULL_BAND_PREFIX = {"600_800": "tsslcd-authors-15min-default", "2400_2600": "tsslcd-authors-15min-default-2400"}
DEFAULT_PREFIX = "oct2026-spectral-context-15min-tsslcd"
SAVGOL = "off"


def read_plan(path: str | Path) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """(non-noise regions, noise-floor ranges as calibration ranges) of a region plan."""
    frame = pd.read_csv(path)
    noise = frame["is_noise_floor"].astype(str).str.lower().isin({"true", "1"})
    regions = [{"band_id": str(r.band_id), "start_mhz": float(r.start_mhz), "end_mhz": float(r.end_mhz)} for r in frame.loc[~noise].itertuples()]
    ranges = [[float(r.start_mhz), float(r.end_mhz)] for r in frame.loc[noise].itertuples()]
    return regions, ranges


def done(client, bucket: str, key: str) -> bool:
    try:
        return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
    except Exception:
        return False


def make_specs(args, client, bucket: str) -> list[dict[str, Any]]:
    from training.ray.tsslcd_authors import EPOCHS, parse_eval_smooth

    epochs = {"ctx": 2, "fut": 2, "diff": 5} if args.smoke else {"ctx": args.epochs_ctx, "fut": args.epochs_fut, "diff": args.epochs_diff}
    stride = 50 if args.smoke else 1
    prefix = args.prefix + ("-smoke" if args.smoke else "")
    identity = [parse_eval_smooth("1:1:centered")]
    specs: list[dict[str, Any]] = []
    for band in args.bands:
        regions, calibration = read_plan(ROOT / PLANS[band])
        if args.regions_only:
            regions = [r for r in regions if r["band_id"] in set(args.regions_only)]
        if args.smoke:
            regions = regions[:1]
        base = {"savgol": SAVGOL, "repo_root": args.repo_root, "bucket": bucket, "device": args.device, "epochs": epochs, "test_stride": stride,
                "eval_mask_ratio": 0.0, "resolution": "15min", "band": band, "eval_smooth": identity}
        for seed in args.seeds:
            if "full_context" in args.conditions:
                out = f"{prefix}/{band}/full_context"
                if not done(client, bucket, f"{out}/savgol-{SAVGOL}/seed-{seed}.json"):
                    try:
                        source = json.loads(client.get_object(Bucket=bucket, Key=f"{FULL_BAND_PREFIX[band]}/savgol-{SAVGOL}/seed-{seed}.json")["Body"].read())
                    except Exception as error:
                        print(f"[tsslcd-ablation] no full-band record for {band} seed {seed}: {error}", flush=True)
                        source = None
                    if source and source.get("ok") and source.get("artifact_keys"):
                        every = [r for r in read_plan(ROOT / PLANS[band])[0]]
                        specs.append({**base, "seed": seed, "output_prefix": out, "eval_only": True, "weights_prefix": FULL_BAND_PREFIX[band],
                                      "save_weights": False, "hparams": source.get("hparams"), "source_metrics": source.get("metrics"),
                                      "per_bin_output": True, "score_regions": every, "condition": "full_context"})
                    elif source is not None:
                        print(f"[tsslcd-ablation] full-band run {band} seed {seed} is not ok or has no saved weights", flush=True)
            if "region_only" in args.conditions:
                for region in regions:
                    out = f"{prefix}/{band}/region_only/{region['band_id']}"
                    if done(client, bucket, f"{out}/savgol-{SAVGOL}/seed-{seed}.json"):
                        continue
                    specs.append({**base, "seed": seed, "output_prefix": out, "save_weights": False, "hparams": None,
                                  "mask_region": {**region, "calibration_ranges": calibration, "seed": int(seed)},
                                  "score_regions": [region], "condition": "region_only"})
    return specs


def parse_args(argv=None) -> argparse.Namespace:
    from training.ray.tsslcd_authors import EPOCHS

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=sorted(PLANS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--conditions", nargs="+", default=["full_context", "region_only"], choices=["full_context", "region_only"])
    p.add_argument("--regions-only", nargs="+", default=None, help="only these region ids (e.g. R3 R5)")
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--smoke", action="store_true", help="one region per band, 2/2/5 epochs, test stride 50, prefix + '-smoke'")
    p.add_argument("--epochs-ctx", type=int, default=EPOCHS["ctx"])
    p.add_argument("--epochs-fut", type=int, default=EPOCHS["fut"])
    p.add_argument("--epochs-diff", type=int, default=EPOCHS["diff"])
    p.add_argument("--gpu-fraction", type=float, default=0.25)
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--device", default=None)
    p.add_argument("--bucket", default=None)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import ray

    sys.path.insert(0, str(ROOT))
    args = parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client
    from training.ray.tsslcd_authors import job

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    specs = make_specs(args, client, bucket)
    print(f"[tsslcd-ablation] {len(specs)} runs to do (bands {args.bands}, seeds {args.seeds}, smoke={args.smoke})", flush=True)
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=4, num_gpus=args.gpu_fraction, max_retries=1)(job)
    pending, queue = [], list(specs)
    while queue or pending:
        while queue and len(pending) < args.max_parallel:       # at most --max-parallel cells in flight: a short queue for the other jobs
            pending.append(remote.remote(queue.pop(0)))
        done_refs, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done_refs[0])
        shown = r.get("region_mae") and {k: round(v["mae_db_t12"], 3) for k, v in list(r["region_mae"].items())[:3]}
        print(f"[tsslcd-ablation] {r.get('mask_region', {}).get('band_id', 'full')} band={r.get('band')} seed={r.get('seed')} ok={r.get('ok')} "
              f"t12 {shown or r.get('error')} peak {r.get('peak_gpu_mb')} MB ({r.get('wall_seconds')}s)", flush=True)


if __name__ == "__main__":
    main()
