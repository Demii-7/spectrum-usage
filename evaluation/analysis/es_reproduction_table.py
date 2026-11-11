"""Per-cell mean, sample standard deviation and standard error for the ElectroSense reproduction table.

Rows: the TSLib baselines (tuned by us, seeds 40-44; untuned Reformer, seeds 2021-2023) from ``electrosense-results/`` and Time-LLM / OFA
(untuned, article regime) from ``es-repro-seeds/``. All values are window-averaged raw MSE (the article's convention). The standard error
is sd / sqrt(n) with the sample sd (n - 1); it measures seed-to-seed variation only, not the uncertainty from the finite test period.
A BERT run is flagged ``collapsed`` when its MSE exceeds ``--collapse-factor`` times the median of its cell's seeds (the documented BERT
training collapse); the table reports every seed and the count of flagged ones so the choice of excluding them stays visible.

    python evaluation/analysis/es_reproduction_table.py --out evaluation/results/es_reproduction_table.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st

import pandas as pd

HORIZONS = (12, 48, 192)
CHANNELS = {4: "spec4", 25: "spec25"}
TSL_SOURCES = {"electrosense-results/tuned-tsl/tsl/": ("DLinear", "PatchTST"), "electrosense-results/tuned-tsl-2/tsl/": ("Informer", "Autoformer", "LightTS"),
               "electrosense-results/rerun-a/tsl/": ("Reformer",)}
# Seed-fixed reruns (the library's run.py now reads TSL_SEED; ``--seedfix``): tuned winners reused, library defaults, seeds 40-44, new prefixes.
SEEDFIX_SOURCES = {"electrosense-results/seedfix-tuned/tsl/": (" (tuned)", ("DLinear", "PatchTST", "Informer", "Autoformer", "LightTS")),
                   "electrosense-results/seedfix-default/tsl/": (" (defaults)", ("DLinear", "PatchTST", "Informer", "Autoformer", "LightTS", "Reformer")),
                   "electrosense-results/seedfix-tuned-reformer/tsl/": (" (tuned)", ("Reformer",))}
LLM_ARMS = {"timellm_gpt2": "Time-LLM (GPT-2)", "timellm_bert": "Time-LLM (BERT)", "ofa_gpt2": "OFA (GPT-2)"}


def summarize(values: list[float]) -> dict:
    """Mean, sample sd (n - 1), standard error and n."""
    n = len(values)
    sd = st.stdev(values) if n > 1 else float("nan")
    return {"mean": sum(values) / n, "sd": sd, "stderr": sd / math.sqrt(n) if n > 1 else float("nan"), "n": n}


def collapsed(values: list[float], factor: float) -> list[bool]:
    """Flag runs whose value exceeds ``factor`` times the median of the cell."""
    median = st.median(values)
    return [v > factor * median for v in values]


def main() -> None:
    import boto3

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default="spectrum")
    parser.add_argument("--collapse-factor", type=float, default=1.5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seedfix", action="store_true",
                        help="read the seed-fixed TSLib reruns (electrosense-results/seedfix-*) instead of the original runs, whose five seeds are identical")
    args = parser.parse_args()
    client = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])

    def records(prefix):
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".json"):
                    yield obj["Key"], json.loads(client.get_object(Bucket=args.bucket, Key=obj["Key"])["Body"].read())

    cells: dict[tuple, list[float]] = {}
    sources = {k: ("", v) for k, v in TSL_SOURCES.items()} if not args.seedfix else SEEDFIX_SOURCES
    for prefix, (suffix, models) in sources.items():
        for _, d in records(prefix):
            ch = {v: k for k, v in CHANNELS.items()}.get(d.get("dataset"))
            if d.get("ok") and d.get("model") in models and ch and d.get("horizon") in HORIZONS:
                cells.setdefault((d["model"] + suffix, ch, d["horizon"]), []).append(d["mse_raw"])
    for _, d in records("es-repro-seeds/"):
        if d.get("ok") and d.get("arm") in LLM_ARMS and d.get("test_mse_raw") is not None:
            cells.setdefault((LLM_ARMS[d["arm"]], d["channels"], d["horizon"]), []).append(d["test_mse_raw"])
    rows = []
    for (model, ch, h), values in sorted(cells.items()):
        flags = collapsed(values, args.collapse_factor)
        row = {"model": model, "channels": ch, "horizon": h, **summarize(values), "n_collapsed": sum(flags)}
        healthy = [v for v, f in zip(values, flags) if not f]
        if len(healthy) != len(values) and healthy:
            row.update({f"healthy_{k}": v for k, v in summarize(healthy).items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    show = frame.assign(cell=lambda f: f.apply(lambda r: f"{r['mean']:.3f}±{r['stderr']:.3f} (n={int(r['n'])})", axis=1))
    print(show.pivot_table(index="model", columns=["channels", "horizon"], values="cell", aggfunc="first").to_string())
    print(f"{len(frame)} cells -> {args.out}")


if __name__ == "__main__":
    main()
