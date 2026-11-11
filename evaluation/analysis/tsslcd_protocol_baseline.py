"""The lookback median scored under TSS-LCD's own evaluation protocol (1-minute POWDER, 600-800 and 2400-2600 MHz), next to TSS-LCD.

Runs INSIDE the TSS-LCD harness (``training.ray.tsslcd_authors``) so the test windows, segments, normalization, savgol filter and scoring are the
harness's own: ``build_partition`` (profile 1min, scaler fit on the training rows), ``window_starts(..., CONTEXT + FUTURE, 1)`` (stride 1, consecutive
windows) and ``score_grid`` with eval smoothing off and 6:3 centered, against both the raw target and the model target (the series smoothed by the same
filter). Nothing is copied from the harness.

Prediction: at every origin, for horizon h, the median over the last k_h input steps, k_h = the validation-selected window of
``median-study/selected_k.json`` (``k_per_horizon`` of ``1min/<band>``), for every column (receiver x bin) of the normalized series, then
``scaler.inverse``. Two input choices, deterministic (no seeds):
  (i)  "model inputs": the median of ``test_model``, i.e. the same inputs TSS-LCD sees. With savgol ``centered`` these inputs contain samples from AFTER the
       origin (the authors' filter looks ahead), which is the leak of their protocol and the like-for-like comparison with it;
  (ii) "raw inputs": the median of ``test_raw`` (no look-ahead).
TSS-LCD rows come from the stored ``tsslcd-authors-full/`` (600-800: savgol off and centered, 5 seeds, authors' defaults) and ``tsslcd-authors-full-2400/``
(2400-2600: savgol off and causal only, so no centered rows exist there), averaged over seeds.

Runs as a Ray job on the cluster (data live on the workers); writes ``evaluation/results/tsslcd_protocol_baseline.csv`` (band, savgol, inputs, eval_smooth,
target, horizon, metric, value) and uploads it to ``s3://spectrum/analysis-outputs/``.

    python evaluation/analysis/tsslcd_protocol_baseline.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.environ.get("REPO_ROOT", "/workspace/spectrum-usage")
BANDS = ("600_800", "2400_2600")
SAVGOL = ("off", "centered")
EVAL_SMOOTH = [(1, 1, "centered"), (6, 3, "centered")]
OUT = "evaluation/results/tsslcd_protocol_baseline.csv"
OUT_FULL = "evaluation/results/tsslcd_protocol_baseline_fullweek.csv"
# The stored 600-800 TSS-LCD runs with savgol off and centered (``tsslcd-authors-full/``, CAMPAIGN 22.2) were scored on 9,888 windows: the test files had been synced only
# up to 2026-11-03 18:46 UTC, 74 minutes before the end of the test week (the current data give 9,962 windows; ending the range at 18:46 reproduces 9,888 exactly).
# To compare like with like the 600-800 median is scored on that same window set in the main CSV; the full-week version is written separately.
MATCHED_END = "2026-11-03T18:46:00Z"


def trailing_median(series: np.ndarray, starts: np.ndarray, k: int, context: int, chunk: int = 400) -> np.ndarray:
    """Median over the last ``k`` rows of each window's context, per column: (N, D)."""
    out = np.empty((len(starts), series.shape[1]), dtype=np.float64)
    offsets = np.arange(context - k, context)
    for i in range(0, len(starts), chunk):
        out[i:i + chunk] = np.median(series[starts[i:i + chunk][:, None] + offsets[None, :]], axis=1)
    return out


def parse_metrics(metrics: dict, horizons) -> list[tuple]:
    """``<target>__<eval>__<metric>`` keys -> (eval_smooth, target, horizon, metric, value)."""
    rows = []
    for key, value in metrics.items():
        target, name, metric = key.split("__", 2)
        if metric.startswith("mean_horizon_"):
            rows.append((name, target, "mean", metric[len("mean_horizon_"):], float(value)))
        else:
            base, _, h = metric.rpartition("_t")
            rows.append((name, target, int(h), base, float(value)))
    return rows


def band_job(band: str, savgol: str, selected: dict, test_end: str | None = None) -> dict:
    """One (band, savgol): both median variants under the harness protocol. Runs on a worker."""
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from training.ray import tsslcd_authors as T

    if test_end:
        T.RANGES["test"] = (T.RANGES["test"][0], test_end)
    prof = T.PROFILES["1min"]
    context, future, horizons = prof["context"], prof["future"], prof["horizons"]
    _, _, _, scaler = T.build_partition(ROOT, "train", None, savgol, band=band, profile="1min")
    test_raw, test_model, bounds, _ = T.build_partition(ROOT, "test", scaler, savgol, band=band, profile="1min")
    starts = T.window_starts(bounds, context + future, 1)
    horizon_rows = starts[:, None] + context + np.array([h - 1 for h in horizons])[None, :]
    gt_model_h = scaler.inverse(test_model[horizon_rows])
    gt_raw_h = scaler.inverse(test_raw[horizon_rows])
    segment_of_window = np.searchsorted([b for _, b in bounds], starts, side="right")
    kph = {int(h): int(k) for h, k in selected[f"1min/{band}"]["k_per_horizon"].items()}
    out = {"band": band, "savgol": savgol, "test_end": test_end or T.RANGES["test"][1], "n_windows": int(len(starts)), "k_per_horizon": kph, "rows": []}
    for label, series in (("(i) model inputs", test_model), ("(ii) raw inputs", test_raw)):
        med = {k: trailing_median(series, starts, k, context) for k in sorted(set(kph.values()))}
        pred = scaler.inverse(np.stack([med[kph[h]] for h in horizons], axis=1))
        metrics = T.score_grid(pred, gt_model_h, gt_raw_h, segment_of_window, EVAL_SMOOTH, consecutive=True, horizons=horizons)
        for name, target, h, metric, value in parse_metrics(metrics, horizons):
            out["rows"].append({"band": band, "savgol": savgol, "inputs": label, "eval_smooth": name, "target": target, "horizon": h, "metric": metric, "value": value})
    return out


def tsslcd_rows(client) -> list[dict]:
    rows = []
    for band, prefix in (("600_800", "tsslcd-authors-full"), ("2400_2600", "tsslcd-authors-full-2400")):
        for savgol in SAVGOL:
            acc: dict[tuple, list[float]] = {}
            for seed in range(40, 45):
                try:
                    d = json.loads(client.get_object(Bucket="spectrum", Key=f"{prefix}/savgol-{savgol}/seed-{seed}.json")["Body"].read())
                except Exception:
                    continue
                if d.get("ok"):
                    for name, target, h, metric, value in parse_metrics(d["metrics"], None):
                        acc.setdefault((name, target, h, metric), []).append(value)
            for (name, target, h, metric), v in acc.items():
                rows.append({"band": band, "savgol": savgol, "inputs": f"TSS-LCD (authors' defaults, mean of {len(v)} seeds)", "eval_smooth": name,
                             "target": target, "horizon": h, "metric": metric, "value": float(np.mean(v))})
    return rows


def main() -> None:
    import ray

    sys.path.insert(0, ROOT)
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    selected = json.loads(client.get_object(Bucket="spectrum", Key="median-study/selected_k.json")["Body"].read())
    ray.init(address="auto", ignore_reinit_error=True)
    remote = ray.remote(num_cpus=4, memory=16 * 2 ** 30, max_retries=1)(band_job)
    jobs = [(b, s, MATCHED_END if b == "600_800" else None) for b in BANDS for s in SAVGOL]
    jobs_full = [("600_800", s, None) for s in SAVGOL]
    refs = [remote.remote(b, s, selected, e) for b, s, e in jobs + jobs_full]
    all_results = ray.get(refs)
    results, results_full = all_results[:len(jobs)], all_results[len(jobs):]
    tss = tsslcd_rows(client)
    frame = pd.DataFrame([r for res in results for r in res["rows"]] + tss)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    pd.DataFrame([r for res in results_full for r in res["rows"]] + [r for r in tss if r["band"] == "600_800"]).to_csv(OUT_FULL, index=False)
    for f in (OUT, OUT_FULL):
        client.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(f), Body=open(f, "rb").read())
    pd.set_option("display.width", 220)
    print("windows and selected k per horizon:")
    for res in results + results_full:
        print(f"  {res['band']} savgol={res['savgol']} test end {res['test_end']}: {res['n_windows']} windows, k_per_horizon={res['k_per_horizon']}")
    mae = frame[frame.metric == "mae_db"]
    # sanity 1: savgol off, no eval smoothing, raw target, raw inputs, per horizon
    s1 = mae[(mae.savgol == "off") & (mae.eval_smooth == "off") & (mae.target == "raw_target") & (mae.inputs == "(ii) raw inputs")]
    print("\nsanity 1: median, savgol off, no averaging, raw target (expected 0.851 / 0.900 / 1.002 and 0.663 / 0.716 / 0.824):")
    print(s1.pivot_table(index="band", columns="horizon", values="value").round(3).to_string())
    # sanity 2: TSS-LCD centered, 6:3, smoothed target (expected about 0.739) and raw (about 1.156)
    t = mae[mae.inputs.str.startswith("TSS-LCD") & (mae.horizon == "mean")]
    print("\nsanity 2: TSS-LCD stored rows (mean MAE over t+1/15/60):")
    print(t.pivot_table(index=["band", "savgol"], columns=["target", "eval_smooth"], values="value").round(3).to_string())
    print("\nMAE averaged over 1/15/60 minutes:")
    for band in BANDS:
        for savgol in SAVGOL:
            d = mae[(mae.band == band) & (mae.savgol == savgol) & (mae.horizon == "mean")]
            w = d.pivot_table(index="inputs", columns=["target", "eval_smooth"], values="value").round(3)
            print(f"\n{band}, savgol {savgol}:")
            print(w.to_string())


if __name__ == "__main__":
    main()
