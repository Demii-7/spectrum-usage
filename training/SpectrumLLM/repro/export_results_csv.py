"""Export SpectrumLLM run results to tidy CSVs for plotting.

Usage: python export_results_csv.py results_spec4_all.json [more.json ...]
Writes, beside the inputs:
  spectrumllm_results.csv  - one row per (dataset, horizon, seed), metrics + references +
                             the published comparison row + checkpoint identity
  spectrumllm_history.csv  - one row per (dataset, horizon, seed, epoch) for learning curves
  spectrumllm_summary.csv  - one row per (dataset, horizon): mean/sd over seeds, ready to plot
"""
import csv, json, os, statistics as stx, sys

# Published comparison, article Tables I-II, raw PSD^2 (metric space established by linear probe).
PUBLISHED = {
    ("spec4_data.csv", "Time-LLM (GPT2)"):  {12: 0.2621, 24: 0.2982, 48: 0.3154, 96: 0.3157, 192: 0.3460},
    ("spec4_data.csv", "Time-LLM (BERT)"):  {12: 0.3045, 24: 0.3068, 48: 0.2911, 96: 0.3081, 192: 0.3438},
    ("spec4_data.csv", "OFA (GPT2)"):       {12: 0.2686, 24: 0.2892, 48: 0.2970, 96: 0.2958, 192: 0.3427},
    ("spec4_data.csv", "DLinear"):          {12: 0.2718, 24: 0.3036, 48: 0.3305, 96: 0.4019, 192: 0.4043},
    ("spec25_data.csv", "Time-LLM (GPT2)"): {12: 0.6166, 24: 0.7592, 48: 0.8564, 96: 0.9686, 192: 1.0429},
    ("spec25_data.csv", "Time-LLM (BERT)"): {12: 0.6321, 24: 0.7564, 48: 0.8666, 96: 0.9720, 192: 1.0311},
    ("spec25_data.csv", "OFA (GPT2)"):      {12: 0.5846, 24: 0.7593, 48: 0.8698, 96: 0.9540, 192: 1.0634},
    ("spec25_data.csv", "DLinear"):         {12: 0.6080, 24: 0.8025, 48: 0.9809, 96: 1.2581, 192: 1.6487},
}
REF_KEYS = ["persistence", "lookback_mean", "seasonal_naive_24h"]


def load(paths):
    rows = []
    for p in paths:
        rows.extend(json.load(open(p)))
    return rows


def main(paths):
    rows = load(paths)
    out_dir = os.path.dirname(os.path.abspath(paths[0]))

    rec = []
    for r in rows:
        ds, K = r["dataset"], r["horizon"]
        d = {
            "dataset": ds, "channels": r["channels"], "horizon": K, "seed": r["seed"],
            "space": r.get("space_trained_in", "std"),
            "mse_raw": r["test_mse_raw"], "mse_std": r["test_mse_std"],
            "best_val_mse": r["best_val_mse_std"],
            "epochs_run": r["epochs_run"], "batch_size": r["batch_size"], "lr": r["lr"],
            "params_total": r["params_total"], "params_trainable": r["params_trainable"],
            "n_train_windows": r["n_train_windows"], "n_test_windows": r["n_test_windows"],
            "train_wall_s": r["train_wall_s"], "infer_s_per_sample": r["infer_s_per_sample"],
            "peak_gpu_mb": r.get("peak_gpu_mb"),
            "checkpoint": r.get("checkpoint"), "checkpoint_sha256_16": r.get("checkpoint_sha256_16"),
            "checkpoint_mb": r.get("checkpoint_mb"),
        }
        for k in REF_KEYS:
            d[f"ref_{k}_raw"] = r["references_raw"][k]
            d[f"ref_{k}_std"] = r["references_std"][k]
        for (pds, model), vals in PUBLISHED.items():
            if pds == ds:
                d[f"published_{model.replace(' ', '_').replace('(', '').replace(')', '')}"] = vals.get(K)
        rec.append(d)
    rec.sort(key=lambda d: (d["dataset"], d["horizon"], d["seed"]))
    write(os.path.join(out_dir, "spectrumllm_results.csv"), rec)

    hist = []
    for r in rows:
        for h in r.get("history", []):
            hist.append({"dataset": r["dataset"], "horizon": r["horizon"], "seed": r["seed"],
                         "epoch": h["epoch"], "train_mse": h["train_mse"], "val_mse": h["val_mse"]})
    hist.sort(key=lambda d: (d["dataset"], d["horizon"], d["seed"], d["epoch"]))
    write(os.path.join(out_dir, "spectrumllm_history.csv"), hist)

    summ = {}
    for r in rows:
        summ.setdefault((r["dataset"], r["horizon"]), []).append(r)
    srec = []
    for (ds, K), group in sorted(summ.items()):
        m = [g["test_mse_raw"] for g in group]
        row = {"dataset": ds, "channels": group[0]["channels"], "horizon": K, "n_seeds": len(m),
               "mse_raw_mean": stx.mean(m),
               "mse_raw_sd": stx.stdev(m) if len(m) > 1 else 0.0,
               "mse_raw_min": min(m), "mse_raw_max": max(m),
               "train_wall_s_mean": stx.mean([g["train_wall_s"] for g in group]),
               "infer_s_per_sample_mean": stx.mean([g["infer_s_per_sample"] for g in group]),
               "params_trainable": group[0]["params_trainable"]}
        for k in REF_KEYS:
            row[f"ref_{k}_raw"] = group[0]["references_raw"][k]
        for (pds, model), vals in PUBLISHED.items():
            if pds == ds:
                row[f"published_{model.replace(' ', '_').replace('(', '').replace(')', '')}"] = vals.get(K)
        srec.append(row)
    write(os.path.join(out_dir, "spectrumllm_summary.csv"), srec)

    for f, n in [("spectrumllm_results.csv", len(rec)), ("spectrumllm_history.csv", len(hist)),
                 ("spectrumllm_summary.csv", len(srec))]:
        print(f"{f}: {n} rows")


def write(path, records):
    if not records:
        return
    keys = list(records[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main(sys.argv[1:] or ["results_spec4_all.json"])
