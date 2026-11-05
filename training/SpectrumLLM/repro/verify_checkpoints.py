"""Re-evaluate every surviving checkpoint against its recorded test metrics.

The first campaign was verified this way (reeval_runs.json, 150/150 bit-exact);
the split-study and res4 campaigns never were. This script closes that gap: for
every run whose checkpoint is still on disk with a matching SHA-256, it rebuilds
the model from the config stored in the checkpoint, loads the trainable weights,
recomputes the test windows for the recorded split, and scores them.

Reported for each run: recorded vs recomputed test MSE in both spaces, at the
evaluation batch sizes the runner could have used (the tokenizer pads prompts
with `padding=True` and no attention mask, so the metric depends on the batch
size -- see README, "reproducibility hazards"). A `match_eval_bs` column names
the batch size that reproduces the recorded value.

Run in the container, from the repository root:
  docker run --rm --gpus all -v "$PWD:$PWD" -v "$HOME/hf_cache:/root/.cache/huggingface" \
    -w "$PWD" -e PYTHONPATH="$PWD/models" --entrypoint python spectrum-split:latest \
    training/SpectrumLLM/repro/verify_checkpoints.py

Writes checkpoint_verification.json next to this script.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

RE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(RE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models"))
from SpectrumLLM import SpectrumLLMForecaster  # noqa: E402
from OFA import OFAForecaster  # noqa: E402

SEQ_LEN = 192
DAY_STEPS = 96
TRAIN_FRAC = 0.7
SOURCES = ["results_split_study.json", "res4_all_match.json", "res4_all_dm32.json",
           "res4_ci_dm32.json", "results_n25_all_tslib.json",
           # 2026-11-05 campaigns: verbatim Time-LLM prompt + prompt_stats=raw
           "results_n25_all_d64_tslib.json", "results_n4_all_d32_tslib.json",
           "results_n4_ci_d32_tslib.json", "results_n25_all_d32_tslib.json",
           "results_n25_ci_d32_tslib.json",
           # OFA and BERT arms under the same protocol
           "results_n4_ofa_ci_tslib.json", "results_n25_ofa_ci_tslib.json",
           "results_n4_bert_ci_tslib.json", "results_n25_bert_ci_tslib.json"]
# evaluation batch sizes the runner could have used: batch_size (no micro split)
# or the micro-batch sizes the README recommends per channel count. The runner's
# evaluate() uses eval_bs = micro_batch_size or batch_size, so the recorded value
# was produced at exactly one of these.
EVAL_BS_CANDIDATES = [32, 16, 8, 4, 2, 1]


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def test_starts(n, horizon, split):
    n_train = int(n * TRAIN_FRAC)
    if split == "tslib":
        num_test = int(n * 0.2)
        return list(range(n - num_test - SEQ_LEN, n - SEQ_LEN - horizon + 1))
    return list(range(n_train - SEQ_LEN, n - SEQ_LEN - horizon + 1))


def evaluate(model, z, starts, horizon, eval_bs, sd, device):
    X = torch.from_numpy(np.stack([z[s:s + SEQ_LEN] for s in starts]))
    Y = torch.from_numpy(np.stack([z[s + SEQ_LEN:s + SEQ_LEN + horizon] for s in starts]))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), eval_bs):
            preds.append(model(X[i:i + eval_bs].to(device)).cpu())
    pred = torch.cat(preds)
    sd_t = torch.from_numpy(sd)
    mse_raw = float(((pred - Y) ** 2).mean())
    mse_std = float((((pred - Y) / sd_t) ** 2).mean())
    return mse_raw, mse_std


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    rows = []
    for src in SOURCES:
        for rec in json.load(open(os.path.join(RE, src))):
            ck = rec.get("checkpoint")
            if not ck:
                continue
            abs_ck = ck if os.path.isabs(ck) else os.path.join(REPO, ck)
            if not os.path.isfile(abs_ck):
                continue
            if only and only not in os.path.basename(abs_ck):
                continue
            if sha16(abs_ck) != rec.get("checkpoint_sha256_16"):
                print(f"SKIP {os.path.basename(abs_ck)} (sha mismatch -- overwritten)",
                      flush=True)
                continue
            payload = torch.load(abs_ck, map_location="cpu", weights_only=False)
            cfg = payload["config"]["model"]
            frame = pd.read_csv(os.path.join(RE, "data", rec["dataset"]))
            raw = frame.drop(columns=["date"]).to_numpy(np.float32)
            n, _ = raw.shape
            mu = raw[:int(n * TRAIN_FRAC)].mean(0)
            sd = raw[:int(n * TRAIN_FRAC)].std(0)
            assert np.allclose(mu, payload["normalization"]["mean_dbm"], atol=1e-3), \
                "checkpoint normalisation stats disagree with the training split"
            z = raw if rec.get("space_trained_in", "raw") == "raw" else (raw - mu) / sd

            model_cls = OFAForecaster if payload.get("model_name") == "ofa" \
                else SpectrumLLMForecaster
            model = model_cls({"model": cfg}).to(device)
            missing, unexpected = model.load_state_dict(payload["model_state_dict"],
                                                        strict=False)
            # missing = the frozen backbone (never saved); unexpected must be empty
            assert not unexpected, f"unexpected keys: {unexpected[:5]}"

            starts = test_starts(n, rec["horizon"], rec["split"])
            assert len(starts) == rec["n_test_windows"], \
                f"test window count {len(starts)} != recorded {rec['n_test_windows']}"

            row = {
                "source": src, "dataset": rec["dataset"],
                "channels": rec["channels"], "horizon": rec["horizon"],
                "seed": rec["seed"], "split": rec["split"],
                "arm": payload.get("model_name", "spectrumllm"),
                "prompt_stats": rec.get("prompt_stats", ""),
                "channel_mode": "chan_indep" if cfg.get("channel_independent") else "all_channel",
                "d_model": cfg["d_model"], "d_ff": cfg["d_ff"],
                "checkpoint": os.path.basename(abs_ck),
                "n_frozen_keys_loaded": len(missing),
                "recorded_test_mse_raw": rec["test_mse_raw"],
                "recorded_test_mse_std": rec["test_mse_std"],
            }
            match = None
            # try the batch size the runner would actually have used first
            # (eval_bs = micro_batch_size or batch_size), then the alternatives
            inferred = 32 if row["channel_mode"] == "all_channel" else \
                (8 if rec["channels"] == 4 else 4)
            ordered = [inferred] + [b for b in EVAL_BS_CANDIDATES if b != inferred]
            for bs in ordered:
                mse_raw, mse_std = evaluate(model, z, starts, rec["horizon"], bs, sd, device)
                rel_raw = abs(mse_raw - rec["test_mse_raw"]) / rec["test_mse_raw"]
                rel_std = abs(mse_std - rec["test_mse_std"]) / rec["test_mse_std"]
                row[f"mse_raw@bs{bs}"] = round(mse_raw, 6)
                row[f"rel_err@bs{bs}"] = round(rel_raw, 8)
                if rel_raw < 1e-4 and rel_std < 1e-4 and match is None:
                    match = bs
            row["match_eval_bs"] = match if match is not None else \
                ("close" if min(row[f"rel_err@bs{b}"] for b in EVAL_BS_CANDIDATES) < 1e-2
                 else "none")
            rows.append(row)
            print(f"  {os.path.basename(abs_ck)}  recorded {rec['test_mse_raw']:.4f} "
                  f"-> recomputed @bs32 {row['mse_raw@bs32']}  match_eval_bs={row['match_eval_bs']}",
                  flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out = os.path.join(RE, "checkpoint_verification.json")
    # a filtered re-run replaces only the rows it recomputed; keep the rest
    if only and os.path.isfile(out):
        old = json.load(open(out))
        rows = [r for r in old if only not in r["checkpoint"]] + rows
    json.dump(rows, open(out, "w"), indent=1)
    n_ok = sum(1 for r in rows if isinstance(r["match_eval_bs"], int))
    n_close = sum(1 for r in rows if r["match_eval_bs"] == "close")
    print(f"\n{len(rows)} checkpoints re-evaluated: {n_ok} reproduce the recorded "
          f"metrics exactly, {n_close} within 1%, "
          f"{len(rows) - n_ok - n_close} MISMATCHED -> {out}", flush=True)


if __name__ == "__main__":
    main()
