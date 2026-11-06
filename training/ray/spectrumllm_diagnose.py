"""Why does Time-LLM with a BERT backbone diverge at horizon 192 on ElectroSense? (CPU diagnosis on saved checkpoints)

Time-LLM's prompt is rendered per series and tokenized with ``padding=True``; the padded prompt embeddings are fed to the
backbone with no attention mask (``models/SpectrumLLM.py``). The padding length therefore depends on which series share a
batch, and every position attends to the pad tokens (for BERT, bidirectionally). The collaborator's notes already record
that the metric changes with the evaluation batch size. This script rescores saved checkpoints three ways to test whether
the divergence at K=192 is an evaluation artifact of that padding or a property of the trained weights:

    recorded : the batch size the run used (8 for 4-channel, 4 for 25-channel channel-independent runs), as in training
    window   : one window per batch (the 25 channels of a window are still padded together)
    nopad    : every series scored with its own, unpadded prompt (series grouped by prompt length, so no pad token exists)

For each it reports raw MSE and the distribution of per-window error, plus persistence and lookback-mean MSE on the same
windows. Healthy checkpoints (BERT at 4 channels on the literal split, GPT-2 at 192) are scored the same way as controls.

    python -m training.ray.spectrumllm_diagnose
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
from typing import Any

import numpy as np
import pandas as pd

SEQ = 192
CKPT = "checkpoints/repro/spectrumllm"
DATA = "datasets/electrosense-madrid"
CASES = {
    "bert_tslib_n25_h192": dict(ckpt="tslib/seed-0/electrosense_madrid_500_800_n25_h192_tslib_bbbertbaseuncased_ci_spectrumllm.pt",
                                data="spec25_data.csv", split="tslib", batch=4, stride=4, recorded=1.551, role="diverged"),
    "bert_tslib_n4_h192": dict(ckpt="tslib/seed-0/electrosense_madrid_500_800_n4_h192_tslib_bbbertbaseuncased_ci_spectrumllm.pt",
                               data="spec4_data.csv", split="tslib", batch=8, stride=1, recorded=0.794, role="diverged"),
    "bert_literal_n25_h192_seed0": dict(ckpt="literal/seed-0/electrosense_madrid_500_800_n25_h192_bbbertbaseuncased_ci_spectrumllm.pt",
                                        data="spec25_data.csv", split="literal", batch=4, stride=4, recorded=1.390, role="diverged"),
    "bert_literal_n4_h192_seed1": dict(ckpt="literal/seed-1/electrosense_madrid_500_800_n4_h192_bbbertbaseuncased_ci_spectrumllm.pt",
                                       data="spec4_data.csv", split="literal", batch=8, stride=1, recorded=0.308, role="control"),
    "gpt2_tslib_n25_h192": dict(ckpt="tslib/seed-0/electrosense_madrid_500_800_n25_h192_tslib_ci_spectrumllm.pt",
                                data="spec25_data.csv", split="tslib", batch=4, stride=4, recorded=1.029, role="control"),
    "gpt2_tslib_n4_h192": dict(ckpt="tslib/seed-0/electrosense_madrid_500_800_n4_h192_tslib_ci_spectrumllm.pt",
                               data="spec4_data.csv", split="tslib", batch=8, stride=1, recorded=0.296, role="control"),
}


def test_window_starts(n: int, split: str, horizon: int, seq: int = SEQ) -> list[int]:
    """Origins of the test windows, exactly as ``training/SpectrumLLM/repro/run_paper_regime.py`` computes them."""
    n_train = int(n * 0.7)
    if split == "tslib":
        num_test = int(n * 0.2)
        return list(range(n - num_test - seq, n - seq - horizon + 1))
    return list(range(n_train - seq, n - seq - horizon + 1))


def window_error_summary(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Overall raw MSE and how concentrated the error is across windows (arrays are (windows, K, channels))."""
    per_window = ((pred - truth) ** 2).mean(axis=(1, 2))
    order = np.sort(per_window)[::-1]
    top = max(1, int(round(0.05 * len(order))))
    bias = (pred - truth).mean(axis=(0, 1))
    return {"mse": float(per_window.mean()), "median_window_mse": float(np.median(per_window)),
            "p90_window_mse": float(np.percentile(per_window, 90)), "max_window_mse": float(order[0]),
            "top5pct_share_of_error": float(order[:top].sum() / order.sum()),
            "mean_abs_channel_bias": float(np.abs(bias).mean()), "worst_channel_bias": float(bias[np.argmax(np.abs(bias))])}


def forward_nopad(model, x):
    """The model's forward pass with every series scored under its own, unpadded prompt.

    Same computation as ``SpectrumLLMForecaster.forward``, except the prompts are rendered one series at a time and the series
    are grouped by prompt length, so no pad token exists anywhere. Used only to separate padding effects from the weights.
    """
    import torch

    batch = x.shape[0]
    normalized = model.revin.normalize(x)
    prompt_source = x if model.prompt_stats == "raw" else normalized
    if model.channel_independent:
        fold = lambda a: a.permute(0, 2, 1).reshape(batch * model.input_size, model.input_sequence_length, 1)
        normalized, prompt_source = fold(normalized), fold(prompt_source)
    aligned = model.reprogramming(model.patch_embedding(model.tokenizer_module(normalized)), model._word_embeddings())
    prefixes = [model._prompt_prefix(prompt_source[i:i + 1]) for i in range(normalized.shape[0])]
    out = [None] * normalized.shape[0]
    by_length: dict[int, list[int]] = {}
    for i, p in enumerate(prefixes):
        by_length.setdefault(p.shape[1], []).append(i)
    for length, idx in by_length.items():
        sequence = torch.cat([torch.cat([prefixes[i] for i in idx]).to(aligned.dtype), aligned[idx]], dim=1)
        hidden = model.backbone(inputs_embeds=sequence).last_hidden_state[:, length:, : model.d_ff]
        projected = model.output_projection(hidden)
        for k, i in enumerate(idx):
            out[i] = projected[k]
    prediction = torch.stack(out)
    if model.channel_independent:
        prediction = prediction.reshape(batch, model.input_size, model.prediction_horizon).permute(0, 2, 1)
    else:
        prediction = prediction.reshape(batch, model.prediction_horizon, model.input_size)
    return model.revin.denormalize(prediction)


def run_case(name: str, spec: dict[str, Any], repo_root: str, bucket: str) -> dict[str, Any]:
    import torch

    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, "models"))
    from training.ray.llm_powder import use_isolated_libs
    from training.ray.storage import spectrum_s3_client

    use_isolated_libs()
    torch.set_num_threads(int(os.environ.get("DIAG_THREADS", "8")))
    from SpectrumLLM import SpectrumLLMForecaster

    client = spectrum_s3_client()
    started = time.time()
    result: dict[str, Any] = {"case": name, **{k: spec[k] for k in ("split", "batch", "stride", "recorded", "role")}}
    with tempfile.TemporaryDirectory(prefix="diag-") as scratch:
        ckpt_path = Path(scratch) / "model.pt"
        client.download_file(bucket, f"{CKPT}/{spec['ckpt']}", str(ckpt_path))
        # the checkpoint is the collaborator's own artifact in our bucket (numpy objects need weights_only=False)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        frame = pd.read_csv(io.BytesIO(client.get_object(Bucket=bucket, Key=f"{DATA}/{spec['data']}")["Body"].read()))
    raw = frame.drop(columns=["date"]).to_numpy(np.float32)
    n, channels = raw.shape
    cfg = payload.get("config") or {"model": {
        "input_size": channels, "input_sequence_length": SEQ, "prediction_horizon": 192, "patch_length": 16,
        "patch_stride": 8, "d_model": 32, "d_ff": 32, "num_prototypes": 1000, "attention_heads": 8, "dropout": 0.1,
        "prompt_mode": "static", "prompt_stats": "normalized", "backbone": "google-bert/bert-base-uncased",
        "llm_layers": 6, "revin_affine": False, "channel_independent": True,
        "domain_hint": "radio spectrum power spectral density"}}
    horizon = int(cfg["model"]["prediction_horizon"])
    result.update(backbone=cfg["model"]["backbone"], prompt_stats=cfg["model"].get("prompt_stats"),
                  training_results=payload.get("training_results"))
    torch.manual_seed(int(payload.get("training_results", {}).get("seed", 0)))
    model = SpectrumLLMForecaster(cfg)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    result["missing_keys"], result["unexpected_keys"] = len(missing), len(unexpected)
    model.eval()

    starts = test_window_starts(n, spec["split"], horizon)[:: spec["stride"]]
    x_all = torch.from_numpy(np.stack([raw[s:s + SEQ] for s in starts]))
    truth = np.stack([raw[s + SEQ:s + SEQ + horizon] for s in starts])
    result["windows"] = len(starts)

    def score(batch_windows: int, nopad: bool = False) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for lo in range(0, len(x_all), batch_windows):
                chunk = x_all[lo:lo + batch_windows]
                preds.append((forward_nopad(model, chunk) if nopad else model(chunk)).numpy())
        return np.concatenate(preds)

    variants = {"recorded": (spec["batch"], False), "window": (1, False), "nopad": (spec["batch"], True)}
    result["variants"] = {}
    for key, (batch_windows, nopad) in variants.items():
        t0 = time.time()
        pred = score(batch_windows, nopad)
        result["variants"][key] = {**window_error_summary(pred, truth), "batch_windows": batch_windows,
                                   "seconds": round(time.time() - t0, 1)}
        print(f"[diagnose] {name} {key}: MSE {result['variants'][key]['mse']:.4f}", flush=True)
    history = x_all.numpy()
    result["persistence_mse"] = float(((np.repeat(history[:, -1:, :], horizon, axis=1) - truth) ** 2).mean())
    result["lookback_mean_mse"] = float(((history.mean(axis=1, keepdims=True) - truth) ** 2).mean())
    result["wall_seconds"] = round(time.time() - started, 1)
    result["ok"] = True
    return result


def job(payload: dict[str, Any]) -> dict[str, Any]:
    os.chdir(payload["repo_root"])
    from training.ray.storage import spectrum_s3_client

    name, spec = payload["name"], payload["spec"]
    try:
        result = run_case(name, spec, payload["repo_root"], payload["bucket"])
    except Exception as error:
        import traceback

        result = {"case": name, "ok": False, "error": f"{type(error).__name__}: {error}"[:400],
                  "trace": traceback.format_exc()[-800:]}
    try:
        spectrum_s3_client().put_object(Bucket=payload["bucket"], Key=f"{payload['output_prefix']}/{name}.json",
                                        Body=json.dumps(result, default=str).encode())
    except Exception:
        pass
    return result


def main(argv=None) -> None:
    import ray

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--output-prefix", default="spectrumllm-bert-diagnosis")
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = parser.parse_args(argv)
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=args.cpus, num_gpus=0)(job)
    pending = [remote.remote({"name": c, "spec": CASES[c], "repo_root": args.repo_root, "bucket": bucket,
                              "output_prefix": args.output_prefix}) for c in args.cases]
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        v = r.get("variants", {})
        print(f"[diagnose] DONE {r['case']} ok={r.get('ok')} recorded={r.get('recorded')} "
              + " ".join(f"{k}={v[k]['mse']:.3f}" for k in v) + (f" error={r.get('error')}" if not r.get("ok") else ""),
              flush=True)


if __name__ == "__main__":
    main()
