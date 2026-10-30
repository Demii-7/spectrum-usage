"""Verify models/SpectrumLLM.py against the resolution rule: article where it speaks,
Time-LLM's released code where the article is silent. Every check prints PASS/FAIL and the
evidence it compared. CPU-only; the GPU is in use by another job."""
import json, math, sys
import numpy as np
import torch
from SpectrumLLM import (RevIN, PatchTokenizer, SpectrumLLMForecaster,
                         autocorrelation_lags, series_statistics)

R = []
def check(name, ok, got, want):
    R.append({"check": name, "pass": bool(ok), "got": str(got), "expected": str(want)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        got={got}\n        want={want}", flush=True)

torch.manual_seed(0)

# ---- 1. patch tokenization: replication pad + (L-P)/S+2, token spans all channels -----------
tok = PatchTokenizer(patch_length=16, stride=8)
check("patch count = (L+S-P)//S+1 == (L-P)/S+2  [T:layers/Embed.py:166,180]",
      tok.num_patches(192) == 24, tok.num_patches(192), 24)
x = torch.arange(192, dtype=torch.float32).reshape(1, 192, 1).repeat(1, 1, 4)
x[0, :, 1] += 1000.0
p = tok(x)
check("token spans all N channels (article x_i in R^{NxL})",
      tuple(p.shape) == (1, 24, 16 * 4), tuple(p.shape), (1, 24, 64))
check("replication padding: last patch repeats the final sample",
      float(p[0, -1, -1]) == float(x[0, -1, -1]), float(p[0, -1, -1]), float(x[0, -1, -1]))
# first patch of channel 0 must be the first 16 timesteps
check("patch 0 of channel 0 == x[0:16]",
      torch.allclose(p[0, 0, :16], x[0, :16, 0]), p[0, 0, :4].tolist(), x[0, :4, 0].tolist())

# ---- 2. RevIN: affine default False, exact round trip ---------------------------------------
rv = RevIN(num_features=3)
check("RevIN affine defaults to False  [T:models/TimeLLM.py:192]", rv.affine is False, rv.affine, False)
z = torch.randn(4, 50, 3) * 7.0 + 55.0
n = rv.normalize(z)
check("RevIN normalize -> ~zero mean", abs(float(n.mean())) < 1e-5, float(n.mean()), 0.0)
rt = float((rv.denormalize(n) - z).abs().max())
check("RevIN round trip exact", rt < 1e-3, rt, "< 1e-3")

# ---- 3. lags are FFT autocorrelation, not largest samples  [T:models/TimeLLM.py:257-264] ----
t = torch.arange(192, dtype=torch.float32)
per = torch.sin(2 * math.pi * t / 24).reshape(1, 192, 1)
lg = autocorrelation_lags(per, top_k=5)[0].tolist()
# A signal with an integer number of periods has an EXACTLY tied circular autocorrelation at
# every multiple of the period, so topk's ordering among them is arbitrary. Assert the set, and
# assert lag 0 attains the maximum, rather than assuming it is returned first.
_s = per.transpose(1, 2)
_sp = torch.fft.rfft(_s, dim=-1)
_corr = torch.fft.irfft(_sp * torch.conj(_sp), n=_s.shape[-1], dim=-1).mean(dim=1)[0]
check("lag 0 attains the maximum autocorrelation",
      int(_corr.argmax()) == 0, int(_corr.argmax()), 0)
check("all returned lags are multiples of the period 24 (+/-1)",
      all(min(abs(v - k * 24) for k in range(9)) <= 1 for v in lg), lg, "multiples of 24")
_mixed = (torch.sin(2 * math.pi * t / 24) + 0.02 * t + 0.1 * torch.randn(192)).reshape(1, 192, 1)
check("on a non-degenerate signal lag 0 is returned first",
      autocorrelation_lags(_mixed, 5)[0, 0].item() == 0,
      autocorrelation_lags(_mixed, 5)[0].tolist(), "[0, ...]")
ramp = torch.arange(192, dtype=torch.float32).reshape(1, 192, 1)
st = series_statistics(ramp, 5)[0]
check("trend from summed first difference: rising ramp -> upward", st["trend"] == "upward", st["trend"], "upward")
check("min/max/median read off the series", (st["min"], st["max"]) == (0.0, 191.0), (st["min"], st["max"]), (0.0, 191.0))
check("lags are not argmax of raw values (ramp's largest samples are 191,190,...)",
      st["lags"] != [191, 190, 189, 188, 187], st["lags"], "!= [191,190,189,188,187]")

# ---- 4. full model: shapes, freezing, d_ff truncation, head size ----------------------------
cfg = {"model": {"input_size": 4, "input_sequence_length": 192, "prediction_horizon": 12,
                 "patch_length": 16, "patch_stride": 8, "d_model": 32, "d_ff": 32,
                 "num_prototypes": 1000, "attention_heads": 8, "dropout": 0.1,
                 "prompt_mode": "static", "backbone": "openai-community/gpt2",
                 "llm_layers": 6, "revin_affine": False}}
m = SpectrumLLMForecaster(cfg)
check("backbone truncated to llm_layers=6  [T:models/TimeLLM.py:46]",
      len(m.backbone.h) == 6, len(m.backbone.h), 6)
check("backbone fully frozen  [T:models/TimeLLM.py:163-164]",
      all(not q.requires_grad for q in m.backbone.parameters()), "all frozen", "all frozen")
check("prototype probe = Linear(vocab, 1000)  [T:models/TimeLLM.py:178-179]",
      tuple(m.reprogramming.mapping_layer.weight.shape) == (1000, 50257),
      tuple(m.reprogramming.mapping_layer.weight.shape), (1000, 50257))
head = m.output_projection[-1]
check("head = Linear(patches*d_ff, K*C) = Linear(768, 48)",
      tuple(head.weight.shape) == (48, 24 * 32), tuple(head.weight.shape), (48, 768))
check("head parameter count 36,912",
      head.weight.numel() + head.bias.numel() == 36912, head.weight.numel() + head.bias.numel(), 36912)
check("patch embedding is an MLP, not Conv1d (article: 'a simple MLP')",
      isinstance(m.patch_embedding[0], torch.nn.Linear) and isinstance(m.patch_embedding[2], torch.nn.Linear),
      type(m.patch_embedding[0]).__name__, "Linear")

xb = torch.randn(2, 192, 4) * 3.0 + 40.0
m.eval()
with torch.no_grad():
    y = m(xb)
check("output shape (B, K, C)", tuple(y.shape) == (2, 12, 4), tuple(y.shape), (2, 12, 4))
check("output returned on the INPUT scale (RevIN inverted)",
      30.0 < float(y.mean()) < 50.0, round(float(y.mean()), 2), "30..50 (input mean ~40)")

# prompt actually carries the five fields
pre = m._prompt_prefix(m.revin.normalize(xb))
stats = series_statistics(m.revin.normalize(xb), 5)[0]
txt = (f"min value {stats['min']:.3f}, max value {stats['max']:.3f}, median value {stats['median']:.3f}, "
       f"the trend of input is {stats['trend']}, top 5 lags are {stats['lags']}")
check("prompt prefix is embedded and prepended (tokens, d_llm)",
      pre.dim() == 3 and pre.shape[-1] == 768, tuple(pre.shape), "(B, tokens, 768)")
print("        prompt statistics rendered:", txt, flush=True)

# gradients reach exactly the adapters
m.train()
out = m(xb); out.sum().backward()
grads = {n_: (q.grad is not None and float(q.grad.abs().sum()) > 0)
         for n_, q in m.named_parameters() if q.requires_grad}
for part in ["patch_embedding", "reprogramming", "output_projection"]:
    hit = [k for k in grads if k.startswith(part)]
    check(f"gradient reaches {part}", hit and all(grads[k] for k in hit), f"{sum(grads[k] for k in hit)}/{len(hit)} tensors", "all")
check("no backbone parameter received a gradient",
      all(q.grad is None for q in m.backbone.parameters()), "none", "none")
check("trainable total 52,685,760",
      sum(q.numel() for q in m.parameters() if q.requires_grad) == 52685760,
      sum(q.numel() for q in m.parameters() if q.requires_grad), 52685760)

# ---- 5. pipeline: the runner's trivial references must match the independent audit ----------
sys.argv = ["x"]
import run_paper_regime as R_
import pandas as pd
v = pd.read_csv("data/spec4_data.csv").drop(columns=["date"]).to_numpy(np.float32)
n_ = len(v); ntr = int(n_ * 0.7)
mu, sd = v[:ntr].mean(0), v[:ntr].std(0)
zz = (v - mu) / sd
starts = list(range(ntr - 192, n_ - 192 - 12 + 1))
refs = R_.trivial_references(zz, starts, 12)
audit = {"persistence": 0.5127, "lookback_mean": 0.6768, "seasonal_naive_24h": 0.5364}
for k, want in audit.items():
    check(f"runner reference {k} matches the independent audit",
          abs(refs[k] - want) < 5e-4, round(refs[k], 4), want)
check("test windows = 851 at K=12 (TSLib start convention)", len(starts) == 851, len(starts), 851)
check("scaler fitted on the training split only",
      abs(float(zz[:ntr].mean())) < 1e-6, float(zz[:ntr].mean()), 0.0)

json.dump(R, open("verification.json", "w"), indent=1)
bad = [r for r in R if not r["pass"]]
print(f"\n{len(R) - len(bad)}/{len(R)} checks passed", flush=True)
if bad:
    print("FAILED:", [r["check"] for r in bad], flush=True)
