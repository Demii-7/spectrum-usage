
"""Extract model forecasts + persistence for plotting, from a saved checkpoint.

Runs on CPU so it does not disturb a training job on the GPU. The checkpoint stores only the
trainable tensors, so the config must be restated exactly as the run used it -- including
prompt_stats="normalized", because these weights predate the prompt fix.
"""
import json, os, sys, numpy as np, pandas as pd, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models"))
from SpectrumLLM import SpectrumLLMForecaster

SEQ, K, SEED = 192, 96, 0
ck = os.path.join(REPO, "checkpoints", "repro", "2d", "spectrumllm", f"seed-{SEED}",
                  f"electrosense_madrid_500_800_n25_h{K}_ci_spectrumllm.pt")
payload = torch.load(ck, map_location="cpu", weights_only=False)
freqs = payload["frequencies"]
frame = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "spec25_data.csv"))
dates = frame["date"].to_numpy()
raw = frame.drop(columns=["date"]).to_numpy(np.float32)
n, N = raw.shape

# tslib split, exactly as run_paper_regime computes it
n_train = int(n * 0.7); num_test = int(n * 0.2)
test_starts = list(range(n - num_test - SEQ, n - SEQ - K + 1))

cfg = {"model": {"input_size": N, "input_sequence_length": SEQ, "prediction_horizon": K,
                 "patch_length": 16, "patch_stride": 8, "d_model": 32, "d_ff": 32,
                 "num_prototypes": 1000, "attention_heads": 8, "dropout": 0.1,
                 "prompt_mode": "static", "prompt_stats": "normalized",
                 "backbone": "openai-community/gpt2", "llm_layers": 6, "revin_affine": False,
                 "channel_independent": True,
                 "domain_hint": "radio spectrum power spectral density"}}
torch.manual_seed(SEED)
model = SpectrumLLMForecaster(cfg)
missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected), flush=True)
model.eval()

# a handful of forecast origins spread across the test span
picks = [test_starts[0], test_starts[len(test_starts)//3], test_starts[2*len(test_starts)//3]]
X = torch.from_numpy(np.stack([raw[s:s+SEQ] for s in picks]))
with torch.no_grad():
    pred = model(X).numpy()                       # (picks, K, N)
truth = np.stack([raw[s+SEQ:s+SEQ+K] for s in picks])
hist  = np.stack([raw[s:s+SEQ] for s in picks])
persist = np.repeat(hist[:, -1:, :], K, axis=1)   # last observed value held flat

out = os.path.join(os.path.dirname(__file__), f"pred_plot_K{K}.npz")
np.savez_compressed(out,
                    hist=hist, truth=truth, pred=pred, persist=persist,
                    freqs=np.array(freqs, dtype=np.float32),
                    origins=np.array(picks),
                    hist_dates=np.stack([dates[s:s+SEQ] for s in picks]).astype("U19"),
                    fut_dates=np.stack([dates[s+SEQ:s+SEQ+K] for s in picks]).astype("U19"))
mse_m = float(((pred-truth)**2).mean()); mse_p = float(((persist-truth)**2).mean())
print(f"on these {len(picks)} windows: model MSE {mse_m:.4f}  persistence MSE {mse_p:.4f}", flush=True)
print("saved", out, flush=True)
