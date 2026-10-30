"""Train the SpectrumLLM port on the authors' own ElectroSense data, in the paper's regime.

Paper regime (Liu et al., IEEE Wireless Communications 2026, p.198): lookback T=192 steps
(48 h at 15-min sampling), horizons K in {12,24,48,96,192}, chronological 7:3 split, MSE loss.
Hyperparameters the paper does not state (patch length/stride, prototype count, d_model, heads,
epochs, batch, LR, optimizer, seed) are ours and are recorded in every result row.

Errors are reported in BOTH spaces, because the paper does not say which it uses:
  mse_std : train-standardized units  (Time-Series-Library convention; comparable to Tables I-II
            under that reading, and to the trivial references in the model card)
  mse_raw : original CSV units
"""
import argparse, hashlib, json, os, time
import numpy as np
import pandas as pd
import torch
from torch import nn

from SpectrumLLM import SpectrumLLMForecaster

SEQ_LEN = 192
TRAIN_FRAC = 0.7
DAY_STEPS = 96  # 24 h at 15-min sampling


def windows(arr, starts, seq_len, horizon):
    X = np.stack([arr[s:s + seq_len] for s in starts])
    Y = np.stack([arr[s + seq_len:s + seq_len + horizon] for s in starts])
    return torch.from_numpy(X), torch.from_numpy(Y)


def trivial_references(z, starts, horizon, seq_len=SEQ_LEN):
    X, Y = windows(z, starts, seq_len, horizon)
    X, Y = X.numpy(), Y.numpy()
    snaive = np.stack([z[s + seq_len - DAY_STEPS:s + seq_len - DAY_STEPS + horizon] for s in starts])
    return {
        "persistence": float(((np.repeat(X[:, -1:, :], horizon, 1) - Y) ** 2).mean()),
        "lookback_mean": float(((np.repeat(X.mean(1, keepdims=True), horizon, 1) - Y) ** 2).mean()),
        "seasonal_naive_24h": float(((snaive - Y) ** 2).mean()),
    }


def run_one(csv_path, horizon, seed, args, device):
    raw = pd.read_csv(csv_path).drop(columns=["date"]).to_numpy(np.float32)
    n, channels = raw.shape
    n_train = int(n * TRAIN_FRAC)
    mu = raw[:n_train].mean(0)
    sd = raw[:n_train].std(0)
    zscore = (raw - mu) / sd
    # The article's MSE is in raw PSD units (established by a least-squares linear probe against its
    # published DLinear row: 5.8-8.5% relative error in raw space vs 18-38% standardized). So the model
    # is fed raw PSD and the loss is taken in raw units; RevIN normalises per instance inside the model,
    # which is exactly Time-LLM's arrangement when the loader does not scale.
    z = raw if args.space == "raw" else zscore

    # chronological windows: train/val inside the training split, test per the TSLib convention
    train_last = n_train - SEQ_LEN - horizon
    all_train = list(range(0, train_last + 1))
    cut = int(len(all_train) * (1.0 - args.val_fraction))
    train_starts, val_starts = all_train[:cut], all_train[cut:]
    test_starts = list(range(n_train - SEQ_LEN, n - SEQ_LEN - horizon + 1))

    torch.manual_seed(seed)
    np.random.seed(seed)

    config = {"model": {
        "input_size": channels, "input_sequence_length": SEQ_LEN, "prediction_horizon": horizon,
        "patch_length": args.patch_length, "patch_stride": args.patch_stride,
        "d_model": args.d_model, "d_ff": args.d_ff,
        "num_prototypes": args.num_prototypes, "attention_heads": args.heads,
        "dropout": 0.1, "prompt_mode": args.prompt_mode, "backbone": args.backbone,
        "llm_layers": args.llm_layers, "revin_affine": False,
        "domain_hint": "radio spectrum power spectral density",
    }}
    model = SpectrumLLMForecaster(config).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    Xtr, Ytr = windows(z, train_starts, SEQ_LEN, horizon)
    Xva, Yva = windows(z, val_starts, SEQ_LEN, horizon)
    Xte, Yte = windows(z, test_starts, SEQ_LEN, horizon)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = nn.MSELoss()

    def evaluate(X, Y):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), args.batch_size):
                preds.append(model(X[i:i + args.batch_size].to(device)).cpu())
        pred = torch.cat(preds)
        sd_t = torch.from_numpy(sd)
        if args.space == "raw":
            mse_raw = float(((pred - Y) ** 2).mean())
            mse_std = float((((pred - Y) / sd_t) ** 2).mean())
        else:
            mse_std = float(((pred - Y) ** 2).mean())
            mse_raw = float((((pred - Y) * sd_t) ** 2).mean())
        return mse_std, mse_raw, pred

    best_val, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(Xtr))
        running = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            running += float(loss) * len(idx)
        val_std, _, _ = evaluate(Xva, Yva)
        history.append({"epoch": epoch, "train_mse": running / len(order), "val_mse": val_std})
        print(f"    epoch {epoch:3d}  train {running / len(order):.4f}  val {val_std:.4f}", flush=True)
        if val_std < best_val - 1e-6:
            best_val, bad = val_std, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"    early stop at epoch {epoch}", flush=True)
                break
    train_s = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    test_std, test_raw, _ = evaluate(Xte, Yte)

    # Persist the trained weights. Only the trainable tensors are stored: the frozen backbone is
    # reconstructible from `backbone` + `llm_layers`, and storing it would triple the file for no
    # information. The normalisation statistics travel with the checkpoint because a prediction
    # cannot be placed on the PSD scale without them.
    ckpt_path = ckpt_sha = None
    ckpt_mb = 0.0
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        payload = {
            "trainable_state_dict": {k: v for k, v in model.state_dict().items()
                                     if k in trainable_names},
            "config": config,
            "norm_mu": mu, "norm_sd": sd,
            "meta": {"dataset": os.path.basename(csv_path), "horizon": horizon, "seed": seed,
                     "space": args.space, "seq_len": SEQ_LEN, "train_frac": TRAIN_FRAC,
                     "test_mse_raw": test_raw, "test_mse_std": test_std,
                     "epochs_run": len(history), "lr": args.lr, "batch_size": args.batch_size},
        }
        ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"{os.path.basename(csv_path).replace('_data.csv','')}_K{horizon}_seed{seed}_{args.space}.pt")
        torch.save(payload, ckpt_path)
        ckpt_sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()[:16]
        ckpt_mb = round(os.path.getsize(ckpt_path) / 1e6, 1)
        print(f"    checkpoint -> {ckpt_path} ({ckpt_mb} MB, sha256 {ckpt_sha})", flush=True)

    # single-sample inference timing
    model.eval()
    one = Xte[:1].to(device)
    with torch.no_grad():
        for _ in range(3):
            model(one)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(20):
            model(one)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_s = (time.time() - t1) / 20

    row = {
        "dataset": os.path.basename(csv_path), "channels": channels, "horizon": horizon,
        "seed": seed, "backbone": args.backbone, "llm_layers": args.llm_layers,
        "patch_length": args.patch_length, "patch_stride": args.patch_stride,
        "num_prototypes": args.num_prototypes, "d_model": args.d_model, "d_ff": args.d_ff,
        "heads": args.heads,
        "prompt_mode": args.prompt_mode,
        "epochs_run": len(history), "epochs_max": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "val_fraction": args.val_fraction,
        "n_train_windows": len(train_starts), "n_val_windows": len(val_starts),
        "n_test_windows": len(test_starts),
        "params_total": total, "params_trainable": trainable,
        "best_val_mse_std": best_val,
        "test_mse_std": test_std, "test_mse_raw": test_raw,
        "train_wall_s": round(train_s, 1), "infer_s_per_sample": round(infer_s, 5),
        "peak_gpu_mb": (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                        if device.type == "cuda" else None),
        "space_trained_in": args.space,
        "checkpoint": ckpt_path, "checkpoint_sha256_16": ckpt_sha, "checkpoint_mb": ckpt_mb,
        "references_std": trivial_references(zscore, test_starts, horizon),
        "references_raw": trivial_references(raw, test_starts, horizon),
        "history": history,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--datasets", nargs="+", default=["spec4_data.csv", "spec25_data.csv"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[12, 24, 48, 96, 192])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    # schedule defaults from Time-LLM's run_main.py argparse (the article states none)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--patch-length", type=int, default=16)
    ap.add_argument("--patch-stride", type=int, default=8)
    ap.add_argument("--num-prototypes", type=int, default=1000)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--d-ff", type=int, default=32)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--llm-layers", type=int, default=6)
    ap.add_argument("--backbone", default="openai-community/gpt2")
    ap.add_argument("--prompt-mode", default="static")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="where trained weights are written; empty string disables saving")
    ap.add_argument("--space", choices=["raw", "std"], default="raw",
                    help="units the model is trained and scored in; the article reports raw PSD")
    ap.add_argument("--out", default="results_paper_regime.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)

    rows = []
    if os.path.exists(args.out):
        rows = json.load(open(args.out))
    done = {(r["dataset"], r["horizon"], r["seed"], r["backbone"], r.get("space_trained_in", "std"))
            for r in rows}
    for ds in args.datasets:
        for horizon in args.horizons:
            for seed in args.seeds:
                key = (ds, horizon, seed, args.backbone, args.space)
                if key in done:
                    print(f"skip {key} (done)", flush=True)
                    continue
                print(f"=== {ds} K={horizon} seed={seed} backbone={args.backbone}", flush=True)
                row = run_one(os.path.join(args.data_dir, ds), horizon, seed, args, device)
                print(f"    test_mse_RAW={row['test_mse_raw']:.4f} "
                      f"(std-space {row['test_mse_std']:.4f}) "
                      f"refs_std={row['references_std']} "
                      f"({row['train_wall_s']}s, {row['params_trainable']:,} trainable)", flush=True)
                rows.append(row)
                json.dump(rows, open(args.out, "w"), indent=1)
    print("DONE", len(rows), "rows ->", args.out, flush=True)


if __name__ == "__main__":
    main()
