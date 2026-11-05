"""DLinear under the exact Time-Series-Library protocol (the settled one).

This is the canonical DLinear runner for the SpectrumLLM reproduction. The
protocol was identified by running Time-Series-Library's own pipeline on the
ElectroSense data (2026-11-04; see repro/RESULTS_PAPER_REGIME.md) and verified
against 8 of the 10 published DLinear cells:

  split      Dataset_Custom's 70/10/20 chronological split, borders verbatim
  scaling    per-channel z-score, statistics from the training span only
  model      models/DLinear.py -- the exact TSLib port (shared maps, (1/L)*ones init)
  training   Adam, lr 5e-3 (the LTSF-Linear script value), lradj 'type1'
             (constant for two epochs, then halved every epoch), 10 epochs,
             patience 3 on validation MSE, batch 32, no gradient clipping
  seed       2021 (TSLib's hardcoded fix_seed)
  reporting  test MSE in raw PSD units (TSLib's --inverse convention) and in
             standardized units, both recorded per run

Run it via training/SpectrumLLM/repro/run_dlinear.sh (container) or directly.
Output is a JSON list of run records compatible with repro/build_master.py.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "models"))
from DLinear import DLinearForecaster  # noqa: E402

SEQ_LEN = 192
TSLIB_SEED = 2021
LTSF_LR = 0.005


def load_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    freqs = [float(c) for c in rows[0][1:]]
    data = np.array([[float(x) for x in r[1:]] for r in rows[1:]], dtype=np.float32)
    return freqs, data


def windows(z, starts, horizon):
    X = np.stack([z[s:s + SEQ_LEN] for s in starts])
    Y = np.stack([z[s + SEQ_LEN:s + SEQ_LEN + horizon] for s in starts])
    return torch.from_numpy(X), torch.from_numpy(Y)


def evaluate(model, X, Y, sd, device, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            preds.append(model(X[i:i + batch_size].to(device)).cpu())
    pred = torch.cat(preds)
    sd_t = torch.from_numpy(sd)
    mse_std = float(((pred - Y) ** 2).mean())
    mse_raw = float((((pred - Y) * sd_t) ** 2).mean())
    return mse_std, mse_raw


def run_one(csv_path, horizon, args, device):
    freqs, raw = load_csv(csv_path)
    n, channels = raw.shape
    # Dataset_Custom borders, verbatim
    n_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_vali = n - n_train - num_test
    train_starts = list(range(0, n_train - SEQ_LEN - horizon + 1))
    val_starts = list(range(n_train - SEQ_LEN, n_train + num_vali - SEQ_LEN - horizon + 1))
    test_starts = list(range(n - num_test - SEQ_LEN, n - SEQ_LEN - horizon + 1))

    mu = raw[:n_train].mean(0)
    sd = raw[:n_train].std(0)
    z = ((raw - mu) / sd).astype(np.float32)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = DLinearForecaster({"model": {
        "input_sequence_length": SEQ_LEN, "prediction_horizon": horizon,
        "moving_avg": 25}}).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    Xtr, Ytr = windows(z, train_starts, horizon)
    Xva, Yva = windows(z, val_starts, horizon)
    Xte, Yte = windows(z, test_starts, horizon)

    best_val, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(args.epochs):
        # TSLib lradj 'type1': lr = base * 0.5 ** (epoch - 1), i.e. constant for
        # the first two epochs, then halved every epoch
        lr = args.lr * (0.5 ** max(0, epoch - 1))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        order = torch.randperm(len(Xtr))
        running = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx].to(device)), Ytr[idx].to(device))
            loss.backward()
            opt.step()  # no gradient clipping: TSLib has none
            running += float(loss) * len(idx)
        val_std, _ = evaluate(model, Xva, Yva, sd, device, args.batch_size)
        history.append({"epoch": epoch, "lr": lr,
                        "train_mse": running / len(order), "val_mse": val_std})
        if val_std < best_val:
            best_val, bad = val_std, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    test_std, test_raw = evaluate(model, Xte, Yte, sd, device, args.batch_size)

    return {
        "dataset": os.path.basename(csv_path), "channels": channels, "horizon": horizon,
        "seed": args.seed, "arm": "dlinear", "backbone": "none",
        "channel_independent": False, "split": "tslib", "space_trained_in": "std",
        "moving_avg": 25, "lr": args.lr, "lradj": "type1",
        "epochs_max": args.epochs, "epochs_run": len(history),
        "patience": args.patience, "batch_size": args.batch_size,
        "n_train_windows": len(train_starts), "n_val_windows": len(val_starts),
        "n_test_windows": len(test_starts), "params_total": params,
        "best_val_mse_std": best_val,
        "test_mse_std": test_std, "test_mse_raw": test_raw,
        "train_wall_s": round(train_s, 1),
        "protocol": "exact Time-Series-Library (see training/SpectrumLLM/run_dlinear.py)",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data-dir", default=os.path.join(here, "repro", "data"))
    ap.add_argument("--datasets", nargs="+",
                    default=["spec4_data.csv", "spec25_data.csv"])
    ap.add_argument("--horizons", nargs="+", type=int,
                    default=[12, 24, 48, 96, 192])
    ap.add_argument("--seed", type=int, default=TSLIB_SEED)
    ap.add_argument("--lr", type=float, default=LTSF_LR)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(here, "repro",
                                                  "results_dlinear_settled.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    rows = []
    for ds in args.datasets:
        for h in args.horizons:
            t = time.time()
            row = run_one(os.path.join(args.data_dir, ds), h, args, device)
            rows.append(row)
            json.dump(rows, open(args.out, "w"), indent=1)
            print(f"{ds} K={h}: test_mse_raw={row['test_mse_raw']:.4f} "
                  f"(std {row['test_mse_std']:.4f}, {row['epochs_run']} epochs, "
                  f"{time.time()-t:.0f}s)", flush=True)
    print(f"DONE {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
