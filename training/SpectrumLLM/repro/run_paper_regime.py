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
import argparse, hashlib, json, os, re, time
import numpy as np
import pandas as pd
import torch
from torch import nn

from SpectrumLLM import SpectrumLLMForecaster
from OFA import OFAForecaster
from DLinear import DLinearForecaster

SEQ_LEN = 192
TRAIN_FRAC = 0.7
SITE = "electrosense_madrid"   # data/<site>/ dir name; see data/sync_object_storage.py
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
    frame = pd.read_csv(csv_path).drop(columns=["date"])
    # Column headers are the channel centre frequencies in MHz; they travel into the
    # checkpoint as "frequencies", which the repo loaders use to align target bands.
    freqs = [float(c) for c in frame.columns]
    raw = frame.to_numpy(np.float32)
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

    # Chronological windows. The article says only "split into training and testing sets with a
    # ratio of 7:3", which fixes the training fraction at 70% but not what happens to the other
    # 30%. Two readings, both tested:
    #   literal -- all 30% is test; the validation windows for early stopping are carved out of
    #              the END of the training span, so nothing between 70% and 100% is withheld.
    #   tslib   -- the remaining 30% is 10% validation then 20% test, chronologically, which is
    #              what Time-Series-Library's Dataset_Custom does (num_train=0.7, num_test=0.2,
    #              num_vali=remainder). Each non-training span starts SEQ_LEN early so its windows
    #              carry real history, matching that loader's border1s offsets.
    # The two differ in BOTH the scored rows and the fitting set, which is easy to miss: under
    # literal the last 10% of the training windows become validation, so fitting uses ~90% of the
    # windows inside the first 70% of rows (1626 at K=12); under tslib validation comes from the
    # 70-80% band instead, so fitting uses all of them (1807 at K=12). Normalisation statistics
    # are taken from raw[:n_train] either way, matching the loader's scaler-on-train-border.
    if args.split == "tslib":
        # Verbatim from Time-Series-Library data_provider/data_loader.py, Dataset_Custom:
        #   num_train = int(len(df_raw) * 0.7); num_test = int(len(df_raw) * 0.2)
        #   num_vali  = len(df_raw) - num_train - num_test
        #   border1s  = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
        #   border2s  = [num_train, num_train + num_vali, len(df_raw)]
        # Note n - int(0.2n) is not int(0.8n) in general, so the test border is computed the
        # library's way rather than from 0.8.
        num_test = int(n * 0.2)
        num_vali = n - n_train - num_test
        train_starts = list(range(0, n_train - SEQ_LEN - horizon + 1))
        val_starts = list(range(n_train - SEQ_LEN, n_train + num_vali - SEQ_LEN - horizon + 1))
        test_starts = list(range(n - num_test - SEQ_LEN, n - SEQ_LEN - horizon + 1))
    else:
        train_last = n_train - SEQ_LEN - horizon
        all_train = list(range(0, train_last + 1))
        cut = int(len(all_train) * (1.0 - args.val_fraction))
        train_starts, val_starts = all_train[:cut], all_train[cut:]
        test_starts = list(range(n_train - SEQ_LEN, n - SEQ_LEN - horizon + 1))
    if not val_starts or not test_starts:
        raise SystemExit(f"empty val/test window set for K={horizon} under split={args.split}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    config = {"model": {
        "input_size": channels, "input_sequence_length": SEQ_LEN, "prediction_horizon": horizon,
        "patch_length": args.patch_length, "patch_stride": args.patch_stride,
        "d_model": args.d_model, "d_ff": args.d_ff,
        "num_prototypes": args.num_prototypes, "attention_heads": args.heads,
        "dropout": 0.1, "prompt_mode": args.prompt_mode, "prompt_stats": args.prompt_stats,
        "backbone": args.backbone,
        "llm_layers": args.llm_layers, "revin_affine": False,
        "channel_independent": args.channel_independent,
        "domain_hint": "radio spectrum power spectral density",
    }}
    # arm selects which benchmarked method to build.
    build = {'ofa': OFAForecaster, 'dlinear': DLinearForecaster}.get(
        args.arm, SpectrumLLMForecaster)
    model = build(config).to(device)
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
            eval_bs = args.micro_batch_size or args.batch_size
            for i in range(0, len(X), eval_bs):
                preds.append(model(X[i:i + eval_bs].to(device)).cpu())
        pred = torch.cat(preds)
        sd_t = torch.from_numpy(sd)
        if args.space == "raw":
            mse_raw = float(((pred - Y) ** 2).mean())
            mse_std = float((((pred - Y) / sd_t) ** 2).mean())
        else:
            mse_std = float(((pred - Y) ** 2).mean())
            mse_raw = float((((pred - Y) * sd_t) ** 2).mean())
        # Endpoint-only error (the single step t+H, the convention of the repo's own evaluation), next to the window-averaged one
        # above (steps 1..H, the Time-Series-Library and article convention); the two differ whenever the error grows with the step.
        last_err = (pred[:, -1] - Y[:, -1]) ** 2
        if args.space == "raw":
            evaluate.endpoint = {"raw": float(last_err.mean()), "std": float((last_err / sd_t ** 2).mean())}
        else:
            evaluate.endpoint = {"std": float(last_err.mean()), "raw": float((last_err * sd_t ** 2).mean())}
        return mse_std, mse_raw, pred

    best_val, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(args.epochs):
        # Time-Series-Library's schedule: lr = base_lr * decay**epoch, set at epoch start
        if args.lr_decay != 1.0:
            for g in opt.param_groups:
                g["lr"] = args.lr * (args.lr_decay ** epoch)
        model.train()
        order = torch.randperm(len(Xtr))
        running = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            opt.zero_grad()
            # Gradient accumulation. Channel-independent mode expands one sample into N
            # sequences, so a 32-sample batch is 800 sequences at N=25 and will not fit a
            # 24 GB card. Splitting into micro-batches keeps the OPTIMIZER step at
            # args.batch_size samples, so the schedule is identical across channel modes and
            # the comparison is not confounded by batch size.
            micro = args.micro_batch_size or len(idx)
            for j in range(0, len(idx), micro):
                sub = idx[j:j + micro]
                xb, yb = Xtr[sub].to(device), Ytr[sub].to(device)
                # weight each micro-batch by its share so the summed gradient equals the
                # gradient of the mean loss over the whole batch
                loss = loss_fn(model(xb), yb) * (len(sub) / len(idx))
                loss.backward()
                running += float(loss) * len(idx)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
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
    test_endpoint = dict(evaluate.endpoint)             # endpoint-only (step t+H) MSE of the same predictions

    # Persist the trained weights. Only the trainable tensors are stored: the frozen backbone is
    # reconstructible from `backbone` + `llm_layers`, and storing it would triple the file for no
    # information. The normalisation statistics travel with the checkpoint because a prediction
    # cannot be placed on the PSD scale without them.
    ckpt_path = ckpt_sha = None
    ckpt_mb = 0.0
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        # Repo checkpoint contract: training/common/train_integrated.py writes
        # {model_name, model_state_dict, normalization, frequencies, training_results,
        # pretraining}, and training/common/transfer_2d.py rejects any checkpoint whose
        # "normalization" key is missing. Layout follows CHAMELEON.md:159 --
        # checkpoints/<representation>/<model>/seed-<N>/<chunk_id>_<model>.pt with
        # chunk_id = <site>_<start_mhz>_<end_mhz>_h<horizon>.
        payload = {
            "model_name": "spectrumllm" if args.arm == "timellm" else args.arm,
            "model_state_dict": {k: v for k, v in model.state_dict().items()
                                 if k in trainable_names},
            "normalization": {
                "mean_dbm": mu, "std_dbm": sd, "site": SITE,
                "source_split": "train_before_validation", "frequency_axis": -1,
            },
            "frequencies": freqs,
            "training_results": {
                "dataset": os.path.basename(csv_path), "horizon": horizon, "seed": seed,
                "space": args.space, "seq_len": SEQ_LEN, "train_frac": TRAIN_FRAC,
                "test_mse_raw": test_raw, "test_mse_std": test_std,
                "epochs_run": len(history), "lr": args.lr, "batch_size": args.batch_size,
            },
            "pretraining": None,
            "config": config,
            "notes": ("model_state_dict holds TRAINABLE parameters only; the frozen backbone is "
                      "rebuilt from config['model']['backbone'] + llm_layers. mean_dbm/std_dbm are "
                      "the repo's key names; the upstream ElectroSense CSV states no unit."),
        }
        # n<channels> is REQUIRED: spec4/spec25/spec151 all span 500-800 MHz, so a chunk id
        # built from the band alone collides across scenarios and silently overwrites.
        # A non-default backbone gets its own tag so the ablation never overwrites the
        # pretrained run it is the control for.
        # The SPLIT is always tagged and non-default capacities (d_model/d_ff) get a
        # capacity tag: both were missing before 2026-11-04, which silently destroyed
        # weights twice (res4_all_dm32 overwritten by res4_all_match; the split study's
        # literal K=12 overwritten by its own tslib K=12 run) and left 55 first-campaign
        # runs pointing at paths later occupied by different models. See MATRIX.md,
        # "Chunk ids encode neither the split nor the capacity".
        bb_tag = "" if args.backbone == "openai-community/gpt2" else \
            "_bb" + re.sub(r"[^a-z0-9]+", "", args.backbone.lower().split("/")[-1])
        ci_tag = "_ci" if args.channel_independent else ""
        arm_tag = "" if args.arm == "timellm" else f"_{args.arm}"
        cap_tag = "" if (args.d_model == 32 and args.d_ff == 32) else \
            f"_dm{args.d_model}ff{args.d_ff}"
        chunk_id = (f"{SITE}_{int(min(freqs))}_{int(max(freqs))}"
                    f"_n{len(freqs)}_h{horizon}_{args.split}{bb_tag}{ci_tag}{arm_tag}{cap_tag}")
        ckpt_path = os.path.join(args.checkpoint_dir, "2d", "spectrumllm", f"seed-{seed}",
                                 f"{chunk_id}_spectrumllm.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
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
        "lr": args.lr, "val_fraction": args.val_fraction, "split": args.split,
        "prompt_stats": args.prompt_stats,
        "n_train_windows": len(train_starts), "n_val_windows": len(val_starts),
        "n_test_windows": len(test_starts),
        "params_total": total, "params_trainable": trainable,
        "best_val_mse_std": best_val,
        "test_mse_std": test_std, "test_mse_raw": test_raw,
        "test_mse_raw_endpoint": test_endpoint["raw"], "test_mse_std_endpoint": test_endpoint["std"],
        "train_wall_s": round(train_s, 1), "infer_s_per_sample": round(infer_s, 5),
        "peak_gpu_mb": (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                        if device.type == "cuda" else None),
        "space_trained_in": args.space,
        "channel_independent": args.channel_independent,
        "arm": args.arm,
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
    ap.add_argument("--micro-batch-size", type=int, default=0,
                    help="split each optimizer step into micro-batches of this many samples "
                         "(0 = no split); needed for channel-independent runs on wide inputs")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-decay", type=float, default=1.0,
                    help="multiply the base lr by this after every epoch, as "
                         "Time-Series-Library's adjust_learning_rate does "
                         "(its default is 0.5; 1.0 = constant lr)")
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
    ap.add_argument("--prompt-stats", choices=["raw", "normalized"], default="raw",
                    help="units the prompt's input statistics describe. raw: the dBm-scale "
                         "input (default). normalized: post-RevIN z-scores, which is what "
                         "Time-LLM does and what runs before 2026-11-04 used.")
    ap.add_argument("--arm", choices=["timellm", "ofa", "dlinear"], default="timellm",
                    help="timellm: reprogramming + prompt, frozen backbone. "
                         "ofa: no reprogramming/prompt, ln+wpe tuned. dlinear: decomposition-linear.")
    ap.add_argument("--channel-independent", action="store_true",
                    help="fold channels into the batch (Time-LLM's arrangement) instead of "
                         "making the whole multivariate patch one token (the article's eq. 1)")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="where trained weights are written; empty string disables saving")
    ap.add_argument("--split", choices=["literal", "tslib"], default="tslib",
                    help="how to read the article's 7:3. tslib (default): 70/10/20 chronological, "
                         "Time-Series-Library's Dataset_Custom arrangement, which is what reproduces "
                         "the published 25-channel row. literal: last 30%% is all test, "
                         "validation carved from the training span (the earlier campaign)")
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
    done = {(r["dataset"], r["horizon"], r["seed"], r["backbone"],
             r.get("space_trained_in", "std"), bool(r.get("channel_independent", False)),
             r.get("arm", "timellm"), r.get("split", "literal"),
             r.get("prompt_stats", "normalized"))
            for r in rows}
    for ds in args.datasets:
        for horizon in args.horizons:
            for seed in args.seeds:
                key = (ds, horizon, seed, args.backbone, args.space, args.channel_independent,
                       args.arm, args.split, args.prompt_stats)
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
