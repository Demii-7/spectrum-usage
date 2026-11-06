#!/usr/bin/env python
"""Evaluate a first-commit-campaign checkpoint with the UPSTREAM code.

The first-commit campaign (/home/cc/TSS-LCD) trained the authors' d2d4f29
modules; the upstream/ copies here are byte-identical (F2Cconv1d.py was
renamed F2Cnet.py at HEAD; Context2CondNew.py carries only the inert
TSSLCD_ABLATE guards). The checkpoints therefore load directly, and this
script lets the spectrum-usage harness score them without the other repo.

Examples:
    python3 run_firstcommit_eval.py --arm paper_s0
    python3 run_firstcommit_eval.py --arm spefe_s0 --ablate spefe
    python3 run_firstcommit_eval.py --arm chrono_s0 --no-eval-smooth

Protocol matches the campaign runner (repro/first_commit_repro.py):
their exact evaluation - first future step, optional w6/w3 eval-time
smoothing (on by default), inverse min-max to dBm, persistence baseline.
"""
import argparse
import json
import os
import sys

p = argparse.ArgumentParser()
p.add_argument('--arm', required=True,
               help='dir name under checkpoints/repro/tsslcd_firstcommit/')
p.add_argument('--ablate', default=None, choices=['temfe', 'spefe', 'spafe', 'ffm'],
               help='set TSSLCD_ABLATE to match the arm\'s training-time switch')
p.add_argument('--data', default=None)
p.add_argument('--no-eval-smooth', dest='eval_smooth', action='store_false',
               help='skip their w6/w3 moving averages (report raw single-run error)')
p.add_argument('--mask-seed', type=int, default=1234,
               help='seed for the 10% context masking applied at test time')
p.add_argument('--batch', type=int, default=256)
p.add_argument('--out', default=None, help='metrics json path (default: <ckpt>/metrics_upstream_eval.json)')
args = p.parse_args()

if args.ablate:
    os.environ['TSSLCD_ABLATE'] = args.ablate      # must be set BEFORE the import

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, 'upstream'))   # the authors' modules

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import torch                                         # noqa: E402
from scipy.signal import savgol_filter               # noqa: E402
from torch.utils.data import DataLoader, Subset      # noqa: E402

from Context2CondNew import ContextTransformerAE     # noqa: E402
from F2Cnet import FutureAutoencoder                 # noqa: E402
from NoiseNet import DiffusionModel, inference       # noqa: E402
from DataSetPrepare import MinMaxScalerCustom, MultiBandDataset  # noqa: E402

C, FH, D, LATENT, STEPS = 50, 10, 750, 32, 1000
CKPT = os.path.join(REPO, 'checkpoints', 'repro', 'tsslcd_firstcommit', args.arm)
DATA = args.data or os.path.join(REPO, 'data', 'tsslcd', 'tsslcd_authors_source.csv')
assert os.path.isdir(CKPT), f'no checkpoint dir: {CKPT}'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- their preprocessing: full-series min-max + savgol(51,2) ----
raw = pd.read_csv(DATA).values.astype(np.float32)[:, :D]   # headerless: row 0 eaten, as in their loader
sc = MinMaxScalerCustom(); sc.fit(raw)
y = savgol_filter(sc.transform(raw), window_length=51, polyorder=2,
                  axis=0, mode='interp').astype(np.float32)
ds = MultiBandDataset(y, C, FH)
N = len(ds)
te_idx = list(range(int(N * 0.555), N))                    # the first commit's test range

# ---- models + our weights ----
ae_ctx = ContextTransformerAE(seq_len=C, input_dim=D, latent_dim=LATENT).to(device)
ae_fut = FutureAutoencoder(FH, D, LATENT).to(device)
diff = DiffusionModel(cond_dim=LATENT, latent_dim=LATENT, n_timestep=STEPS,
                      device=device).to(device)
ae_ctx.load_state_dict(torch.load(os.path.join(CKPT, 'ae_ctx.pth'), map_location=device))
ae_fut.load_state_dict(torch.load(os.path.join(CKPT, 'ae_fut.pth'), map_location=device))
diff.load_state_dict(torch.load(os.path.join(CKPT, 'diffusion.pth'), map_location=device))
ae_ctx.eval(); ae_fut.eval(); diff.eval()

# ---- their evaluation ----
torch.manual_seed(args.mask_seed)                          # test-time masking is random
loader = DataLoader(Subset(ds, te_idx), batch_size=args.batch, shuffle=False)
preds, gts = [], []
with torch.no_grad():
    for ctx, fut in loader:
        pred = inference(ae_ctx, ae_fut, diff, ctx.to(device), device)  # (B, FH, D)
        preds.append(pred[:, 0, :]); gts.append(fut[:, 0, :].numpy())
P, G = np.concatenate(preds), np.concatenate(gts)

if args.eval_smooth:
    k1, k2 = np.ones(6) / 6, np.ones(3) / 3                # their w6/w3 eval-time averages
    for i in range(P.shape[1]):
        P[:, i] = np.convolve(P[:, i], k1, mode='same')
        G[:, i] = np.convolve(G[:, i], k2, mode='same')

pred_dbm, gt_dbm = sc.inverse_transform(P), sc.inverse_transform(G)
pers = np.stack([y[i + C - 1] for i in te_idx])            # last observed (smoothed) value
pers_dbm = sc.inverse_transform(pers)
gt_for_pers = sc.inverse_transform(np.stack([y[i + C] for i in te_idx]))


def met(a, b):
    e = a - b
    return dict(mse=float(np.mean(e ** 2)), rmse=float(np.sqrt(np.mean(e ** 2))),
                mae=float(np.mean(np.abs(e))),
                mape_pct=float(np.mean(np.abs(e / np.abs(b))) * 100))


res = dict(arm=args.arm, ablate=args.ablate, eval_smooth=args.eval_smooth,
           n_test=len(te_idx), model=met(pred_dbm, gt_dbm),
           persistence=met(pers_dbm, gt_for_pers),
           data=os.path.basename(DATA), code='upstream (byte-identical to d2d4f29)')
out = args.out or os.path.join(CKPT, 'metrics_upstream_eval.json')
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
print(f'wrote {out}')
