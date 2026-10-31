# run_paper_faithful.py - TSS-LCD built to the PAPER's specification.
#
# Every setting below is Table II / Section V-A of Cheng et al., IEEE TCCN 2026.
# Where the paper is silent, the authors' released code decides, and the choice
# is labelled [CODE]. Where both are silent it is labelled [UNDETERMINED].
#
#   T1=50, T2=10, N=1000 diffusion steps, L=3 nodes, F=250 bins,
#   2 transformer layers, 4 attention heads, hidden d=1024, dropout 0.1,
#   latent dim 16, LSE/LSD hidden 1024, lr 1e-4, batch 32, 500 epochs, Adam,
#   25% of the historical input zeroed, 7:3 split, early stopping on val loss.
#
# NO SMOOTHING. The paper never mentions a filter; the authors' code applies
# savgol(51,2) before windowing, which both redefines the target and leaks up to
# 25 steps of future into the context. Omitted here by instruction.
#
# [CODE]  min-max normalisation to [0,1] per frequency - the paper does not say
#         how inputs are scaled, but the released code does this and the metrics
#         are reported in dBm, so we scale and invert.
# [CODE]  5 attention heads are impossible at the paper's own numbers unless the
#         model dimension is 1024: 750 % 4 != 0 and the code asserts divisibility.
#         Our branches project to hidden=1024 first, so Table II's "4 heads" and
#         "hidden 1024" are both honoured.
# [UNDETERMINED] "Training epochs 500" is one number for a three-stage pipeline.
#         Read here as 500 per stage; early stopping usually ends it sooner.
# [UNDETERMINED] The paper divides "samples" 7:3 but never says whether the
#         sliding windows are shuffled first. --split selects; with stride-1
#         windows, 'random' puts near-duplicates of each test window into train.
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, torch.optim as optim
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, Dataset, Subset
import sys
sys.path.insert(0, '/work')
from models.TSSLCD import (TSSConditionConstructor, LatentSpaceEncoder,
                           LatentSpaceDecoder, DiffusionModel)


class MinMax:
    def fit(self, x):
        self.lo, self.hi = x.min(0), x.max(0)
        self.rng = np.where((self.hi - self.lo) == 0, 1e-8, self.hi - self.lo)
        return self
    def to(self, x):   return (x - self.lo) / self.rng
    def back(self, x): return x * self.rng + self.lo


class Windows(Dataset):
    """Stride-1 (context, future) pairs. 25% of the context is zeroed (paper)."""
    def __init__(self, y, C, H, mask_ratio=0.25):
        self.y, self.C, self.H, self.m = y, C, H, mask_ratio
        self.n = len(y) - C - H + 1
    def __len__(self): return self.n
    def __getitem__(self, i):
        ctx = torch.from_numpy(self.y[i:i+self.C]).float()
        fut = torch.from_numpy(self.y[i+self.C:i+self.C+self.H]).float()
        ctx = ctx.masked_fill(torch.rand_like(ctx) < self.m, 0.0)
        return ctx, fut


def read(p, D):
    return pd.read_csv(p).apply(pd.to_numeric, errors='coerce').values.astype(np.float32)[:, :D]


def met(p, g):
    e = p - g
    return dict(mse=float((e**2).mean()), rmse=float(np.sqrt((e**2).mean())),
                mae=float(np.abs(e).mean()), mape_pct=float(100*np.mean(np.abs(e/(g+1e-8)))))


def fit_stage(name, params, step_fn, tr, va, epochs, lr, patience, out, tag):
    """Adam + early stopping on validation loss; keeps the best weights."""
    opt = optim.Adam(params, lr=lr)
    best, best_ep, bad, hist = float('inf'), -1, 0, []
    snap = None
    for ep in range(epochs):
        tot = 0.0
        for b in tr:
            loss = step_fn(b, True)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        with torch.no_grad():
            vl = float(np.mean([step_fn(b, False).item() for b in va]))
        hist.append({'epoch': ep+1, 'train': tot/len(tr), 'val': vl})
        if vl < best - 1e-6:
            best, best_ep, bad = vl, ep+1, 0
            snap = [p.detach().clone() for p in params]
        else:
            bad += 1
            if bad >= patience:
                print(f'[{tag}] {name} early stop at {ep+1}, best val {best:.6f} @ {best_ep}', flush=True)
                break
        if ep % 25 == 0:
            print(f'[{tag}] {name} ep {ep+1}/{epochs} train={tot/len(tr):.6f} val={vl:.6f}', flush=True)
    if snap is not None:
        with torch.no_grad():
            for p, s in zip(params, snap): p.copy_(s)
    json.dump(hist, open(os.path.join(out, f'history_{name}.json'), 'w'))
    return dict(best_val=best, best_epoch=best_ep, epochs_run=len(hist))


def main(a):
    dev = torch.device('cuda'); t0 = time.time(); C, H = 50, 10
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print(f'=== paper-faithful | split={a.split} seed={a.seed} ===', flush=True)

    raw = np.concatenate([read(a.csv_train, a.D), read(a.csv_test, a.D)], 0)
    n_all = len(raw) - C - H + 1
    idx = np.arange(n_all)
    if a.split == 'random':
        idx = np.random.RandomState(a.seed).permutation(n_all)
    cut = int(round(0.7 * n_all))                     # paper: 7:3
    tr_i, te_i = idx[:cut], idx[cut:]
    vcut = int(round(0.9 * len(tr_i)))                # val carved from train for early stopping
    tr_i, va_i = tr_i[:vcut], tr_i[vcut:]

    sc = MinMax().fit(raw[:max(tr_i)+C+H])            # [CODE] scale on training span only
    y = sc.to(raw)
    if a.smooth:
        # ABLATION ONLY. The paper never mentions a filter; the authors' code applies
        # this centred savgol before windowing, which redefines the target AND leaks
        # up to 25 future steps into the context. Enabled only to measure that effect.
        y = savgol_filter(y, window_length=51, polyorder=2, axis=0, mode='interp')
    y = y.astype(np.float32)
    ds = Windows(y, C, H, mask_ratio=0.25)
    mk = lambda ix, sh: DataLoader(Subset(ds, ix.tolist()), batch_size=a.batch, shuffle=sh)
    tr, va, te = mk(tr_i, True), mk(va_i, False), mk(te_i, False)
    print(f'windows={n_all} train={len(tr_i)} val={len(va_i)} test={len(te_i)}', flush=True)

    L, Fq = a.L, a.D // a.L
    cond = TSSConditionConstructor(T_in=C, L=L, F=Fq).to(dev)       # hidden 1024, 4 heads, 2 layers, latent 16
    enc = LatentSpaceEncoder(T_out=H, L=L, F=Fq, latent_dim=16).to(dev)
    dec = LatentSpaceDecoder(T_out=H, L=L, F=Fq, latent_dim=16).to(dev)
    dif = DiffusionModel(latent_dim=16, n_timestep=1000, device=dev).to(dev)
    nparam = lambda *ms: sum(p.numel() for m in ms for p in m.parameters())
    print(f'params cond={nparam(cond)} ae={nparam(enc,dec)} diff={nparam(dif)} '
          f'total={nparam(cond,enc,dec,dif)}', flush=True)

    def s_ae(b, train):
        enc.train(train); dec.train(train)
        fut = b[1].to(dev); return F.mse_loss(dec(enc(fut)), fut)
    def s_cond(b, train):
        cond.train(train); enc.eval()
        ctx, fut = b[0].to(dev), b[1].to(dev)
        with torch.no_grad(): z = enc(fut)
        return F.mse_loss(cond(ctx), z)
    def s_dif(b, train):
        dif.train(train); enc.eval(); cond.eval()
        ctx, fut = b[0].to(dev), b[1].to(dev)
        with torch.no_grad(): z, cz = enc(fut), cond(ctx)
        t = torch.randint(0, 1000, (z.size(0),), device=dev)
        noise = torch.randn_like(z)
        return F.mse_loss(dif(dif.q_sample(z, t, noise), cz, t), noise)

    stages = {}
    stages['ae']   = fit_stage('ae',   list(enc.parameters())+list(dec.parameters()), s_ae,   tr, va, a.epochs, a.lr, a.patience, a.out, a.tag)
    stages['cond'] = fit_stage('cond', list(cond.parameters()),                        s_cond, tr, va, a.epochs, a.lr, a.patience, a.out, a.tag)
    stages['diff'] = fit_stage('diff', list(dif.parameters()),                         s_dif,  tr, va, a.epochs, a.lr, a.patience, a.out, a.tag)
    t_train = time.time() - t0
    for nm, m in [('cond', cond), ('enc', enc), ('dec', dec), ('diffusion', dif)]:
        torch.save(m.state_dict(), os.path.join(a.out, f'{nm}.pth'))

    cond.eval(); enc.eval(); dec.eval(); dif.eval()
    g, p, zs, zc = [], [], [], []
    with torch.no_grad():
        for ctx, fut in te:
            cz = cond(ctx.to(dev)); z = dif.p_sample_loop(cz); pr = dec(z)
            g.append(fut[:, 0, :].numpy()); p.append(pr[:, 0, :].cpu().numpy())
            zs.append(z.cpu().numpy()); zc.append(cz.cpu().numpy())
    gt, pred = sc.back(np.concatenate(g, 0)), sc.back(np.concatenate(p, 0))
    yd = sc.back(y)
    persist = np.stack([yd[i+C-1] for i in te_i])
    lbmean  = np.stack([yd[i:i+C].mean(0) for i in te_i])
    ae_ceiling = None
    with torch.no_grad():
        rr, gg = [], []
        for ctx, fut in te:
            rr.append(dec(enc(fut.to(dev)))[:, 0, :].cpu().numpy()); gg.append(fut[:, 0, :].numpy())
        ae_ceiling = met(sc.back(np.concatenate(rr, 0)), sc.back(np.concatenate(gg, 0)))

    res = dict(tag=a.tag, spec='paper-faithful (Table II); 25% mask; 7:3',
               smoothing=bool(a.smooth),
               split=a.split, seed=a.seed, n_test=int(len(gt)),
               params=dict(cond=nparam(cond), ae=nparam(enc, dec), diff=nparam(dif)),
               stages=stages, train_seconds=round(t_train, 1),
               model=met(pred, gt), persistence=met(persist, gt), lookback_mean=met(lbmean, gt),
               autoencoder_ceiling=ae_ceiling,
               latent_std=dict(sampled=float(np.concatenate(zs, 0).std()),
                               condition=float(np.concatenate(zc, 0).std())),
               gt_lag1_mean_abs_dB=float(np.abs(np.diff(gt, axis=0)).mean()))
    res['skill_vs_persistence'] = 1 - res['model']['mse']/res['persistence']['mse']
    res['skill_vs_lookback_mean'] = 1 - res['model']['mse']/res['lookback_mean']['mse']
    pd.DataFrame(gt).to_csv(os.path.join(a.out, 'gt_mat.csv'), index=False)
    pd.DataFrame(pred).to_csv(os.path.join(a.out, 'pred_matours.csv'), index=False)
    json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    q = argparse.ArgumentParser()
    q.add_argument('--tag', required=True); q.add_argument('--out', required=True)
    q.add_argument('--split', choices=['chrono', 'random'], default='chrono')
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--smooth', action='store_true',
                   help='ABLATION: apply the authors savgol(51,2). Paper does not mention it; default off.')
    q.add_argument('--csv-train', default='/work/data/tsslcd/tsslcd_authors_train.csv')
    q.add_argument('--csv-test',  default='/work/data/tsslcd/tsslcd_authors_test.csv')
    q.add_argument('--D', type=int, default=750); q.add_argument('--L', type=int, default=3)
    q.add_argument('--batch', type=int, default=32)        # Table II
    q.add_argument('--lr', type=float, default=1e-4)       # Table II
    q.add_argument('--epochs', type=int, default=500)      # Table II
    q.add_argument('--patience', type=int, default=25)
    main(q.parse_args())
