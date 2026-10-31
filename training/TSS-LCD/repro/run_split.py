# run_split.py - tests whether a RANDOM split over pooled sliding windows
# explains the paper's numbers.
#   --split chrono : train file -> train, test file -> test (contiguous, no overlap)
#   --split random : pool every stride-1 window from train+test concatenated,
#                    shuffle, then take 7:3 -- as the paper's wording implies
#                    ("These samples are divided ... using a 7:3 split ratio",
#                    with 10,021 = 10,080 - 60 + 1 windows pooled before splitting).
# With stride-1 windows, neighbours share 59 of 60 steps, so a random split puts
# near-duplicates of every test window into training.
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from scipy.signal import savgol_filter, savgol_coeffs
from Context2CondNew import ContextTransformerAE
from F2Cnet import FutureAutoencoder
from NoiseNet import DiffusionModel, inference
from DataSetPrepare import MinMaxScalerCustom, MultiBandDataset
W, PO = 51, 2

def smooth_series(x, mode):
    if mode == 'none': return x.copy()
    if mode == 'centered':
        return savgol_filter(x, window_length=W, polyorder=PO, axis=0, mode='interp')
    c = savgol_coeffs(W, PO, pos=W-1, use='dot').astype(np.float64)
    xp = np.concatenate([np.repeat(x[:1], W-1, axis=0), x], axis=0)
    return np.stack([c @ xp[t:t+W] for t in range(x.shape[0])])

def rd(p, D):
    return pd.read_csv(p).apply(pd.to_numeric, errors='coerce').values.astype(np.float32)[:, :D]

def train_ae(model, loader, dev, epochs, lr, tag):
    model.to(dev); opt = optim.Adam(model.parameters(), lr=lr); model.train()
    for ep in range(epochs):
        tot = 0.0
        for ctx, fut in loader:
            inp = (ctx if isinstance(model, ContextTransformerAE) else fut).to(dev)
            rec, _ = model(inp); loss = F.mse_loss(rec, inp)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 50 == 0 or ep == epochs-1:
            print(f'[{tag}] {model.__class__.__name__} ep {ep+1}/{epochs} loss={tot/len(loader):.6f}', flush=True)
    return model

def train_diff(ae_c, ae_f, loader, dev, a, tag):
    m = DiffusionModel(cond_dim=a.latent_dim, latent_dim=a.latent_dim,
                       n_timestep=a.n_timesteps, device=dev).to(dev)
    opt = optim.Adam(m.parameters(), lr=a.lr); ae_c.eval(); ae_f.eval()
    for ep in range(a.diff_epochs):
        tot = 0.0
        for ctx, fut in loader:
            ctx, fut = ctx.to(dev), fut.to(dev)
            with torch.no_grad():
                _, zc = ae_c(ctx); _, zf = ae_f(fut)
            t = torch.randint(0, a.n_timesteps, (zf.size(0),), device=dev)
            noise = torch.randn_like(zf)
            loss = F.mse_loss(m(m.q_sample(zf, t, noise), zc, t), noise)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 200 == 0 or ep == a.diff_epochs-1:
            print(f'[{tag}] diff ep {ep+1}/{a.diff_epochs} loss={tot/len(loader):.6f}', flush=True)
    return m

def met(p, g):
    e = p - g
    return dict(mse=float((e**2).mean()), rmse=float(np.sqrt((e**2).mean())),
                mae=float(np.abs(e).mean()), mape_pct=float(100*np.mean(np.abs(e/(g+1e-8)))))

def main(a):
    dev = torch.device('cuda'); t0 = time.time(); C, Fh = a.context_length, a.future_length
    print(f'=== split={a.split} smooth={a.smooth} seed={a.seed} ===', flush=True)
    if a.split == 'random':
        raw = np.concatenate([rd(a.csv_train, a.D), rd(a.csv_test, a.D)], axis=0)
        sc = MinMaxScalerCustom(); sc.fit(raw)
        y = smooth_series(sc.transform(raw), a.smooth).astype(np.float32)
        ds = MultiBandDataset(y, C, Fh)
        idx = np.random.RandomState(a.seed).permutation(len(ds))
        ntr = int(round(0.7*len(ds)))
        tr_idx, te_idx = idx[:ntr], idx[ntr:]
        tr = DataLoader(Subset(ds, tr_idx.tolist()), batch_size=a.batch_size, shuffle=True)
        te = DataLoader(Subset(ds, te_idx.tolist()), batch_size=a.batch_size, shuffle=False)
        y_eval, eval_idx, sc_eval = y, te_idx, sc
        print(f'pooled windows={len(ds)} train={len(tr_idx)} test={len(te_idx)}', flush=True)
    else:
        rtr = rd(a.csv_train, a.D); sc = MinMaxScalerCustom(); sc.fit(rtr)
        ytr = smooth_series(sc.transform(rtr), a.smooth).astype(np.float32)
        tr = DataLoader(MultiBandDataset(ytr, C, Fh), batch_size=a.batch_size, shuffle=True)
        rte = rd(a.csv_test, a.D); sce = MinMaxScalerCustom(); sce.fit(rte)
        yte = smooth_series(sce.transform(rte), a.smooth).astype(np.float32)
        dte = MultiBandDataset(yte, C, Fh)
        te = DataLoader(dte, batch_size=a.batch_size, shuffle=False)
        y_eval, eval_idx, sc_eval = yte, np.arange(len(dte)), sce

    ae_c = ContextTransformerAE(seq_len=C, input_dim=a.D, latent_dim=a.latent_dim).to(dev)
    ae_f = FutureAutoencoder(Fh, a.D, a.latent_dim).to(dev)
    ae_c = train_ae(ae_c, tr, dev, a.ae_epochs_context, a.lr, a.tag)
    ae_f = train_ae(ae_f, tr, dev, a.ae_epochs_future, a.lr, a.tag)
    dif = train_diff(ae_c, ae_f, tr, dev, a, a.tag)
    t_train = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    for nm, m in [('ae_ctx', ae_c), ('ae_fut', ae_f), ('diffusion', dif)]:
        torch.save(m.state_dict(), os.path.join(a.out, f'{nm}.pth'))

    ae_c.eval(); ae_f.eval(); dif.eval(); g, p = [], []
    with torch.no_grad():
        for ctx, fut in te:
            pr = inference(ae_c, ae_f, dif, ctx.to(dev), dev)
            g.append(fut[:, 0, :].numpy()); p.append(pr[:, 0, :])
    gt = sc_eval.inverse_transform(np.concatenate(g, 0))
    pred = sc_eval.inverse_transform(np.concatenate(p, 0))
    yd = sc_eval.inverse_transform(y_eval)
    persist = np.stack([yd[i+C-1] for i in eval_idx])
    lbmean = np.stack([yd[i:i+C].mean(0) for i in eval_idx])
    res = dict(tag=a.tag, split=a.split, smooth=a.smooth, seed=a.seed, n_test=int(gt.shape[0]),
               epochs=[a.ae_epochs_context, a.ae_epochs_future, a.diff_epochs],
               batch_size=a.batch_size, latent_dim=a.latent_dim,
               train_seconds=round(t_train, 1), model=met(pred, gt),
               persistence=met(persist, gt), lookback_mean=met(lbmean, gt))
    res['skill_vs_persistence'] = 1 - res['model']['mse']/res['persistence']['mse']
    res['skill_vs_lookback_mean'] = 1 - res['model']['mse']/res['lookback_mean']['mse']
    pd.DataFrame(gt).to_csv(os.path.join(a.out, 'gt_mat.csv'), index=False)
    pd.DataFrame(pred).to_csv(os.path.join(a.out, 'pred_matours.csv'), index=False)
    json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1), flush=True)

if __name__ == '__main__':
    q = argparse.ArgumentParser()
    q.add_argument('--tag', required=True); q.add_argument('--out', required=True)
    q.add_argument('--split', choices=['chrono', 'random'], required=True)
    q.add_argument('--smooth', choices=['centered', 'causal', 'none'], default='centered')
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--csv-train', default='./dataset/trainDataset.csv')
    q.add_argument('--csv-test', default='./dataset/testDataset.csv')
    q.add_argument('--context-length', type=int, default=50)
    q.add_argument('--future-length', type=int, default=10)
    q.add_argument('--D', type=int, default=750)
    q.add_argument('--latent-dim', type=int, default=32)
    q.add_argument('--n-timesteps', type=int, default=1000)
    q.add_argument('--batch-size', type=int, default=256)
    q.add_argument('--lr', type=float, default=1e-4)
    q.add_argument('--ae-epochs-context', type=int, default=300)
    q.add_argument('--ae-epochs-future', type=int, default=300)
    q.add_argument('--diff-epochs', type=int, default=2000)
    main(q.parse_args())
