# run_ours.py - OUR TSS-LCD implementation (models/TSSLCD.py) on the authors'
# data, under leak-free preprocessing. Preprocessing, evaluation protocol and
# reference baselines are identical to run_variant.py, so results are directly
# comparable to the leaky/leak-free runs of the authors' own code.
#
# Our three stages differ structurally from theirs and this is deliberate:
#   theirs : context AE (reconstruct context) -> future AE -> diffusion
#   ours   : future AE -> conditioner REGRESSES the future latent -> diffusion
# Our stage 2 has no context-reconstruction loss; it optimises a different
# objective against a different target. That is the main thing this run measures.
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from scipy.signal import savgol_filter, savgol_coeffs

from DataSetPrepare import MinMaxScalerCustom, MultiBandDataset
from TSSLCD import (TSSConditionConstructor, LatentSpaceEncoder,
                    LatentSpaceDecoder, DiffusionModel)

W, PO = 51, 2


def smooth_series(x, mode):
    if mode == 'none':
        return x.copy()
    if mode == 'centered':
        return savgol_filter(x, window_length=W, polyorder=PO, axis=0, mode='interp')
    if mode == 'causal':
        c = savgol_coeffs(W, PO, pos=W - 1, use='dot').astype(np.float64)
        T = x.shape[0]
        xp = np.concatenate([np.repeat(x[:1], W - 1, axis=0), x], axis=0)
        out = np.empty_like(x, dtype=np.float64)
        for t in range(T):
            out[t] = c @ xp[t:t + W]
        return out
    raise ValueError(mode)


def build(path, D, C, Fh, bs, shuffle, smooth, scaler):
    raw = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').values.astype(np.float32)[:, :D]
    if scaler is None:
        scaler = MinMaxScalerCustom(); scaler.fit(raw)
    y = smooth_series(scaler.transform(raw), smooth).astype(np.float32)
    return DataLoader(MultiBandDataset(y, C, Fh), batch_size=bs, shuffle=shuffle), scaler, y


def metrics(pred, gt):
    e = pred - gt
    return dict(mse=float((e**2).mean()), rmse=float(np.sqrt((e**2).mean())),
                mae=float(np.abs(e).mean()),
                mape_pct=float(100*np.mean(np.abs(e/(gt+1e-8)))))


def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    print(f'=== OURS tag={a.tag} smooth={a.smooth} scaler={a.scaler} ===', flush=True)
    tr, tr_scaler, _ = build(a.csv_train, a.D, a.context_length, a.future_length,
                             a.batch_size, True, a.smooth, None)
    L, Fq = 1, a.D
    cond = TSSConditionConstructor(T_in=a.context_length, L=L, F=Fq, latent_dim=a.latent_dim).to(dev)
    enc = LatentSpaceEncoder(T_out=a.future_length, L=L, F=Fq, latent_dim=a.latent_dim).to(dev)
    dec = LatentSpaceDecoder(T_out=a.future_length, L=L, F=Fq, latent_dim=a.latent_dim).to(dev)
    diff = DiffusionModel(latent_dim=a.latent_dim, n_timestep=a.n_timesteps, device=dev).to(dev)
    n = lambda *ms: sum(p.numel() for m in ms for p in m.parameters())
    print(f'params cond={n(cond):,} ae={n(enc,dec):,} diff={n(diff):,} total={n(cond,enc,dec,diff):,}', flush=True)

    # stage 1 - future autoencoder
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    for ep in range(a.ae_epochs_future):
        tot = 0.0
        for ctx, fut in tr:
            fut = fut.to(dev)
            loss = F.mse_loss(dec(enc(fut)), fut)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 25 == 0 or ep == a.ae_epochs_future - 1:
            print(f'[ours] futureAE ep {ep+1}/{a.ae_epochs_future} loss={tot/len(tr):.6f}', flush=True)

    # stage 2 - conditioner regresses the future latent
    enc.eval()
    opt = optim.Adam(cond.parameters(), lr=a.lr)
    for ep in range(a.cond_epochs):
        tot = 0.0
        for ctx, fut in tr:
            ctx, fut = ctx.to(dev), fut.to(dev)
            with torch.no_grad():
                z_fut = enc(fut)
            loss = F.mse_loss(cond(ctx), z_fut)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 25 == 0 or ep == a.cond_epochs - 1:
            print(f'[ours] conditioner ep {ep+1}/{a.cond_epochs} loss={tot/len(tr):.6f}', flush=True)

    # stage 3 - latent diffusion
    cond.eval()
    opt = optim.Adam(diff.parameters(), lr=a.lr)
    for ep in range(a.diff_epochs):
        tot = 0.0
        for ctx, fut in tr:
            ctx, fut = ctx.to(dev), fut.to(dev)
            with torch.no_grad():
                z_fut = enc(fut); cz = cond(ctx)
            t = torch.randint(0, a.n_timesteps, (z_fut.size(0),), device=dev)
            noise = torch.randn_like(z_fut)
            loss = F.mse_loss(diff(diff.q_sample(z_fut, t, noise), cz, t), noise)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 100 == 0 or ep == a.diff_epochs - 1:
            print(f'[ours] diffusion ep {ep+1}/{a.diff_epochs} loss={tot/len(tr):.6f}', flush=True)
    t_train = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    for nm, m in [('cond', cond), ('enc', enc), ('dec', dec), ('diffusion', diff)]:
        torch.save(m.state_dict(), os.path.join(a.out, f'{nm}.pth'))

    # evaluation - identical protocol to run_variant.py (t+1 only)
    te_scaler = None if a.scaler == 'perfile' else tr_scaler
    te, te_scaler, y_te = build(a.csv_test, a.D, a.context_length, a.future_length,
                                a.batch_size, False, a.smooth, te_scaler)
    enc.eval(); dec.eval(); cond.eval(); diff.eval()
    g, p, zs, zc = [], [], [], []
    with torch.no_grad():
        for ctx, fut in te:
            ctx = ctx.to(dev)
            cz = cond(ctx)
            z = diff.p_sample_loop(cz)
            pr = dec(z)
            g.append(fut[:, 0, :].numpy()); p.append(pr[:, 0, :].cpu().numpy())
            zs.append(z.cpu().numpy()); zc.append(cz.cpu().numpy())
    gt = te_scaler.inverse_transform(np.concatenate(g, 0))
    pred = te_scaler.inverse_transform(np.concatenate(p, 0))
    C, N = a.context_length, gt.shape[0]
    yd = te_scaler.inverse_transform(y_te)
    persist = yd[C-1:C-1+N]
    lbmean = np.stack([yd[i:i+C].mean(0) for i in range(N)])

    res = dict(tag=a.tag, impl='ours', smooth=a.smooth, scaler=a.scaler, n_test=int(N),
               epochs=dict(future_ae=a.ae_epochs_future, conditioner=a.cond_epochs,
                           diffusion=a.diff_epochs),
               params=dict(cond=n(cond), ae=n(enc, dec), diff=n(diff), total=n(cond, enc, dec, diff)),
               train_seconds=round(t_train, 1),
               model=metrics(pred, gt), persistence=metrics(persist, gt),
               lookback_mean=metrics(lbmean, gt),
               latent_std=dict(sampled=float(np.concatenate(zs, 0).std()),
                               condition=float(np.concatenate(zc, 0).std())))
    res['skill_vs_persistence'] = 1 - res['model']['mse']/res['persistence']['mse']
    res['skill_vs_lookback_mean'] = 1 - res['model']['mse']/res['lookback_mean']['mse']
    res['gt_lag1_mean_abs_dB'] = float(np.abs(np.diff(gt, axis=0)).mean())
    pd.DataFrame(gt).to_csv(os.path.join(a.out, 'gt_mat.csv'), index=False)
    pd.DataFrame(pred).to_csv(os.path.join(a.out, 'pred_matours.csv'), index=False)
    json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    q = argparse.ArgumentParser()
    q.add_argument('--tag', default='ours'); q.add_argument('--out', required=True)
    q.add_argument('--smooth', choices=['centered', 'causal', 'none'], default='causal')
    q.add_argument('--scaler', choices=['perfile', 'trainfit'], default='trainfit')
    q.add_argument('--csv-train', default='./dataset/trainDataset.csv')
    q.add_argument('--csv-test', default='./dataset/testDataset.csv')
    q.add_argument('--context-length', type=int, default=50)
    q.add_argument('--future-length', type=int, default=10)
    q.add_argument('--D', type=int, default=750)
    q.add_argument('--latent-dim', type=int, default=32)
    q.add_argument('--n-timesteps', type=int, default=1000)
    q.add_argument('--batch-size', type=int, default=256)
    q.add_argument('--lr', type=float, default=1e-4)
    q.add_argument('--ae-epochs-future', type=int, default=300)
    q.add_argument('--cond-epochs', type=int, default=300)
    q.add_argument('--diff-epochs', type=int, default=2000)
    main(q.parse_args())
