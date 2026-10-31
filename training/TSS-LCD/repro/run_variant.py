# run_variant.py - leakage-controlled TSS-LCD runner.
# Model code, training loops and evaluation protocol are taken verbatim from the
# authors' train.py / test.py. The ONLY thing that varies is preprocessing:
#   --smooth centered : savgol(51,2) over the whole series  -> their original (LEAKY:
#                       a centred window at index i consumes i+-25, so the context
#                       window carries information from after the forecast origin)
#   --smooth causal   : savgol(51,2) evaluated at the window's LAST point, so the
#                       value at index i uses only i-50..i                 (leak-free)
#   --smooth none     : no smoothing at all                               (leak-free)
#   --scaler perfile  : min-max refit on whichever file is loaded -> their original
#                       (test statistics leak into test normalisation)
#   --scaler trainfit : min-max fit on train, applied unchanged to test   (leak-free)
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from scipy.signal import savgol_filter, savgol_coeffs

from Context2CondNew import ContextTransformerAE
from F2Cnet import FutureAutoencoder
from NoiseNet import DiffusionModel, inference
from DataSetPrepare import MinMaxScalerCustom, MultiBandDataset

W, PO = 51, 2


def smooth_series(x, mode):
    if mode == 'none':
        return x.copy()
    if mode == 'centered':
        return savgol_filter(x, window_length=W, polyorder=PO, axis=0, mode='interp')
    if mode == 'causal':
        # polynomial fit over the trailing window, evaluated at its last sample
        c = savgol_coeffs(W, PO, pos=W - 1, use='dot').astype(np.float64)
        T, D = x.shape
        pad = np.repeat(x[:1], W - 1, axis=0)          # edge-pad with the first row
        xp = np.concatenate([pad, x], axis=0)
        out = np.empty_like(x, dtype=np.float64)
        for t in range(T):
            out[t] = c @ xp[t:t + W]
        return out
    raise ValueError(mode)


def load_matrix(path, D):
    # their loader: pd.read_csv WITHOUT header=None, so row 0 is consumed as a header
    df = pd.read_csv(path)
    return df.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)[:, :D]


def build(path, D, C, Fh, bs, shuffle, smooth, scaler):
    raw = load_matrix(path, D)
    if scaler is None:
        scaler = MinMaxScalerCustom(); scaler.fit(raw)
    y = smooth_series(scaler.transform(raw), smooth).astype(np.float32)
    ds = MultiBandDataset(y, C, Fh)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle), scaler, y


# ---- training loops: verbatim from the authors' train.py --------------------
def train_autoencoder(model, loader, device, epochs, lr, tag):
    model.to(device); opt = optim.Adam(model.parameters(), lr=lr); model.train()
    for ep in range(epochs):
        tot = 0.0
        for ctx, fut in loader:
            inp = ctx if isinstance(model, ContextTransformerAE) else fut
            inp = inp.to(device)
            recon, _ = model(inp); loss = F.mse_loss(recon, inp)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 25 == 0 or ep == epochs - 1:
            print(f'[{tag}] AE {model.__class__.__name__} ep {ep+1}/{epochs} loss={tot/len(loader):.6f}', flush=True)
    return model


def train_diffusion(ae_ctx, ae_fut, loader, device, a, tag):
    m = DiffusionModel(cond_dim=a.latent_dim, latent_dim=a.latent_dim,
                       n_timestep=a.n_timesteps, device=device).to(device)
    opt = optim.Adam(m.parameters(), lr=a.lr); ae_ctx.eval(); ae_fut.eval()
    for ep in range(a.diff_epochs):
        tot = 0.0
        for ctx, fut in loader:
            ctx, fut = ctx.to(device), fut.to(device)
            _, z_ctx = ae_ctx(ctx); _, z_fut = ae_fut(fut)
            t = torch.randint(0, a.n_timesteps, (z_fut.size(0),), device=device)
            noise = torch.randn_like(z_fut)
            loss = F.mse_loss(m(m.q_sample(z_fut, t, noise), z_ctx, t), noise)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 100 == 0 or ep == a.diff_epochs - 1:
            print(f'[{tag}] Diff ep {ep+1}/{a.diff_epochs} loss={tot/len(loader):.6f}', flush=True)
    return m


def metrics(pred, gt):
    e = pred - gt
    return dict(mse=float((e**2).mean()), rmse=float(np.sqrt((e**2).mean())),
                mae=float(np.abs(e).mean()),
                mape_pct=float(100*np.mean(np.abs(e/(gt+1e-8)))))


def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = a.tag; t0 = time.time()
    print(f'=== variant {tag}: smooth={a.smooth} scaler={a.scaler} ===', flush=True)

    tr_loader, tr_scaler, _ = build(a.csv_train, a.D, a.context_length, a.future_length,
                                    a.batch_size, True, a.smooth, None)
    ae_ctx = ContextTransformerAE(seq_len=a.context_length, input_dim=a.D,
                                  latent_dim=a.latent_dim).to(dev)
    ae_fut = FutureAutoencoder(a.future_length, a.D, a.latent_dim).to(dev)
    ae_ctx = train_autoencoder(ae_ctx, tr_loader, dev, a.ae_epochs_context, a.lr, tag)
    ae_fut = train_autoencoder(ae_fut, tr_loader, dev, a.ae_epochs_future, a.lr, tag)
    diff = train_diffusion(ae_ctx, ae_fut, tr_loader, dev, a, tag)
    os.makedirs(a.out, exist_ok=True)
    for n, m in [('ae_ctx', ae_ctx), ('ae_fut', ae_fut), ('diffusion', diff)]:
        torch.save(m.state_dict(), os.path.join(a.out, f'{n}.pth'))
    t_train = time.time() - t0

    # ---- evaluation: authors' protocol, t+1 only -----------------------------
    te_scaler = None if a.scaler == 'perfile' else tr_scaler
    te_loader, te_scaler, y_te = build(a.csv_test, a.D, a.context_length, a.future_length,
                                       a.batch_size, False, a.smooth, te_scaler)
    ae_ctx.eval(); ae_fut.eval(); diff.eval()
    g, p = [], []
    with torch.no_grad():
        for ctx, fut in te_loader:
            pr = inference(ae_ctx, ae_fut, diff, ctx.to(dev), dev)
            g.append(fut[:, 0, :].cpu().numpy()); p.append(pr[:, 0, :])
    gt = te_scaler.inverse_transform(np.concatenate(g, 0))
    pred = te_scaler.inverse_transform(np.concatenate(p, 0))

    # references on the identical target, from the unmasked preprocessed series
    C, N = a.context_length, gt.shape[0]
    yd = te_scaler.inverse_transform(y_te)
    persist = yd[C-1:C-1+N]                                     # last observed value
    lbmean = np.stack([yd[i:i+C].mean(0) for i in range(N)])    # mean of the 50-step context

    res = dict(tag=tag, smooth=a.smooth, scaler=a.scaler, n_test=int(N),
               epochs=[a.ae_epochs_context, a.ae_epochs_future, a.diff_epochs],
               train_seconds=round(t_train, 1),
               model=metrics(pred, gt), persistence=metrics(persist, gt),
               lookback_mean=metrics(lbmean, gt))
    res['skill_vs_persistence'] = 1 - res['model']['mse']/res['persistence']['mse']
    res['skill_vs_lookback_mean'] = 1 - res['model']['mse']/res['lookback_mean']['mse']
    res['gt_lag1_mean_abs_dB'] = float(np.abs(np.diff(gt, axis=0)).mean())

    pd.DataFrame(gt).to_csv(os.path.join(a.out, 'gt_mat.csv'), index=False)
    pd.DataFrame(pred).to_csv(os.path.join(a.out, 'pred_matours.csv'), index=False)
    json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    q = argparse.ArgumentParser()
    q.add_argument('--tag', required=True); q.add_argument('--out', required=True)
    q.add_argument('--smooth', choices=['centered', 'causal', 'none'], required=True)
    q.add_argument('--scaler', choices=['perfile', 'trainfit'], required=True)
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
