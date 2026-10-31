import numpy as np, pandas as pd, torch, json
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
from Context2CondNew import ContextTransformerAE
from F2Cnet import FutureAutoencoder
from NoiseNet import DiffusionModel, inference
from DataSetPrepare import MinMaxScalerCustom, MultiBandDataset

dev = torch.device('cuda')
D, C, Fh, LD, NT = 750, 50, 10, 32, 1000
raw = pd.read_csv('./dataset/testDataset.csv').apply(pd.to_numeric, errors='coerce').values.astype(np.float32)[:, :D]
sc = MinMaxScalerCustom(); sc.fit(raw)
y = savgol_filter(sc.transform(raw), 51, 2, axis=0, mode='interp').astype(np.float32)
ld = DataLoader(MultiBandDataset(y, C, Fh), batch_size=256, shuffle=False)

ae_ctx = ContextTransformerAE(seq_len=C, input_dim=D, latent_dim=LD).to(dev)
ae_fut = FutureAutoencoder(Fh, D, LD).to(dev)
dif = DiffusionModel(cond_dim=LD, latent_dim=LD, n_timestep=NT, device=dev).to(dev)
for m, f in [(ae_ctx,'ae_ctx'), (ae_fut,'ae_fut'), (dif,'diffusion')]:
    m.load_state_dict(torch.load(f'./run_paper/{f}.pth', map_location=dev)); m.eval()

acc = {k: [] for k in ['gt','ae_rt','dec_zctx','full','z_fut','z_ctx','z_samp']}
with torch.no_grad():
    for ctx, fut in ld:
        ctx, fut = ctx.to(dev), fut.to(dev)
        rec_f, z_fut = ae_fut(fut)            # AE round-trip ceiling
        _,     z_ctx = ae_ctx(ctx)            # condition latent
        z_s = dif.p_sample_loop(z_ctx)        # sampled latent
        dec_c = ae_fut.decode(z_ctx) if hasattr(ae_fut,'decode') else None
        full  = inference(ae_ctx, ae_fut, dif, ctx, dev)
        acc['gt'].append(fut[:,0,:].cpu().numpy())
        acc['ae_rt'].append(rec_f[:,0,:].cpu().numpy())
        acc['full'].append(full[:,0,:])
        acc['z_fut'].append(z_fut.cpu().numpy()); acc['z_ctx'].append(z_ctx.cpu().numpy())
        acc['z_samp'].append(z_s.cpu().numpy())
        if dec_c is not None: acc['dec_zctx'].append(dec_c[:,0,:].cpu().numpy())
cat = {k: np.concatenate(v,0) for k,v in acc.items() if v}
inv = sc.inverse_transform
gt = inv(cat['gt'])
out = {}
for k in ['ae_rt','dec_zctx','full']:
    if k in cat:
        e = inv(cat[k]) - gt; out[k+'_mse_dB'] = float((e**2).mean())
out['persistence_mse_dB'] = float(((inv(y[C-1:C-1+len(gt)]) - gt)**2).mean())
for k in ['z_fut','z_ctx','z_samp']:
    out[k+'_std'] = float(cat[k].std()); out[k+'_absmean'] = float(np.abs(cat[k]).mean())
out['z_ctx_vs_z_fut_mse'] = float(((cat['z_ctx']-cat['z_fut'])**2).mean())
print(json.dumps(out, indent=1))
json.dump(out, open('./diagnose_paper.json','w'), indent=1)
