"""Compare short-timescale noise statistics of the SpectrumLLM ElectroSense files with POWDER 1-min,
15-min linear-power averages and 15-min decimated samples (run in the cluster head container; CSV to stdout)."""
import numpy as np, pandas as pd, glob, os
def stats(v):
    d = np.diff(v, axis=0); d = d - d.mean(axis=0)
    g0 = (d*d).mean(0); g1 = (d[1:]*d[:-1]).mean(0); g2 = (d[2:]*d[:-2]).mean(0)
    rho1, rho2 = g1/g0, g2/g0
    eps = np.maximum(-g1, 0); var = v.var(0)
    return dict(rho1=np.median(rho1), rho2=np.median(rho2), noise_share_of_var=np.median(eps/var),
                noise_share_of_diffvar=np.median(2*eps/g0))
rows = []
for f in sorted(glob.glob("/workspace/spectrum-usage/data/electrosense_raw/*.csv")):
    df = pd.read_csv(f); v = df.drop(columns=[c for c in df.columns if not c.replace('.','',1).replace('-','',1).isdigit()]).to_numpy(float)
    s = stats(v); s.update(src=os.path.basename(f), n=len(v), ch=v.shape[1], vmin=v.min(), vmax=v.max(),
                           skew=float(pd.DataFrame(v).skew().median()),
                           frac_unique=float(np.mean([len(np.unique(c))/len(c) for c in v.T])))
    rows.append(s)
root = "/workspace/spectrum-usage/data/powder"
for node in ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1"):
    for band in ("600_800", "2400_2600"):
        raw = pd.read_csv(f"{root}/{node}/20261015T1434Z/{band}/power_1mhz_avg_per_minute.csv")
        raw.index = pd.to_datetime(raw.pop("timestamp_utc"), utc=True).dt.floor("min")
        raw = raw[~raw.index.duplicated()].asfreq("min")
        lin = 10 ** (raw / 10)
        blk = 10 * np.log10(lin.resample("15min").mean())
        for name, v in (("1min", raw), ("15min_avg", blk), ("15min_sample", raw.iloc[::15])):
            v = v.dropna().to_numpy(float)
            s = stats(v); s.update(src=f"POWDER {node} {band} {name}", n=len(v), ch=v.shape[1], vmin=v.min(), vmax=v.max(),
                                   skew=float(pd.DataFrame(v).skew().median()),
                                   frac_unique=float(np.mean([len(np.unique(c))/len(c) for c in v.T])))
            rows.append(s)

import sys; pd.DataFrame(rows).set_index("src").round(4).to_csv(sys.stdout)
