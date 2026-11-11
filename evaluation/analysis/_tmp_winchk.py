import os, sys
os.chdir("/workspace/spectrum-usage"); sys.path.insert(0, "/workspace/spectrum-usage")
from training.ray import tsslcd_authors as T
import numpy as np, pandas as pd
for end in ("2026-11-03T20:00:00Z", "2026-11-03T18:46:00Z"):
    T.RANGES["test"] = (T.RANGES["test"][0], end)
    raw, model, bounds, _ = T.build_partition("/workspace/spectrum-usage", "test", T.build_partition("/workspace/spectrum-usage", "train", None, "off")[3], "off")
    st = T.window_starts(bounds, T.CONTEXT + T.FUTURE, 1)
    print("test end", end, "windows", len(st), "segments", len(bounds), "rows", raw.shape[0], flush=True)
import pandas as pd
r = pd.read_csv("/workspace/spectrum-usage/data/powder/humanities-nuc1/20261025T1355Z/600_800/power_1mhz_avg_per_minute.csv", usecols=["timestamp_utc"])
print("humanities test-run file: first", r.timestamp_utc.iloc[0], "last", r.timestamp_utc.iloc[-1], "rows", len(r))
