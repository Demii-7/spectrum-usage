"""Per-region points behind the October entropy figure: for every (resolution, band 600-800, model, non-noise region) the region's
power-difference entropy, the model's test MAE and the lookback-mean MAE (dB, means over seeds and the common horizons), and their
difference. Same inputs and definitions as ``entropy_advantage_october.py`` (it imports that script's readers).

    python evaluation/analysis/entropy_advantage_regions_october.py [--out evaluation/results/entropy_advantage_october_regions.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import entropy_advantage_october as E  # noqa: E402
from training.ray.storage import spectrum_s3_client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entropy", default="evaluation/results/october_entropy.csv")
    parser.add_argument("--plan", default="evaluation/analysis/plan_regions_600_800.csv")
    parser.add_argument("--out", default="evaluation/results/entropy_advantage_october_regions.csv")
    args = parser.parse_args()
    client = spectrum_s3_client()
    entropy, plan = pd.read_csv(args.entropy), pd.read_csv(args.plan)
    keep = plan[~plan.is_noise_floor.astype(str).str.lower().eq("true")]
    rows = []
    band = "600_800"
    for resolution in ("1min", "15min"):
        ent_reg = entropy[(entropy.resolution == resolution) & (entropy.band == band) & (entropy.level == "region")].set_index("id").entropy
        base_key = f"median-study/sweep/{resolution}/{band}/per-bin/mean_k{E.BASELINE_K[resolution]}.csv"
        base_table = E.read_csv(client, base_key)
        base_h = set(int(h) for h in base_table[base_table["split"].astype(str).str.endswith("test")]["horizon"].unique())
        for label, keys in sorted(E.model_tables(client, resolution, band).items()):
            try:
                _, hs = E.per_bin_error(client, keys[:1], None)
                common = hs & base_h
                if not common:
                    continue
                err, _ = E.per_bin_error(client, keys, common)
                base, _ = E.per_bin_error(client, [base_key], common)
            except Exception as error:
                print(f"[skip] {resolution} {label}: {error}", file=sys.stderr)
                continue
            for r in keep.itertuples():
                sel = [b for b in err.index if r.start_mhz - 1e-6 <= b <= r.end_mhz + 1e-6]
                if sel:
                    rows.append({"resolution": resolution, "model": label, "n_seeds": len(keys), "horizons": sorted(common), "region": r.band_id,
                                 "start_mhz": r.start_mhz, "end_mhz": r.end_mhz, "n_bins": len(sel), "entropy": float(ent_reg.get(r.band_id, float("nan"))),
                                 "mae_db": float(err.loc[sel].mean()), "lookbackmean_mae_db": float(base.loc[sel].mean())})
    table = pd.DataFrame(rows)
    table["gap_db"] = table.mae_db - table.lookbackmean_mae_db
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.groupby(["resolution", "model"]).agg(n=("region", "count"), seeds=("n_seeds", "first"), gap=("gap_db", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
