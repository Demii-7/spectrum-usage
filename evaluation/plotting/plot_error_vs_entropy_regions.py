#!/usr/bin/env python3
"""Plot per-region forecast error against annotation-derived entropy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "evaluation/results/tables/ray_2d_per_frequency_metrics.csv")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/annotations")
    parser.add_argument("--entropy", type=Path, default=ROOT / "evaluation/results/tables/entropy_band_metrics_2d.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/figures/error_vs_entropy_regions.png")
    parser.add_argument("--table-output", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def load_annotations(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("**/*.csv"))
    frames = [pd.read_csv(path, dtype={"run_id": str, "band": str}) for path in paths]
    if not frames:
        raise ValueError(f"No annotation CSVs found under {root}")
    annotations = pd.concat(frames, ignore_index=True)
    required = {"band", "region_id", "region_start_mhz", "region_end_mhz"}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotations are missing column(s): {', '.join(sorted(missing))}")
    return annotations.drop_duplicates(["band", "region_id"])


def build_table(args: argparse.Namespace) -> pd.DataFrame:
    metrics = pd.read_csv(args.metrics)
    metrics = metrics.loc[metrics["horizon"].eq(args.horizon)].copy()
    annotations = load_annotations(args.annotations)
    rows = []
    for _, metric in metrics.iterrows():
        matches = annotations.loc[
            annotations["band"].eq("600_800")
            & annotations["region_start_mhz"].le(metric["bin_mhz"])
            & annotations["region_end_mhz"].ge(metric["bin_mhz"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one annotation for {metric['bin_mhz']} MHz, found {len(matches)}")
        region = matches.iloc[0]
        rows.append({**metric.to_dict(), "region_id": int(region["region_id"]),
                     "region_start_mhz": region["region_start_mhz"],
                     "region_end_mhz": region["region_end_mhz"]})
    per_region = pd.DataFrame(rows).groupby(["model", "seed", "region_id", "region_start_mhz", "region_end_mhz"], as_index=False)["mae_db"].mean()
    entropy = pd.read_csv(args.entropy).rename(columns={"model_name": "model"})
    entropy["region_id"] = entropy["band_id"].str.extract(r"R(\d+)").astype(int)
    entropy = entropy[["model", "seed", "region_id", "entropy"]]
    result = per_region.merge(entropy, on=["model", "seed", "region_id"], validate="one_to_one")
    return result.sort_values(["model", "seed", "region_id"])


def plot(data: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for model, group in data.groupby("model"):
        ax.scatter(group["entropy"], group["mae_db"], s=28, alpha=0.65, label=model)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Mean regional MAE (dB)")
    ax.set_title("Forecast error versus entropy by annotated region")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table = build_table(args)
    table_output = args.table_output or args.output.with_suffix(".csv")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_output, index=False)
    plot(table, args.output)
    print(f"Wrote {len(table)} region/model rows to {table_output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
