from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from training.common.config import load_config, resolve_path


DEFAULT_INPUTS = (
    Path("training/results/baselines"),
    Path("training/results/LinearAutoRegressive"),
    Path("training/results/ConvLSTM"),
    Path("training/results/STS-PredNet"),
    Path("training/results/TimeRAN"),
)


def read_metric_file(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def combine_metric_files(input_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames = [read_metric_file(path, filename) for path in input_dirs]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def add_skill_scores(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty or "skill_vs_persistence" in aggregate.columns:
        return aggregate
    key_cols = ["chunk_id", "split", "horizon"]
    baseline_names = {
        "persistence": "skill_vs_persistence",
        "historical_mean": "skill_vs_hist_mean",
        "same_time_last3day_mean": "skill_vs_same_time_last3day_mean",
    }
    out = aggregate.copy()
    for baseline, column in baseline_names.items():
        baseline_mae = (
            out[out["model"] == baseline]
            .loc[:, key_cols + ["mae_db"]]
            .rename(columns={"mae_db": f"{baseline}_mae_db"})
        )
        if baseline_mae.empty:
            continue
        out = out.merge(baseline_mae, on=key_cols, how="left")
        out[column] = 1.0 - out["mae_db"] / out[f"{baseline}_mae_db"]
        out = out.drop(columns=[f"{baseline}_mae_db"])
    return out


def write_summary(aggregate: pd.DataFrame, out: Path) -> None:
    lines = ["# Overall Training Results", ""]
    if aggregate.empty:
        lines.append("No aggregate metrics were found.")
        (out / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("Metrics use denormalized dBm predictions.")
    lines.append("")
    lines.append("## Best Model by Chunk and Horizon")
    lines.append("| Chunk | Split | Horizon | Best model | MAE dB | RMSE dB |")
    lines.append("|---|---|---:|---|---:|---:|")
    idx = aggregate.groupby(["chunk_id", "split", "horizon"])["mae_db"].idxmin()
    winners = aggregate.loc[idx].sort_values(["chunk_id", "split", "horizon"])
    for _, row in winners.iterrows():
        lines.append(
            f"| {row['chunk_id']} | {row['split']} | {int(row['horizon'])} | "
            f"{row['model']} | {row['mae_db']:.3f} | {row['rmse_db']:.3f} |"
        )

    if "skill_vs_same_time_last3day_mean" in aggregate.columns:
        lines.append("")
        lines.append("## H=1 Skill vs Same-Time Last-3-Days Mean")
        lines.append("| Chunk | Split | Model | MAE dB | Skill |")
        lines.append("|---|---|---|---:|---:|")
        view = aggregate[aggregate["horizon"] == 1].sort_values(["chunk_id", "split", "mae_db"])
        for _, row in view.iterrows():
            skill = row.get("skill_vs_same_time_last3day_mean")
            skill_text = "" if pd.isna(skill) else f"{skill:.3f}"
            lines.append(f"| {row['chunk_id']} | {row['split']} | {row['model']} | {row['mae_db']:.3f} | {skill_text} |")

    (out / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, action="append", dest="input_dirs")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_config(args.config)
    input_dirs = args.input_dirs or [resolve_path(path) for path in DEFAULT_INPUTS]
    out = args.output_dir or (resolve_path(config["outputs"]["root_dir"]) / "overall")
    out.mkdir(parents=True, exist_ok=True)

    aggregate = add_skill_scores(combine_metric_files(input_dirs, "aggregate_metrics.csv"))
    frequency = combine_metric_files(input_dirs, "per_frequency_metrics.csv")
    band = combine_metric_files(input_dirs, "per_band_metrics.csv")

    aggregate.to_csv(out / "aggregate_metrics.csv", index=False)
    frequency.to_csv(out / "per_frequency_metrics.csv", index=False)
    band.to_csv(out / "per_band_metrics.csv", index=False)
    write_summary(aggregate, out)
    print(f"Wrote overall aggregate metrics to {out / 'aggregate_metrics.csv'}")
    print(f"Wrote overall summary to {out / 'metrics_summary.md'}")


if __name__ == "__main__":
    main()
