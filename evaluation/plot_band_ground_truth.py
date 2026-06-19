#!/usr/bin/env python3
"""Plot a per-band 1 MHz spectrum heatmap from collector CSV output."""

import argparse
import json
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_like_tss_lcd(data, window_length=51, polyorder=2):
    n_time = data.shape[0]
    window = min(window_length, n_time if n_time % 2 else n_time - 1)
    if window <= polyorder:
        return data

    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    scale = col_max - col_min
    scale[scale == 0] = 1e-8

    norm = (data - col_min) / scale
    smooth = savgol_filter(
        norm,
        window_length=window,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )
    return smooth * scale + col_min


def resolve_paths(input_path):
    if input_path.is_dir():
        csv_path = input_path / "power_1mhz_avg_per_minute.csv"
        metadata_path = input_path / "metadata.json"
    else:
        csv_path = input_path
        metadata_path = input_path.with_name("metadata.json")
    return csv_path, metadata_path


def discover_inputs(input_path):
    if input_path.is_file() and tarfile.is_tarfile(input_path):
        with tarfile.open(input_path) as archive:
            members = [
                member.name
                for member in archive.getmembers()
                if member.name.endswith("power_1mhz_avg_per_minute.csv")
            ]
        return [("tar", input_path, name) for name in sorted(members, key=band_sort_key)]

    if input_path.is_dir() and not (input_path / "power_1mhz_avg_per_minute.csv").exists():
        return [
            ("file", path, None)
            for path in sorted(
                input_path.rglob("power_1mhz_avg_per_minute.csv"),
                key=band_sort_key,
            )
        ]

    csv_path, _ = resolve_paths(input_path)
    return [("file", csv_path, None)]


def band_sort_key(path):
    name = path.name if isinstance(path, Path) else path
    band = Path(name).parent.name
    try:
        return (0, int(band.split("_", 1)[0]), band)
    except ValueError:
        return (1, 0, band)


def read_inputs(input_ref):
    kind, input_path, member_name = input_ref
    if kind == "tar":
        metadata = {}
        with tarfile.open(input_path) as archive:
            with archive.extractfile(member_name) as csv_file:
                df = pd.read_csv(csv_file)

            metadata_name = str(Path(member_name).with_name("metadata.json"))
            try:
                metadata_member = archive.getmember(metadata_name)
            except KeyError:
                metadata_member = None
            if metadata_member is not None:
                with archive.extractfile(metadata_member) as metadata_file:
                    metadata = json.load(metadata_file)

        freqs_mhz = df.columns.to_numpy(dtype=float)
        power = df.to_numpy(dtype=float)
        return Path(member_name), freqs_mhz, power, metadata

    input_path = Path(input_path)
    csv_path, metadata_path = resolve_paths(input_path)
    df = pd.read_csv(csv_path)
    freqs_mhz = df.columns.to_numpy(dtype=float)
    power = df.to_numpy(dtype=float)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return csv_path, freqs_mhz, power, metadata


def title_for(csv_path, metadata):
    site = metadata.get("site")
    start_mhz = metadata.get("frequency_start_mhz")
    stop_mhz = metadata.get("frequency_stop_mhz")

    parts = []
    if site:
        parts.append(site)
    if start_mhz is not None and stop_mhz is not None:
        parts.append(f"{start_mhz}-{stop_mhz} MHz")
    elif csv_path.parent.name:
        parts.append(csv_path.parent.name.replace("_", "-"))
    parts.append("Ground-Truth")
    return "\n".join([" ".join(parts[:-1]).strip(), parts[-1]]) if len(parts) > 1 else parts[0]


def output_path_for(input_ref, output_path, multiple):
    kind, input_path, member_name = input_ref
    if output_path is not None and not multiple:
        return output_path

    if kind == "tar":
        band = Path(member_name).parent.name
        stem = Path(input_path).stem
        default = Path(input_path).with_name(f"{stem}_{band}_ground_truth.png")
    else:
        csv_path, _ = resolve_paths(Path(input_path))
        default = csv_path.with_suffix(".png")

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / default.name
    return default


def plot(input_ref, output_path, time_steps, from_end, smooth):
    csv_path, freqs_mhz, power, metadata = read_inputs(input_ref)
    if power.shape[1] != 200:
        raise ValueError(f"expected 200 columns, found {power.shape[1]}")
    if smooth:
        power = smooth_like_tss_lcd(power)
    if time_steps is not None:
        power = power[-time_steps:] if from_end else power[:time_steps]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    finite = power[np.isfinite(power)]
    vmin, vmax = np.percentile(finite, [1, 99])

    fig, ax = plt.subplots(figsize=(5.2, 1.8))
    im = ax.imshow(
        power.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[0, power.shape[0], freqs_mhz[0] - 0.5, freqs_mhz[-1] + 0.5],
    )

    start_tick = int(np.ceil((freqs_mhz[0] - 0.5) / 25.0) * 25)
    stop_tick = int(np.floor((freqs_mhz[-1] + 0.5) / 25.0) * 25)
    yticks = np.arange(start_tick, stop_tick + 1, 25)

    ax.set_title(title_for(csv_path, metadata), fontsize=7.5, fontweight="bold", pad=2)
    ax.set_xlabel("Time Step", fontsize=7.5, fontweight="bold", labelpad=1)
    ax.set_ylabel("Frequency (MHz)", fontsize=7.5, fontweight="bold", labelpad=1)
    ax.set_xlim(0, power.shape[0])
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=6.5, top=True, right=True, length=3, width=0.9, pad=1)

    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    colorbar.ax.tick_params(labelsize=6.5, direction="in", length=2, width=0.8, pad=1)
    colorbar.set_label("Power (dB)", fontsize=7)

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.28, top=0.8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Band directory, run directory, tarball, or power_1mhz_avg_per_minute.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path for one band, or output directory for multi-band inputs.",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=0,
        help="Number of time steps to plot. Use 0 to plot the full CSV.",
    )
    parser.add_argument(
        "--from-end",
        action="store_true",
        help="Plot the last --max-time rows instead of the first rows.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Plot raw values instead of applying TSS-LCD-style smoothing.",
    )
    args = parser.parse_args()

    input_refs = discover_inputs(args.input)
    if not input_refs:
        raise ValueError(f"No power_1mhz_avg_per_minute.csv files found in {args.input}")

    time_steps = None if args.max_time == 0 else args.max_time
    multiple = len(input_refs) > 1
    for input_ref in input_refs:
        output_path = output_path_for(input_ref, args.output, multiple)
        plot(input_ref, output_path, time_steps, args.from_end, not args.no_smooth)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
