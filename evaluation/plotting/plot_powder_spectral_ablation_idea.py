#!/usr/bin/env python3
"""Draw a synthetic schematic of a POWDER spectral ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "plots" / "powder_spectral_ablation_idea.png"


def make_schematic(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic time-frequency power data for an illustrative figure."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 60.0, 181)
    frequencies = np.linspace(600.0, 800.0, 241)
    time_grid, frequency_grid = np.meshgrid(times, frequencies, indexing="xy")

    power = -96.0 + rng.normal(0.0, 1.6, (len(frequencies), len(times)))

    # Broad, time-varying activity bands make the masking idea legible.
    for center, width, amplitude, period in (
        (652.5, 4.5, 17.0, 19.0),
        (705.0, 7.0, 11.0, 26.0),
        (772.5, 3.5, 21.0, 15.0),
    ):
        frequency_profile = np.exp(-0.5 * ((frequency_grid - center) / width) ** 2)
        time_profile = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * time_grid / period))
        power += amplitude * frequency_profile * time_profile

    # A few transient bursts mimic changing occupancy over the context window.
    for center, start, duration, amplitude in (
        (630.0, 12.0, 8.0, 13.0),
        (748.0, 35.0, 10.0, 15.0),
        (785.0, 47.0, 7.0, 12.0),
    ):
        frequency_profile = np.exp(-0.5 * ((frequency_grid - center) / 2.8) ** 2)
        time_profile = np.exp(-0.5 * ((times - (start + duration / 2)) / (duration / 3)) ** 2)
        power += amplitude * frequency_profile * time_profile[np.newaxis, :]

    return times, frequencies, power


def plot_schematic(output: Path, ablation_min: float, ablation_max: float) -> None:
    times, frequencies, power = make_schematic()
    ablated = np.ma.masked_where(
        np.broadcast_to(
            (frequencies[:, np.newaxis] >= ablation_min)
            & (frequencies[:, np.newaxis] <= ablation_max),
            power.shape,
        ),
        power,
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9dce1")
    norm = Normalize(vmin=-100.0, vmax=-72.0)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    extent = [times[0], times[-1], frequencies[0], frequencies[-1]]
    for axis, values, title in (
        (axes[0], power, "Full context"),
        (axes[1], ablated, "Spectral ablation"),
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=12, weight="bold")
        axis.set_xlabel("Context time (min)")
        axis.set_xlim(times[0], times[-1])
        axis.set_ylim(frequencies[0], frequencies[-1])
        axis.axhspan(
            ablation_min,
            ablation_max,
            facecolor="none",
            edgecolor="#f5f5f5",
            hatch="///",
            linewidth=0.0,
            alpha=0.8,
        )

    axes[0].set_ylabel("Frequency (MHz)")
    axes[1].text(
        0.5,
        (ablation_min + ablation_max) / 2,
        "masked band",
        color="#30343b",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.5},
    )
    colorbar = figure.colorbar(image, ax=axes, pad=0.02, shrink=0.9)
    colorbar.set_label("Illustrative power (dB)")
    figure.suptitle(
        "POWDER validation example: full context vs spectral ablation\n"
        "Schematic only: synthetic data, not validation results",
        fontsize=14,
        weight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ablation-min", type=float, default=744.0)
    parser.add_argument("--ablation-max", type=float, default=760.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_schematic(args.output, args.ablation_min, args.ablation_max)
