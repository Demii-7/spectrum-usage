#!/usr/bin/env python3
"""Plot 3D map of 50x50 spatial data with frequency as z-axis."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    npz_path = Path("/home/cc/spectrum-usage/evaluation/interpolated_maps/idw_ames_curtiss_600_800.npz")
    if not npz_path.exists():
        print(f"File not found: {npz_path}")
        return

    data = np.load(npz_path)

    map_db = data['map_db']
    freqs_mhz = data['freqs_mhz']
    lon_grid = data['lon_grid']
    lat_grid = data['lat_grid']

    print(f"Data shape: {map_db.shape}")
    print(f"Frequency range: {freqs_mhz[0]:.1f} - {freqs_mhz[-1]:.1f} MHz")

    final_map = map_db[-1]

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    x, y = np.meshgrid(lon_grid, lat_grid)
    z = final_map[:, :, 100]

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    z_flat[np.isnan(z_flat)] = 0

    surf = ax.plot_trisurf(x_flat, y_flat, z_flat, cmap='viridis_r', alpha=0.8, shade=True)

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_zlabel('Power (dB)', fontsize=10)

    center_freq_idx = len(freqs_mhz) // 2
    center_freq = freqs_mhz[center_freq_idx]
    ax.set_title(
        '3D Map: Power Distribution over Longitude and Latitude\n'
        f"Frequency: {center_freq:.1f} MHz (center band)",
        fontsize=12, pad=20
    )

    plt.tight_layout()

    output_path = Path("/home/cc/spectrum-usage/evaluation/interpolated_maps/idw_ames_curtiss_600_800_map_3d.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"Saved 3D plot to: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()