#!/usr/bin/env python3
"""
Spectrum data acquisition script for POWDER, ARA, AERPAW, and COSMOS testbeds.

Collects wideband PSD measurements using a UHD-supported USRP and saves
per-minute 200-column CSV files with 1 MHz resolution across one or more
200 MHz bands. Optional per-sweep raw PSD files can also be saved.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal

_HAS_UHD = False
try:
    import uhd
    from uhd import libpyuhd as lib
    _HAS_UHD = True
except ImportError:
    pass


def parse_band_spec(spec_str):
    """Parse comma-separated band specs like '3400:3600,3600:3800'."""
    bands = []
    for part in spec_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            start_str, stop_str = part.split(":")
            start_mhz = float(start_str)
            stop_mhz = float(stop_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid band spec '{part}'. Use start_mhz:stop_mhz"
            )
        width_mhz = stop_mhz - start_mhz
        if abs(width_mhz - 200.0) > 1e-3:
            raise argparse.ArgumentTypeError(
                f"Band '{part}' is {width_mhz:.1f} MHz wide; must be exactly 200 MHz"
            )
        if start_mhz < 10 or stop_mhz > 7000:
            print(f"  Warning: band {start_mhz:.0f}-{stop_mhz:.0f} MHz may exceed "
                  f"SDR frequency range", file=sys.stderr)
        bands.append((int(start_mhz), int(stop_mhz)))
    return bands


def build_tune_plan(band_start_hz, band_stop_hz, fs, fft_size, cutoff, tune_step_hz=None):
    """Compute center frequencies and edge-bin trimming for a 200 MHz band sweep."""
    N = int(np.floor(fft_size * cutoff))
    if N % 2 != 0:
        N -= 1
    n_remove = (fft_size - N) // 2
    effective_bw = N * (fs / fft_size)

    band_width_hz = band_stop_hz - band_start_hz

    half_bw = effective_bw / 2.0
    if tune_step_hz is None:
        n_tunes = int(np.ceil(band_width_hz / effective_bw))
        centers = [band_start_hz + half_bw + i * effective_bw for i in range(n_tunes)]
    else:
        if tune_step_hz <= 0:
            raise ValueError("tune_step_hz must be positive")
        if tune_step_hz > effective_bw:
            raise ValueError("tune_step_hz must be <= retained per-tune bandwidth")

        first_center = band_start_hz + half_bw
        last_center = band_stop_hz - half_bw
        centers = list(np.arange(first_center, last_center + tune_step_hz * 0.5, tune_step_hz))
        if not centers or centers[-1] < last_center:
            centers.append(last_center)
        centers[-1] = min(centers[-1], last_center)
    return centers, n_remove, N


def create_rx_streamer(usrp, channel):
    """Create a single RX streamer to reuse across captures."""
    st_args = lib.usrp.stream_args("fc32", "sc16")
    st_args.channels = [channel]
    return usrp.get_rx_stream(st_args)


def capture_samples(rx_streamer, n_samples, timeout=1.0, max_empty=20):
    """Capture n_samples from USRP, return (samples_1d, timestamp)."""
    n_channels = rx_streamer.get_num_channels()
    if n_channels < 1:
        return None, 0.0

    buffer = np.zeros((n_channels, n_samples), dtype=np.complex64)
    metadata = lib.types.rx_metadata()

    stream_cmd = lib.types.stream_cmd(lib.types.stream_mode.num_done)
    stream_cmd.num_samps = n_samples
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)

    samps_recd = 0
    empty_count = 0
    first_ts = 0.0

    while samps_recd < n_samples:
        chunk = rx_streamer.recv(buffer[:, samps_recd:], metadata, timeout)

        if metadata.error_code == lib.types.rx_metadata_error_code.timeout:
            empty_count += 1
            if empty_count >= max_empty:
                print(f" RX timeout/empty too many times ({empty_count}), giving up", file=sys.stderr)
                break
            continue

        if metadata.error_code == lib.types.rx_metadata_error_code.overflow:
            print(f" RX overflow: {metadata.strerror()}", file=sys.stderr)
            empty_count += 1
            if empty_count >= max_empty:
                break
            continue

        if metadata.error_code != lib.types.rx_metadata_error_code.none:
            print(f" RX warning: {metadata.strerror()}", file=sys.stderr)
            empty_count += 1
            if empty_count >= max_empty:
                break
            continue

        if chunk == 0:
            empty_count += 1
            if empty_count >= max_empty:
                print(f" RX: recv returned 0 samples {empty_count} times, giving up", file=sys.stderr)
                break
            continue

        if samps_recd == 0:
            first_ts = metadata.time_spec.get_real_secs()

        samps_recd += chunk
        empty_count = 0

    return buffer[0, :samps_recd], first_ts

def compute_psd(samples, fs, fft_size):
    """Compute Welch PSD, return (freq_offsets_hz, psd_db)."""
    freqs, psd_lin = signal.welch(
        samples, fs, nperseg=fft_size, return_onesided=False
    )
    psd_db = 10.0 * np.log10(np.maximum(psd_lin, 1e-30))
    psd_db = np.fft.fftshift(psd_db)
    freqs = np.fft.fftshift(freqs)
    return freqs, psd_db


def sweep_band(usrp, rx_streamer, channel, band_start_mhz, band_stop_mhz, fs,
               fft_size, cutoff, n_samples, raw_dir=None, tune_step_hz=None):
    """Tune across one 200 MHz band and return per-tune PSD sweeps.

    Each sweep is (freq_vector_hz, power_vector_db, timestamp). If raw_dir is
    provided, raw .npz files are saved there.
    """
    band_start_hz = band_start_mhz * 1e6
    band_stop_hz = band_stop_mhz * 1e6

    centers, n_remove, N = build_tune_plan(
        band_start_hz, band_stop_hz, fs, fft_size, cutoff, tune_step_hz
    )

    sweeps = []
    for i, fc in enumerate(centers):
        tune_request = lib.types.tune_request(fc)
        usrp.set_rx_freq(tune_request, channel)
        time.sleep(0.10)
        samples, ts = capture_samples(rx_streamer, n_samples)
        if samples is None or len(samples) < fft_size:
            print(f"    Not enough samples, skipping tune {i}", file=sys.stderr)
            continue

        freq_offsets, psd_db = compute_psd(samples, fs, fft_size)

        if n_remove > 0:
            psd_db = psd_db[n_remove:-n_remove]
            freq_offsets = freq_offsets[n_remove:-n_remove]

        freqs_hz = freq_offsets + fc

        fc_mhz = int(round(fc / 1e6))
        if raw_dir is not None:
            raw_path = raw_dir / f"tune_{i:02d}_fc_{fc_mhz}MHz.npz"
            np.savez_compressed(
                raw_path,
                freq_hz=freqs_hz,
                power_db=psd_db,
                timestamp=ts,
                center_freq_hz=fc,
            )
            print(f"    tune {i+1}/{len(centers)}: fc={fc_mhz} MHz "
                  f"-> {raw_path.name} ({len(freqs_hz)} bins)")
        else:
            print(f"    tune {i+1}/{len(centers)}: fc={fc_mhz} MHz "
                  f"({len(freqs_hz)} bins)")

        if tune_step_hz is None:
            sweeps.append((freqs_hz, psd_db, ts))
        else:
            sweeps.append((freqs_hz, psd_db, ts, fc))

    return sweeps


def aggregate_to_1mhz(sweeps, band_start_mhz, band_stop_mhz):
    """Aggregate all sweeps in a band into 200 × 1 MHz bin values.

    Returns (freq_centers_mhz, power_db_200) or (None, None) if no valid data.
    """
    bin_edges = np.arange(band_start_mhz, band_stop_mhz + 1)
    bin_centers = bin_edges[:-1] + 0.5

    bin_weighted_powers = []
    bin_weights = []
    for sweep in sweeps:
        if len(sweep) == 4:
            freqs_hz, psd_db, ts, center_freq_hz = sweep
            max_offset = np.nanmax(np.abs(freqs_hz - center_freq_hz))
            weights = 1.0 - np.abs(freqs_hz - center_freq_hz) / max(max_offset, 1.0)
            weights = np.clip(weights, 0.05, 1.0)
        else:
            freqs_hz, psd_db, ts = sweep
            weights = np.ones_like(psd_db)

        freqs_mhz = freqs_hz / 1e6
        bin_indices = np.digitize(freqs_mhz, bin_edges) - 1

        valid = (bin_indices >= 0) & (bin_indices < 200)
        if not np.any(valid):
            continue

        binned_power = np.full(200, np.nan)
        binned_weight = np.full(200, np.nan)
        for b in range(200):
            mask = bin_indices == b
            if np.any(mask):
                lin = 10.0 ** (psd_db[mask] / 10.0)
                w = weights[mask]
                binned_power[b] = np.average(lin, weights=w)
                binned_weight[b] = np.mean(w)

        bin_weighted_powers.append(binned_power)
        bin_weights.append(binned_weight)

    if not bin_weighted_powers:
        return None, None

    bin_weighted_powers = np.array(bin_weighted_powers)
    bin_weights = np.array(bin_weights)
    mean_lin = np.nansum(bin_weighted_powers * bin_weights, axis=0) / np.nansum(bin_weights, axis=0)
    mean_db = 10.0 * np.log10(np.maximum(mean_lin, 1e-30))
    return bin_centers, mean_db


def append_csv_row(csv_path, row_200, minute_idx, bin_centers):
    """Append one 200-element row to the per-minute CSV, writing header first time."""
    row_str = ",".join(f"{v:.4f}" for v in row_200)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if write_header:
        header = ",".join(f"{bc:.1f}" for bc in bin_centers)
        with open(csv_path, "w") as f:
            f.write(header + "\n")
    with open(csv_path, "a") as f:
        f.write(row_str + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Acquire spectrum data from USRP for POWDER/ARA/AERPAW/COSMOS evaluation"
    )
    parser.add_argument("--site", required=True, help="Site/testbed name (e.g. POWDER, ARA, AERPAW, COSMOS)")
    parser.add_argument(
        "--sdr", default=None,
        help="SDR model string for metadata (auto-detected from USRP if omitted)"
    )
    parser.add_argument(
        "--device-args", default="",
        help="UHD device arguments (e.g. 'mgmt_addr=10.37.2.1,addr=10.39.2.1' for N310)"
    )
    parser.add_argument("--rx-channel", type=int, default=0, help="RX channel index (default 0)")
    parser.add_argument("--antenna", default="RX2", help="RX antenna port (default RX2)")
    parser.add_argument("--gain", type=float, default=35.0, help="RX gain in dB (default 35)")
    parser.add_argument("--sample-rate", type=float, default=30.72e6,
                        help="Sample rate in Hz (default 30.72e6)")
    parser.add_argument("--bandwidth", type=float, default=None,
                        help="RX bandwidth in Hz (default = sample-rate)")
    parser.add_argument("--fft-size", type=int, default=4096,
                        help="FFT size for Welch PSD (default 4096)")
    parser.add_argument("--cutoff", type=float, default=0.836,
                        help="Fraction of FFT bins to retain per tune (default 0.836)")
    parser.add_argument(
        "--tune-step-mhz",
        type=float,
        default=None,
        help=(
            "Spacing between adjacent sweep center frequencies in MHz. "
            "Set below sample-rate*cutoff to overlap tunes; default keeps existing non-overlap behavior."
        ),
    )
    parser.add_argument("--sample-seconds", type=float, default=0.2,
                        help="Seconds of IQ data to capture per tune (default 0.2)")
    parser.add_argument(
        "--bands", required=True,
        help="Comma-separated 200 MHz bands, e.g. '3400:3600' or '3400:3600,3600:3800'"
    )
    parser.add_argument("--duration-minutes", type=float, required=True,
                        help="Total data collection duration in minutes")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for all data files")
    parser.add_argument("--save-raw", action="store_true",
                        help="Save raw per-tune PSD .npz files. Disabled by default.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing output files in --output-dir")
    args = parser.parse_args()

    if not _HAS_UHD:
        print("ERROR: uhd Python module not found.", file=sys.stderr)
        print("Install UHD with Python bindings (e.g. 'apt install uhd-host' or compile from source).",
              file=sys.stderr)
        sys.exit(1)

    bands = parse_band_spec(args.bands)
    fs = args.sample_rate
    fft_size = args.fft_size
    cutoff = args.cutoff
    tune_step_hz = None if args.tune_step_mhz is None else args.tune_step_mhz * 1e6
    bw = args.bandwidth if args.bandwidth is not None else fs
    n_samples = max(fft_size * 2, int(fs * args.sample_seconds))
    duration_sec = args.duration_minutes * 60.0
    output_root = Path(args.output_dir)

    existing_outputs = []
    for start_mhz, stop_mhz in bands:
        band_dir = output_root / f"{start_mhz}_{stop_mhz}"
        csv_path = band_dir / "power_1mhz_avg_per_minute.csv"
        meta_path = band_dir / "metadata.json"
        raw_dir = band_dir / "raw"
        if csv_path.exists():
            existing_outputs.append(str(csv_path))
        if meta_path.exists():
            existing_outputs.append(str(meta_path))
        if args.save_raw and raw_dir.exists() and any(raw_dir.iterdir()):
            existing_outputs.append(str(raw_dir))
    if existing_outputs and not args.overwrite:
        print("ERROR: output files already exist and --overwrite was not set.", file=sys.stderr)
        print("Use a new --output-dir or rerun with --overwrite.", file=sys.stderr)
        for path in existing_outputs:
            print(f"  existing: {path}", file=sys.stderr)
        sys.exit(2)

    print(f"Site:                     {args.site}")
    print(f"SDR:                      {args.sdr or 'auto-detected'}")
    print(f"Device args:              '{args.device_args}'")
    print(f"Bands:                    {bands}")
    print(f"Duration:                 {args.duration_minutes} min")
    print(f"Output root:              {output_root}")
    print(f"Sample rate:              {fs/1e6:.2f} MHz")
    print(f"FFT size:                 {fft_size}")
    print(f"Cutoff:                   {cutoff}")
    if tune_step_hz is None:
        print("Tune step:                non-overlap default")
    else:
        retained_mhz = fs * cutoff / 1e6
        overlap_pct = max(0.0, 100.0 * (1.0 - args.tune_step_mhz / retained_mhz))
        print(f"Tune step:                {args.tune_step_mhz:.3f} MHz ({overlap_pct:.1f}% overlap)")
    print(f"Samples per tune:         {n_samples} ({args.sample_seconds} s)")
    print(f"Gain:                     {args.gain} dB")
    print(f"Antenna:                  {args.antenna}")
    print(f"Save raw PSD files:       {args.save_raw}")
    print(f"Overwrite existing files: {args.overwrite}")
    print()

    # Create output directory structure
    for start_mhz, stop_mhz in bands:
        band_dir = output_root / f"{start_mhz}_{stop_mhz}"
        band_dir.mkdir(parents=True, exist_ok=True)
        if args.save_raw:
            (band_dir / "raw").mkdir(exist_ok=True)

    # --- Connect USRP ---
    print("Initializing USRP ...")
    usrp = uhd.usrp.MultiUSRP(args.device_args)
    mboard = usrp.get_mboard_name()
    print(f"  Board: {mboard}")

    ch = args.rx_channel
    usrp.set_rx_rate(fs, ch)
    usrp.set_rx_freq(lib.types.tune_request(3400e6), ch)
    usrp.set_rx_gain(args.gain, ch)
    usrp.set_rx_antenna(args.antenna, ch)
    usrp.set_rx_bandwidth(bw, ch)

    actual_rate = usrp.get_rx_rate(ch)
    fs = actual_rate
    n_samples = max(fft_size * 2, int(fs * args.sample_seconds))
    actual_gain = usrp.get_rx_gain(ch)
    actual_antenna = usrp.get_rx_antenna(ch)
    actual_bw = usrp.get_rx_bandwidth(ch)
    print(f"  Actual sample rate:  {actual_rate/1e6:.2f} MHz")
    print(f"  Actual gain:         {actual_gain:.1f} dB")
    print(f"  Actual antenna:      {actual_antenna}")
    print(f"  Actual bandwidth:    {actual_bw/1e6:.2f} MHz")
    print()

    # Create streamer
    rx_streamer = create_rx_streamer(usrp, ch)

    # Warm-up capture (discard first result)
    print("Warming up USRP stream ...")
    _ = capture_samples(rx_streamer, n_samples)
    print()

    # --- Per-minute collection loop ---
    start_wall = time.time()
    minute_count = int(np.ceil(args.duration_minutes))
    print(f"Starting collection for {args.duration_minutes} min "
          f"({minute_count} minute(s))")

    for minute_idx in range(minute_count):
        target_time = start_wall + minute_idx * 60.0

        # Wait until this minute's target time
        remaining = target_time - time.time()
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -30.0:
            print(f"\nMinute {minute_idx + 1}: skipping (too late)")
            continue

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n--- Minute {minute_idx + 1}/{minute_count} [{now_utc}] ---")

        for start_mhz, stop_mhz in bands:
            band_label = f"{start_mhz}_{stop_mhz}"
            band_dir = output_root / band_label
            raw_dir = None
            if args.save_raw:
                safe_ts = now_utc.replace(":", "").replace("-", "")
                raw_dir = band_dir / "raw" / f"minute_{minute_idx:04d}_{safe_ts}"
                raw_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Band {start_mhz}-{stop_mhz} MHz")
            sys.stdout.flush()

            sweeps = sweep_band(
                usrp, rx_streamer, ch, start_mhz, stop_mhz,
                fs, fft_size, cutoff, n_samples, raw_dir, tune_step_hz,
            )

            if not sweeps:
                print(f"    No sweeps collected for this band", file=sys.stderr)
                continue

            bin_centers, mean_db = aggregate_to_1mhz(sweeps, start_mhz, stop_mhz)
            if mean_db is None:
                print(f"    Could not aggregate data", file=sys.stderr)
                continue

            csv_path = band_dir / "power_1mhz_avg_per_minute.csv"
            append_csv_row(csv_path, mean_db, minute_idx, bin_centers)
            print(f"    Wrote minute row to {csv_path.name}")

        # Check if we've exceeded total duration
        elapsed = time.time() - start_wall
        if elapsed >= duration_sec:
            print(f"\nReached {args.duration_minutes} min, stopping")
            break

    # --- Write per-band metadata ---
    print("\n--- Writing metadata ---")
    for start_mhz, stop_mhz in bands:
        band_dir = output_root / f"{start_mhz}_{stop_mhz}"
        metadata = {
            "site": args.site,
            "sdr": args.sdr or mboard,
            "device_args": args.device_args,
            "rx_channel": args.rx_channel,
            "antenna": args.antenna,
            "gain_db": args.gain,
            "sample_rate_sps": fs,
            "bandwidth_hz": bw,
            "fft_size": fft_size,
            "cutoff_factor": cutoff,
            "tune_step_mhz": args.tune_step_mhz,
            "overlapped_tuning_enabled": args.tune_step_mhz is not None,
            "sample_seconds_per_tune": args.sample_seconds,
            "frequency_start_mhz": start_mhz,
            "frequency_stop_mhz": stop_mhz,
            "num_bins": 200,
            "bin_resolution_mhz": 1.0,
            "bin_semantics": "center frequencies of 1 MHz intervals "
                             "(left-inclusive, right-exclusive)",
            "time_resolution": "1 minute",
            "power_unit": "dB (relative, uncalibrated PSD from Welch's method)",
            "collection_duration_minutes": args.duration_minutes,
            "save_raw": args.save_raw,
            "overwrite_enabled": args.overwrite,
        }
        meta_path = band_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
