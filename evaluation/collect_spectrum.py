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


def build_tune_plan(band_start_hz, band_stop_hz, fs, fft_size, cutoff,
                    tune_step_hz=None, overscan_hz=0.0):
    """Compute center frequencies and edge-bin trimming for a 200 MHz band sweep."""
    N = int(np.floor(fft_size * cutoff))
    if N % 2 != 0:
        N -= 1
    n_remove = (fft_size - N) // 2
    effective_bw = N * (fs / fft_size)

    plan_start_hz = band_start_hz - overscan_hz
    plan_stop_hz = band_stop_hz + overscan_hz
    band_width_hz = plan_stop_hz - plan_start_hz

    half_bw = effective_bw / 2.0
    if tune_step_hz is None:
        n_tunes = int(np.ceil(band_width_hz / effective_bw))
        centers = [plan_start_hz + half_bw + i * effective_bw for i in range(n_tunes)]
    else:
        if tune_step_hz <= 0:
            raise ValueError("tune_step_hz must be positive")
        if tune_step_hz > effective_bw:
            raise ValueError("tune_step_hz must be <= retained per-tune bandwidth")

        first_center = plan_start_hz + half_bw
        last_center = plan_stop_hz - half_bw
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


def wait_for_lo_lock(usrp, channel, timeout=0.5, ringdown_s=0.02):
    """Wait for LO lock on hardware that exposes the sensor (e.g. N310).

    Falls back silently on hardware that does not (e.g. B2xx), sleeping a fixed
    amount instead so the call is always safe regardless of board type.
    """
    try:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if usrp.get_rx_sensor("lo_locked", channel).to_bool():
                time.sleep(ringdown_s)
                return True
            time.sleep(0.005)
        print(f"  Warning: LO lock timeout after {timeout*1000:.0f} ms", file=sys.stderr)
        return False
    except RuntimeError:
        # Sensor not available on this board (B2xx); fixed sleep is the only option.
        time.sleep(0.10)
        return True


def capture_samples(rx_streamer, n_samples, timeout=1.0, max_empty=20):
    """Capture n_samples from USRP.

    Returns (samples_1d, timestamp, clean) where clean=True means n_samples
    were received without any overflow or timeout errors.
    """
    n_channels = rx_streamer.get_num_channels()
    if n_channels < 1:
        return None, 0.0, False

    buffer = np.zeros((n_channels, n_samples), dtype=np.complex64)
    metadata = lib.types.rx_metadata()

    stream_cmd = lib.types.stream_cmd(lib.types.stream_mode.num_done)
    stream_cmd.num_samps = n_samples
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)

    samps_recd = 0
    empty_count = 0
    first_ts = 0.0
    had_error = False

    while samps_recd < n_samples:
        chunk = rx_streamer.recv(buffer[:, samps_recd:], metadata, timeout)

        if metadata.error_code == lib.types.rx_metadata_error_code.timeout:
            empty_count += 1
            had_error = True
            if empty_count >= max_empty:
                print(f"  RX timeout/empty too many times ({empty_count}), giving up",
                      file=sys.stderr)
                break
            continue

        if metadata.error_code == lib.types.rx_metadata_error_code.overflow:
            print(f"  RX overflow: {metadata.strerror()}", file=sys.stderr)
            had_error = True
            empty_count += 1
            if empty_count >= max_empty:
                break
            continue

        if metadata.error_code != lib.types.rx_metadata_error_code.none:
            print(f"  RX warning: {metadata.strerror()}", file=sys.stderr)
            had_error = True
            empty_count += 1
            if empty_count >= max_empty:
                break
            continue

        if chunk == 0:
            empty_count += 1
            had_error = True
            if empty_count >= max_empty:
                print(f"  RX: recv returned 0 samples {empty_count} times, giving up",
                      file=sys.stderr)
                break
            continue

        if samps_recd == 0:
            first_ts = metadata.time_spec.get_real_secs()

        samps_recd += chunk
        empty_count = 0

    clean = (not had_error) and (samps_recd == n_samples)
    return buffer[0, :samps_recd], first_ts, clean


def compute_psd(samples, fs, fft_size):
    """Compute Welch PSD, return (freq_offsets_hz, psd_db)."""
    freqs, psd_lin = signal.welch(
        samples, fs, nperseg=fft_size, return_onesided=False
    )
    psd_db = 10.0 * np.log10(np.maximum(psd_lin, 1e-30))
    psd_db = np.fft.fftshift(psd_db)
    freqs = np.fft.fftshift(freqs)
    return freqs, psd_db


def tune_with_offset(usrp, channel, fc, dc_offset_hz):
    """Set RX frequency, shifting the LO by dc_offset_hz so DC leakage lands
    outside the retained band.  The DSP NCO corrects back so the baseband is
    still centred on fc.

    With dc_offset_hz=0 this behaves identically to the original tune_request(fc).
    """
    if dc_offset_hz == 0.0:
        tune_request = lib.types.tune_request(fc)
    else:
        tune_request = lib.types.tune_request(fc)
        tune_request.rf_freq = fc + dc_offset_hz
        tune_request.rf_freq_policy = lib.types.tune_request_policy.manual
        tune_request.dsp_freq = -dc_offset_hz
        tune_request.dsp_freq_policy = lib.types.tune_request_policy.manual
    usrp.set_rx_freq(tune_request, channel)


def sweep_band(usrp, rx_streamer, channel, band_start_mhz, band_stop_mhz, fs,
               fft_size, cutoff, n_samples, raw_dir=None, tune_step_hz=None,
               center_notch_hz=0.0, dc_offset_hz=0.0, max_retries=3,
               overscan_hz=0.0):
    """Tune across one 200 MHz band and return per-tune PSD sweeps.

    Each sweep is (freq_vector_hz, power_vector_db, timestamp). If raw_dir is
    provided, raw .npz files are saved there.

    On overflow or timeout the tune is retried up to max_retries times before
    being marked as failed.  Failed tunes are returned as NaN arrays so that
    aggregate_to_1mhz can write NaN for those bins rather than dropping the row.
    """
    band_start_hz = band_start_mhz * 1e6
    band_stop_hz = band_stop_mhz * 1e6

    centers, n_remove, N = build_tune_plan(
        band_start_hz, band_stop_hz, fs, fft_size, cutoff, tune_step_hz,
        overscan_hz,
    )

    sweeps = []
    for i, fc in enumerate(centers):
        fc_mhz = int(round(fc / 1e6))

        samples = None
        ts = 0.0
        for attempt in range(1, max_retries + 1):
            tune_with_offset(usrp, channel, fc, dc_offset_hz)
            wait_for_lo_lock(usrp, channel)

            raw_samples, raw_ts, clean = capture_samples(rx_streamer, n_samples)

            if raw_samples is not None and len(raw_samples) >= fft_size and clean:
                samples, ts = raw_samples, raw_ts
                break

            if attempt < max_retries:
                print(f"    tune {i+1}/{len(centers)}: fc={fc_mhz} MHz "
                      f"attempt {attempt} failed, retrying ...", file=sys.stderr)
            else:
                print(f"    tune {i+1}/{len(centers)}: fc={fc_mhz} MHz "
                      f"all {max_retries} attempts failed, marking NaN", file=sys.stderr)

        if samples is None or len(samples) < fft_size:
            # All retries exhausted with too few samples — emit a NaN sweep so
            # aggregate_to_1mhz fills those bins with NaN rather than leaving a gap.
            nan_freqs = np.linspace(fc - fs / 2, fc + fs / 2, N)
            nan_psd = np.full(N, np.nan)
            if tune_step_hz is None:
                sweeps.append((nan_freqs, nan_psd, 0.0))
            else:
                sweeps.append((nan_freqs, nan_psd, 0.0, fc))
            continue

        freq_offsets, psd_db = compute_psd(samples, fs, fft_size)
        raw_freqs_hz = freq_offsets + fc
        raw_psd_db = psd_db

        if n_remove > 0:
            psd_db = psd_db[n_remove:-n_remove]
            freq_offsets = freq_offsets[n_remove:-n_remove]

        if center_notch_hz > 0:
            keep = np.abs(freq_offsets) >= center_notch_hz
            psd_db = psd_db[keep]
            freq_offsets = freq_offsets[keep]

        freqs_hz = freq_offsets + fc

        if raw_dir is not None:
            raw_path = raw_dir / f"tune_{i:02d}_fc_{fc_mhz}MHz.npz"
            np.savez_compressed(
                raw_path,
                raw_freq_hz=raw_freqs_hz,
                raw_power_db=raw_psd_db,
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

    NaN bins from failed tunes propagate correctly — np.nansum/np.nanmean
    ignores them, so only bins with zero coverage across all sweeps end up NaN
    in the output.

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

        # Zero-weight NaN bins so they don't contaminate the weighted average
        nan_mask = np.isnan(psd_db)
        weights = np.where(nan_mask, 0.0, weights)

        freqs_mhz = freqs_hz / 1e6
        bin_indices = np.digitize(freqs_mhz, bin_edges) - 1

        valid = (bin_indices >= 0) & (bin_indices < 200)
        if not np.any(valid):
            continue

        binned_power = np.full(200, np.nan)
        binned_weight = np.full(200, np.nan)
        for b in range(200):
            mask = (bin_indices == b) & valid & (~nan_mask)
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

    total_weight = np.nansum(bin_weights, axis=0)
    # Bins with zero total weight (all tunes failed for that frequency) → NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_lin = np.nansum(bin_weighted_powers * bin_weights, axis=0) / total_weight
    mean_lin = np.where(total_weight == 0, np.nan, mean_lin)
    mean_db = np.where(
        np.isnan(mean_lin),
        np.nan,
        10.0 * np.log10(np.maximum(mean_lin, 1e-30)),
    )
    return bin_centers, mean_db


def append_csv_row(csv_path, row_200, minute_idx, bin_centers):
    """Append one 200-element row to the per-minute CSV, writing header first time.

    NaN values are written as empty fields so they are unambiguously missing
    rather than a numeric placeholder.
    """
    def fmt(v):
        return "" if np.isnan(v) else f"{v:.4f}"

    row_str = ",".join(fmt(v) for v in row_200)
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
    parser.add_argument("--site", required=True,
                        help="Site/testbed name (e.g. POWDER, ARA, AERPAW, COSMOS)")
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
    parser.add_argument(
        "--overscan-mhz",
        type=float,
        default=0.0,
        help=(
            "Tune this many MHz past each band edge, then crop aggregation to the requested band. "
            "Use with overlapped tuning for more uniform edge coverage (default 0)."
        ),
    )
    parser.add_argument(
        "--center-notch-mhz",
        type=float,
        default=0.0,
        help="Discard bins within this many MHz of each tuned center frequency (default 0)",
    )
    parser.add_argument(
        "--dc-offset-mhz",
        type=float,
        default=6.0,
        help=(
            "Shift the LO by this many MHz so DC leakage lands outside the retained "
            "band; the DSP NCO corrects back to the requested centre frequency. "
            "Default 6.0 MHz. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--tune-retries",
        type=int,
        default=3,
        help="Number of capture retries on overflow or timeout before marking bins NaN (default 3)",
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
    if args.center_notch_mhz < 0:
        print("ERROR: --center-notch-mhz must be non-negative.", file=sys.stderr)
        sys.exit(1)
    if args.dc_offset_mhz < 0:
        print("ERROR: --dc-offset-mhz must be non-negative.", file=sys.stderr)
        sys.exit(1)
    if args.overscan_mhz < 0:
        print("ERROR: --overscan-mhz must be non-negative.", file=sys.stderr)
        sys.exit(1)
    tune_step_hz = None if args.tune_step_mhz is None else args.tune_step_mhz * 1e6
    overscan_hz = args.overscan_mhz * 1e6
    center_notch_hz = args.center_notch_mhz * 1e6
    dc_offset_hz = args.dc_offset_mhz * 1e6
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
    print(f"DC offset shift:          {args.dc_offset_mhz:.1f} MHz")
    print(f"Overscan per band edge:   {args.overscan_mhz:.3f} MHz")
    print(f"Center notch:             {args.center_notch_mhz:.3f} MHz")
    print(f"Tune retries:             {args.tune_retries}")
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

    # Enable firmware DC offset and IQ imbalance correction when supported.
    # Some radios, including N3xx/AD9371 devices, do not implement these UHD APIs.
    try:
        usrp.set_rx_dc_offset(True, ch)
        rx_dc_offset_correction_enabled = True
    except RuntimeError as exc:
        print(f"  RX DC offset correction unavailable: {exc}", file=sys.stderr)
        rx_dc_offset_correction_enabled = False

    try:
        usrp.set_rx_iq_balance(True, ch)
        rx_iq_balance_correction_enabled = True
    except RuntimeError as exc:
        print(f"  RX IQ balance correction unavailable: {exc}", file=sys.stderr)
        rx_iq_balance_correction_enabled = False

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

    # Warm-up capture: discards transients and lets DC/IQ correction converge.
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
                center_notch_hz, dc_offset_hz, args.tune_retries,
                overscan_hz,
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

            nan_count = int(np.sum(np.isnan(mean_db)))
            if nan_count:
                print(f"    Wrote minute row to {csv_path.name} "
                      f"({nan_count}/200 bins NaN due to failed tunes)")
            else:
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
            "overscan_mhz": args.overscan_mhz,
            "dc_offset_shift_mhz": args.dc_offset_mhz,
            "rx_dc_offset_correction_enabled": rx_dc_offset_correction_enabled,
            "rx_iq_balance_correction_enabled": rx_iq_balance_correction_enabled,
            "center_notch_mhz": args.center_notch_mhz,
            "tune_retries": args.tune_retries,
            "sample_seconds_per_tune": args.sample_seconds,
            "frequency_start_mhz": start_mhz,
            "frequency_stop_mhz": stop_mhz,
            "num_bins": 200,
            "bin_resolution_mhz": 1.0,
            "bin_semantics": "center frequencies of 1 MHz intervals "
                             "(left-inclusive, right-exclusive)",
            "time_resolution": "1 minute",
            "power_unit": "dB (relative, uncalibrated PSD from Welch's method)",
            "nan_semantics": "empty CSV field; all capture retries failed for that bin",
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
