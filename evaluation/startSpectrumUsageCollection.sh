#!/bin/bash

set -euo pipefail

COLLECTOR_URL="${SPECTRUM_USAGE_COLLECTOR_URL:-https://raw.githubusercontent.com/Demii-7/spectrum-usage/main/evaluation/collect_spectrum.py}"
COLLECTOR_DIR="${SPECTRUM_USAGE_COLLECTOR_DIR:-/root/spectrum-usage/evaluation}"
COLLECTOR_PATH="$COLLECTOR_DIR/collect_spectrum.py"

SITE="${SPECTRUM_USAGE_SITE:-AERPAW}"
BANDS="${SPECTRUM_USAGE_BANDS:-3400:3600}"
DURATION_MINUTES="${SPECTRUM_USAGE_DURATION_MINUTES:-60}"
RESULTS_ROOT="${RESULTS_DIR:-/root/Results}"
RUN_STAMP="${SPECTRUM_USAGE_RUN_ID:-${LOG_PREFIX:-$(date -u +%Y%m%dT%H%M%SZ)}}"
OUTPUT_DIR="${SPECTRUM_USAGE_OUTPUT_DIR:-$RESULTS_ROOT/spectrum_usage_${RUN_STAMP}_$$}"
DEVICE_ARGS="${SPECTRUM_USAGE_DEVICE_ARGS:-}"
RX_CHANNEL="${SPECTRUM_USAGE_RX_CHANNEL:-0}"
ANTENNA="${SPECTRUM_USAGE_ANTENNA:-RX2}"
GAIN="${SPECTRUM_USAGE_GAIN:-30}"
SAMPLE_RATE="${SPECTRUM_USAGE_SAMPLE_RATE:-30.72e6}"
BANDWIDTH="${SPECTRUM_USAGE_BANDWIDTH:-30.72e6}"
FFT_SIZE="${SPECTRUM_USAGE_FFT_SIZE:-4096}"
CUTOFF="${SPECTRUM_USAGE_CUTOFF:-0.836}"
SAMPLE_SECONDS="${SPECTRUM_USAGE_SAMPLE_SECONDS:-0.2}"

if [ "${LAUNCH_MODE:-none}" = "EMULATION" ] && [ "${SPECTRUM_USAGE_ALLOW_EMULATION:-0}" != "1" ]; then
  echo "Skipping spectrum usage collection in EMULATION mode. Set SPECTRUM_USAGE_ALLOW_EMULATION=1 to override."
  exit 0
fi

mkdir -p "$COLLECTOR_DIR" "$OUTPUT_DIR"

echo "Downloading spectrum collector from $COLLECTOR_URL"
wget -q -O "$COLLECTOR_PATH" "$COLLECTOR_URL"
chmod +x "$COLLECTOR_PATH"

RAW_ARGS=()
if [ "${SPECTRUM_USAGE_SAVE_RAW:-0}" = "1" ]; then
  RAW_ARGS+=(--save-raw)
fi

if [ "${SPECTRUM_USAGE_OVERWRITE:-0}" = "1" ]; then
  RAW_ARGS+=(--overwrite)
fi

echo "Starting spectrum usage collection"
echo "  site=$SITE"
echo "  bands=$BANDS"
echo "  output=$OUTPUT_DIR"
echo "  launch_mode=${LAUNCH_MODE:-unknown}"

python3 -u "$COLLECTOR_PATH" \
  --site "$SITE" \
  --device-args "$DEVICE_ARGS" \
  --rx-channel "$RX_CHANNEL" \
  --antenna "$ANTENNA" \
  --gain "$GAIN" \
  --sample-rate "$SAMPLE_RATE" \
  --bandwidth "$BANDWIDTH" \
  --fft-size "$FFT_SIZE" \
  --cutoff "$CUTOFF" \
  --sample-seconds "$SAMPLE_SECONDS" \
  --bands "$BANDS" \
  --duration-minutes "$DURATION_MINUTES" \
  --output-dir "$OUTPUT_DIR" \
  "${RAW_ARGS[@]}"
