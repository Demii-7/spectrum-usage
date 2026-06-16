#!/bin/bash

set -euo pipefail

COLLECTOR_URL="${SPECTRUM_USAGE_COLLECTOR_URL:-https://raw.githubusercontent.com/Demii-7/spectrum-usage/main/evaluation/collect_spectrum.py}"
COLLECTOR_DIR="${SPECTRUM_USAGE_COLLECTOR_DIR:-/root/spectrum-usage/evaluation}"
COLLECTOR_PATH="$COLLECTOR_DIR/collect_spectrum.py"
RUNNER_PATH="$COLLECTOR_DIR/run_spectrum_usage.sh"

SITE="${SPECTRUM_USAGE_SITE:-AERPAW}"
BANDS="${SPECTRUM_USAGE_BANDS:-3400:3600}"
DURATION_MINUTES="${SPECTRUM_USAGE_DURATION_MINUTES:-60}"
RESULTS_ROOT="${RESULTS_DIR:-/root/Results}"
RUN_STAMP="${SPECTRUM_USAGE_RUN_ID:-${LOG_PREFIX:-$(date -u +%Y%m%dT%H%M%SZ)}}"
OUTPUT_DIR="${SPECTRUM_USAGE_OUTPUT_DIR:-$RESULTS_ROOT/spectrum_usage_${RUN_STAMP}_$$}"
LOG_ROOT="${RESULTS_DIR:-$RESULTS_ROOT}"
LOG_NAME="${LOG_PREFIX:-$RUN_STAMP}"
TIMESTAMP_FORMAT="${TS_FORMAT:-'[%Y-%m-%d %H:%M:%.S]'}"
DEVICE_ARGS="${SPECTRUM_USAGE_DEVICE_ARGS:-}"
RX_CHANNEL="${SPECTRUM_USAGE_RX_CHANNEL:-0}"
ANTENNA="${SPECTRUM_USAGE_ANTENNA:-RX2}"
GAIN="${SPECTRUM_USAGE_GAIN:-30}"
SAMPLE_RATE="${SPECTRUM_USAGE_SAMPLE_RATE:-30.72e6}"
BANDWIDTH="${SPECTRUM_USAGE_BANDWIDTH:-30.72e6}"
FFT_SIZE="${SPECTRUM_USAGE_FFT_SIZE:-4096}"
CUTOFF="${SPECTRUM_USAGE_CUTOFF:-0.836}"
SAMPLE_SECONDS="${SPECTRUM_USAGE_SAMPLE_SECONDS:-0.2}"

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

cat > "$RUNNER_PATH" <<EOF
#!/bin/bash
python3 -u "$COLLECTOR_PATH" \\
  --site "$SITE" \\
  --device-args "$DEVICE_ARGS" \\
  --rx-channel "$RX_CHANNEL" \\
  --antenna "$ANTENNA" \\
  --gain "$GAIN" \\
  --sample-rate "$SAMPLE_RATE" \\
  --bandwidth "$BANDWIDTH" \\
  --fft-size "$FFT_SIZE" \\
  --cutoff "$CUTOFF" \\
  --sample-seconds "$SAMPLE_SECONDS" \\
  --bands "$BANDS" \\
  --duration-minutes "$DURATION_MINUTES" \\
  --output-dir "$OUTPUT_DIR" ${RAW_ARGS[*]}
EOF
chmod +x "$RUNNER_PATH"

screen -S spectrum_usage -dm \
       bash -c "stdbuf -oL -eL '$RUNNER_PATH' \
       2> >(ts $TIMESTAMP_FORMAT >> $LOG_ROOT/${LOG_NAME}_spectrum_usage_log_err.txt) \
       | ts $TIMESTAMP_FORMAT \
       | tee $LOG_ROOT/$LOG_NAME\_spectrum_usage_log.txt"
