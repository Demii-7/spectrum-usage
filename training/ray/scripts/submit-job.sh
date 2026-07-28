#!/usr/bin/env bash
set -euo pipefail

SUBMIT=false
ADDRESS="${RAY_JOBS_ADDRESS:-http://192.168.1.201:8265}"

if [[ "${1:-}" == --submit ]]; then
  SUBMIT=true
  shift
fi
if [[ "${1:-}" == --address ]]; then
  ADDRESS="${2:?--address requires a URL}"
  shift 2
fi
[[ "${1:-}" == -- ]] && shift
(($#)) || { printf 'Usage: %s [--submit] [--address URL] -- COMMAND [ARG ...]\n' "$0" >&2; exit 2; }

printf 'Jobs API: %s\nCommand:' "$ADDRESS"
printf ' %q' "$@"
printf '\nWorking directory in cluster: /workspace/spectrum-usage\n'
if [[ "$SUBMIT" != true ]]; then
  printf '%s\n' 'DRY RUN: pass --submit to submit through the Ray Jobs API.'
  exit 0
fi

read -r -p "Type SUBMIT-RAY-JOB to continue: " confirmation
[[ "$confirmation" == SUBMIT-RAY-JOB ]] || { printf '%s\n' 'Aborted.' >&2; exit 1; }

payload="$(python3 -c 'import json, shlex, sys; print(json.dumps({"entrypoint": "cd /workspace/spectrum-usage && exec " + shlex.join(sys.argv[1:])}))' "$@")"
curl --fail-with-body --silent --show-error \
  --header 'Content-Type: application/json' \
  --data "$payload" \
  "${ADDRESS%/}/api/jobs/"
printf '\n'
