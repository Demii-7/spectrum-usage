#!/usr/bin/env bash
# Sourced by both Ray entrypoints. Never print secret values.
for name in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY SPECTRUM_AWS_ACCESS_KEY_ID SPECTRUM_AWS_SECRET_ACCESS_KEY; do
  file_var="${name}_FILE"
  if [[ -r "${!file_var:-}" ]]; then
    export "$name=$(<"${!file_var}")"
  fi
done
