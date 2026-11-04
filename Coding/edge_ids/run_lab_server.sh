#!/usr/bin/env bash
# Run from the directory containing this file, after transfer and venv setup.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
threads="${EDGE_IDS_THREADS:-8}"
cache_dir="${EDGE_IDS_CACHE_DIR:-$HOME/edge_ids_cache}"
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Create a Python 3.13 venv and install requirements-server.txt first.' >&2
  exit 1
fi
mkdir -p -- "$cache_dir"
df -h -- . "$cache_dir"
free -h
.venv/bin/python verify_server_migration.py
printf 'Starting one CPU defense at a time: %s threads, disk cache %s\n' "$threads" "$cache_dir"
exec .venv/bin/python -u train_adversarial_gb.py --threads "$threads" --matrix external --cache-dir "$cache_dir"
