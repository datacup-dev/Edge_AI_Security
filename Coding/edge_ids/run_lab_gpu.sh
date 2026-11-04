#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EDGE_IDS_DEVICE=cuda
export EDGE_IDS_THREADS="${EDGE_IDS_THREADS:-8}"
export EDGE_IDS_MATRIX="${EDGE_IDS_MATRIX:-memory}"
.venv/bin/python verify_server_migration.py
.venv/bin/python gpu_resume_smoke.py
exec .venv/bin/python -u train_adversarial_gb.py --device cuda --threads "$EDGE_IDS_THREADS" --matrix "$EDGE_IDS_MATRIX"
