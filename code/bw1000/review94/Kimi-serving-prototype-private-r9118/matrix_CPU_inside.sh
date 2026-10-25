#!/bin/bash
set -euo pipefail
export PYTHONPATH=/old/offline-deps:${PYTHONPATH:-} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
timeout --kill-after=5 150 python -B -u /refactor/incoming/Kimi-serving-matrix-r9118/run_matrix_CPU_r9118.py
