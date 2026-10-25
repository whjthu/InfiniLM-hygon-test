#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/old/offline-deps:${PYTHONPATH:-}
timeout --kill-after=5 150 python -B -u /refactor/incoming/Kimi-MoE-lazy-r9125/run_lazy_CPU_r9125.py
