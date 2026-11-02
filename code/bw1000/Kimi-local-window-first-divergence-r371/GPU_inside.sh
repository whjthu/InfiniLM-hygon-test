#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export KIMI_FIRST_LAYER_R371=1
exec timeout --kill-after=8s 585 python -B -u /refactor/incoming/Kimi-local-window-first-divergence-r371/GPU_driver.py --manifest-sha "$KIMI_R371_GPU_MANIFEST" --deadline-unix "$KIMI_R371_GPU_DEADLINE"
