#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export KIMI_PREFIX_CACHE_R376=1
exec timeout --kill-after=8s 585 python -B -u /refactor/incoming/Kimi-preRoPE-prefix-cache-r376/device-v1/GPU_driver.py --manifest-sha "$KIMI_R376_GPU_MANIFEST" --deadline-unix "$KIMI_R376_GPU_DEADLINE"
