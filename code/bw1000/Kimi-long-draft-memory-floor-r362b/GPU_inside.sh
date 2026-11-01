#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/local/kimi-draft-memory-r362
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export KIMI_MEMORY_PRESSURE_R362=1
exec timeout --kill-after=8s 585 python -B -u /refactor/incoming/Kimi-long-draft-memory-floor-r362b/pressure.py --manifest-sha "$KIMI_R362_MANIFEST" --run "$KIMI_R362_RUN" --deadline-unix "$KIMI_R362_DEADLINE"
