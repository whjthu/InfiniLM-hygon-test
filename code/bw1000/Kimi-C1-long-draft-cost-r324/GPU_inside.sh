#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/local/kimi-draft-cost-r324
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
exec timeout --kill-after=5s 580 python -B -u /refactor/incoming/Kimi-C1-long-draft-cost-r324/GPU_fixture.py
