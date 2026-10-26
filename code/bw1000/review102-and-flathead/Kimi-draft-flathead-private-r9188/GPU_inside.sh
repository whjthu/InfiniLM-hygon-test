#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/local/dflash2-r9182
export KIMI_TRUE_DRAFT_BATCH_R9142=1
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
timeout --kill-after=5s 830 python -B -u /refactor/incoming/Kimi-draft-flathead-r9188/GPU_fixture.py
