#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KIMI_TRUE_DRAFT_BATCH_R9142=1
export PYTHONPYCACHEPREFIX=/local/draft-r9144
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
timeout --kill-after=30s 5200 python -B -u /refactor/incoming/Kimi-modern-PP4-cap32-r9145b/batch_draft_worker_r9128.py
