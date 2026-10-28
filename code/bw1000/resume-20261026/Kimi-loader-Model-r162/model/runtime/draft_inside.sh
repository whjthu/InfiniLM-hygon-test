#!/bin/bash
set -euo pipefail
export KIMI_RUNTIME_INNER_S_R9225=7000
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KIMI_TRUE_DRAFT_BATCH_R9142=1
export PYTHONPYCACHEPREFIX=/local/draft-r9144
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
timeout --kill-after=30s 7000 python -B -u /refactor/incoming/Kimi-sparse-profile-Model-r162/runtime/batch_draft_worker_r9202.py
