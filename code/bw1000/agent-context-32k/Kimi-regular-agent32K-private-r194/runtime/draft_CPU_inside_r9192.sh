#!/bin/bash
set -euo pipefail
export KIMI_RUNTIME_INNER_S_R9225=7000
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KIMI_TRUE_DRAFT_BATCH_R9142=1
export PYTHONPYCACHEPREFIX=/local/draft-r9144
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
timeout --kill-after=5s 180 python -B -u /refactor/incoming/Kimi-agent-service-Model-r194/runtime/verify_draft_flat_CPU_r9192.py
