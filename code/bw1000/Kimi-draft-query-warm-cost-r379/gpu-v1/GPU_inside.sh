#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export KIMI_WARM_COST_R379=1
exec timeout --kill-after=8s 585 python -B -u /refactor/incoming/Kimi-draft-query-warm-cost-r379/gpu-v1/GPU_driver.py --manifest-sha "$KIMI_R379_GPU_MANIFEST" --deadline-unix "$KIMI_R379_GPU_DEADLINE"
