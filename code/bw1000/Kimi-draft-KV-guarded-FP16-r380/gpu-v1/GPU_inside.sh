#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
test "${SLURM_LOCALID}" = 0 -o "${SLURM_LOCALID}" = 1
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HIP_VISIBLE_DEVICES="${SLURM_LOCALID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
exec timeout --kill-after=8s 585 python -B -u /refactor/incoming/Kimi-draft-KV-guarded-FP16-r380/gpu-v1/GPU_driver.py
