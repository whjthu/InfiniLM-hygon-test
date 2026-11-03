#!/bin/bash
set -euo pipefail
ulimit -c 0
unset BASH_ENV ENV PYTHONSTARTUP
test "$SLURM_JOB_ID" = 910371
test "$(hostname -s)" = a02r3n16
source /opt/dtk/env.sh
# Same explicit one-device visibility as the actual qualified r399 component.
# Whole node must be fresh; actual count1/PCI/CU/memory is checked in each exe.
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL LD_PRELOAD INFINILM_OWNER_CPU_WIDTH_R2392
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/install/hygon-kimi-5f91ad7-r3672/lib:/usr/local/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
exec timeout --kill-after=10s 285 python3 -B -u /refactor/incoming/Kimi-W2-Runtime-H2D-r439/gpu-v2/GPU_run.py
