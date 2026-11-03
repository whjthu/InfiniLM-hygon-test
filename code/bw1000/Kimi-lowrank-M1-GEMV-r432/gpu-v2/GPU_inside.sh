#!/bin/bash
set -euo pipefail
ulimit -c 0
test "$SLURM_JOB_ID" = 910371
test "$(hostname -s)" = a02r3n03
source /opt/dtk/env.sh
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL INFINILM_OWNER_CPU_WIDTH_R2392
# Exactly one allocated physical GPU, then Runtime logical device0.
case "${SLURM_STEP_GPUS:-0}" in [0-7]) export HIP_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-0}";; *) echo 'Unsupported allocated single GPU identity' >&2; exit 2;; esac
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/old/offline-deps
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export LD_LIBRARY_PATH="/refactor/install/hygon-kimi-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
LEFT=$(python3 -B -c 'import os,time;print(max(0,int(float(os.environ["KIMI_R432_GPU_START"])+570-time.time())))')
test "$LEFT" -gt 30
exec timeout --kill-after=15s "$LEFT" python3 -B -u /refactor/incoming/Kimi-lowrank-M1-GEMV-r432/gpu-v2/GPU_run.py
