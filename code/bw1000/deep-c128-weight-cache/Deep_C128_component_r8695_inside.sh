#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
test -f "$ROCBLAS_TENSILE_LIBPATH/TensileLibrary_lazy_gfx936.dat"
export PYTHONPYCACHEPREFIX="/local/deep-C128-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export LD_LIBRARY_PATH="/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=8 PYTHONFAULTHANDLER=1
unset INFINILM_DEEP_LAYER_TRACE_R4026
timeout --kill-after=15s 480 python -B -u /refactor/incoming/deep_C128_component_r8695.py
python -B /refactor/incoming/verify_Deep_C128_component_r8695.py "/refactor/results/Deep-C128-component-r8695-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json"
