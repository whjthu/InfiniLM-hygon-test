#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
TORCH=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:$TORCH:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset INFINILM_DEEP_LAYER_TRACE_R4026 DEEP_CSA_CAPTURE_R5800
timeout --kill-after=15s 260 python -B -u /refactor/incoming/Deep_routed_budget_r7342.py
