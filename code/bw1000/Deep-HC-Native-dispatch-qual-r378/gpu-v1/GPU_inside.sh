#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export INFINILM_OWNER_CPU_WIDTH_R2392=4
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:/refactor/install/hygon-deep-5f91ad7-r3672/lib:/refactor/candidates/latest-family-native-r269/Deep/python:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec timeout --kill-after=10s 150 python -B -u /refactor/incoming/Deep-HC-Native-dispatch-qual-r378/gpu-v1/GPU_driver.py
