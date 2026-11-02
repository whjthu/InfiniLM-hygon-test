#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:/refactor/install/hygon-deep-5f91ad7-r3672/lib:/refactor/candidates/latest-family-native-r269/Deep/python:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}
exec timeout 270 python3 -B /refactor/incoming/Deep-PD-disaggregation-r382/native/v5/CPU_build.py
