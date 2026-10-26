#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
unset LD_PRELOAD
R=/refactor
COMMON="$R/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM"
SDK="$R/install/hygon-glm-5f91ad7-r3672"
export LD_LIBRARY_PATH="$R/candidates/glm-grouped-native-model-r5117/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
timeout --kill-after=10s 480 python3 -B /refactor/incoming/GLM-M15-component-r121/CPU_build_r121.py
