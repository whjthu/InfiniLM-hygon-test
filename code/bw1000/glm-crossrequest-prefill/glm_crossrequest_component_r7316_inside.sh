#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
TORCH=/usr/local/lib/python3.10/dist-packages/torch/lib
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export LD_LIBRARY_PATH="$SDK/lib:$COMMON:$COMMON/infinilm/_lib:$TORCH:/opt/dtk/cuda/cuda/lib64:/refactor/candidates/glm-hybrid-decode-r5386:/refactor/candidates/glm-grouped-native-model-r5117/helpers:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
timeout --kill-after=15s 510 python3 -B -u /refactor/incoming/glm_crossrequest_component_r7316.py
