#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
TORCH=/usr/local/lib/python3.10/dist-packages/torch/lib
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export LD_LIBRARY_PATH="$SDK/lib:$COMMON:$COMMON/infinilm/_lib:$TORCH:/opt/dtk/cuda/cuda/lib64:/refactor/candidates/glm-hybrid-decode-r5386:/refactor/candidates/glm-grouped-native-model-r5117/helpers:${LD_LIBRARY_PATH:-}"
python3 -B /refactor/incoming/build_glm_crossrequest_Model_r7328.py
