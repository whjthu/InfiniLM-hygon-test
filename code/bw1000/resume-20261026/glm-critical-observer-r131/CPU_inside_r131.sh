#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/glm-hybrid-decode-r5386:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/refactor/candidates/glm-grouped-native-model-r5117/helpers:/opt/dtk/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}"
python3 -B /refactor/incoming/GLM-critical-observer-r131/CPU_build_r131.py
