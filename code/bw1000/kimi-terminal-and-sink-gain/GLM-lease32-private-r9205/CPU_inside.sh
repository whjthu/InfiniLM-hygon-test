#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export LD_LIBRARY_PATH="$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/old/offline-deps:${PYTHONPATH:-}
python -B -u /refactor/incoming/GLM-lease32-r9205/CPU_build.py
