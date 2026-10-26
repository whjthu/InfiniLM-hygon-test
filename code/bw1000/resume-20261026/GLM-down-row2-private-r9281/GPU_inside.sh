#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export LD_LIBRARY_PATH="$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/old/offline-deps:${PYTHONPATH:-}
timeout --kill-after=5s 830 python -B -u /refactor/incoming/GLM-down-row2-r9281/GPU_run.py
