#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
unset LD_PRELOAD
R=/refactor
COMMON="$R/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM"
SDK="$R/install/hygon-glm-5f91ad7-r3672"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:$R/candidates/glm-grouped-native-model-r5117/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
: "${GLM_M15_CPU_RECEIPT_SHA256_R121B:?completed r121b CPU receipt SHA required}"
timeout --kill-after=10s 180 python3 -B /refactor/incoming/GLM-M15-component-r121b/component_harness_GPU_r121b.py
