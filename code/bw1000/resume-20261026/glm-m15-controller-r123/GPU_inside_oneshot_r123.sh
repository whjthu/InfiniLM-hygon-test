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
: "${GLM_M15_CPU_RECEIPT_SHA256_R121B:?completed CPU receipt SHA required}"
: "${GLM_M15_PINNED_SOURCES_R123:?pinned source directory required}"
test "$GLM_M15_PINNED_SOURCES_R123" = /refactor/results/GLM-M15-once-r123-sources
timeout --kill-after=10s 1100 python3 -B "$GLM_M15_PINNED_SOURCES_R123/component_harness_GPU_r121b.py"
