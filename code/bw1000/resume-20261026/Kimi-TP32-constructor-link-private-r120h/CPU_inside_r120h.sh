#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 GPU_DEVICE_ORDINAL=-1
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 GPU_DEVICE_ORDINAL=-1
export TORCH_LIB_DIR=/usr/local/lib/python3.10/dist-packages/torch/lib
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$SDK/lib:$COMMON/python/infinilm/_lib:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:/refactor/candidates/kimi-helper-bounded-bf16-r6054:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
timeout --kill-after=10s 480 python -B -u /refactor/incoming/Kimi-TP32-constructor-link-r120h/CPU_link_r120h.py
