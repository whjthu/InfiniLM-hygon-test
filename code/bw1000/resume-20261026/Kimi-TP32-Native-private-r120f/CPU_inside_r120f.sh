#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 GPU_DEVICE_ORDINAL=-1
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 GPU_DEVICE_ORDINAL=-1
export TORCH_LIB_DIR=/usr/local/lib/python3.10/dist-packages/torch/lib
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$SDK/lib:$COMMON/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
timeout --kill-after=10s 800 python -B -u /refactor/incoming/Kimi-TP32-Native-r120f/CPU_build_r120f.py
