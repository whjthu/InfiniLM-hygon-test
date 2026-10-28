#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
timeout --kill-after=5s 240 python -B -u /refactor/incoming/Deep-C128-cache-component-r152/CPU_build_r152.py
