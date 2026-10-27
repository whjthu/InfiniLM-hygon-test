#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
unset LD_PRELOAD
ROOT=/refactor
COMMON="$ROOT/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM"
SDK="$ROOT/install/hygon-deep-5f91ad7-r3672"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:$ROOT/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
timeout --kill-after=10s 220 python3 -B /refactor/incoming/Deep-prefix-native-CPU-metadata-r128b/verify_CPU_r128b.py
