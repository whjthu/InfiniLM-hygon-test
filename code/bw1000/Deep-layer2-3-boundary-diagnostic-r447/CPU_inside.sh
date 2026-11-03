#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export USE_TORCH=0 USE_TF=0 USE_FLAX=0
unset LD_PRELOAD
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:/refactor/candidates/latest-family-native-r269/Deep/python:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
S=/refactor/incoming/Deep-layer2-3-boundary-diagnostic-r447
export PYTHONPATH="$S:/refactor/incoming/Deep-T1-MoE-activation-quant-r438/model-diagnostic-v2:/old/offline-deps"
export DEEP438_DIAG_TEACHER=/refactor/results/Deep438-diagnostic-v2-private/teacher.json
timeout --kill-after=10s 270 python3 -B -u "$S/build_CPU.py"
