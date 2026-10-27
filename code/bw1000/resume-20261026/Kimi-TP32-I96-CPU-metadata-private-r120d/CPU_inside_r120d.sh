#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR=/usr/local/lib/python3.10/dist-packages/torch/lib
test -d "$TORCH_LIB_DIR"
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPYCACHEPREFIX="/local/TP32-I96-CPU-r120d-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
python3 -B /refactor/incoming/Kimi-TP32-I96-CPU-metadata-r120d/verify_CPU_r120d.py
