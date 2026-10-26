#!/bin/bash
set -euo pipefail
ulimit -c 0
# CPU-only alternate entry; original frozen r120 package remains byte-for-byte.
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
# Existing root-provided container path; no Torch import merely for discovery.
export TORCH_LIB_DIR=/usr/local/lib/python3.10/dist-packages/torch/lib
test -d "$TORCH_LIB_DIR"
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPYCACHEPREFIX="/local/TP32-I96-CPU-r120c-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
bash -n /refactor/incoming/Kimi-TP32-I96-MoE-r120/GPU_inside_PROPOSED.sh
python3 -B /refactor/incoming/Kimi-TP32-I96-MoE-r120/CPU_contract_r120.py
python3 -B /refactor/incoming/Kimi-TP32-I96-MoE-r120/CPU_build_r120c.py
