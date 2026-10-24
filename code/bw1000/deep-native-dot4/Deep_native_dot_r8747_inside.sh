#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
PARTIAL=/refactor/candidates/deep-partial-replay-r4792
PARENT=/refactor/candidates/deep-attention-FP32-5f91ad7-r3688
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
HELPER=/refactor/candidates/models-5f91ad7-r3675/Deep/helpers
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export LD_LIBRARY_PATH="$PARTIAL:$PARENT:$HELPER:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="/local/deep-native-dot-r8747-${SLURM_JOB_ID}-${SLURM_STEP_ID}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
unset INFINILM_DEEP_LAYER_TRACE_R4026 INFINILM_DEEP_OUTPUT_TILE_R4808 NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export INFINILM_OWNER_CPU_WIDTH_R2392=4
export INFINILM_DEEP_MOE_TAIL512_R7211=0
export INFINILM_DEEP_OUTPUT_DIRECT_R7247=0
python -B /refactor/incoming/deep_affinity_env_contract_r7166.py
timeout --kill-after=15s 540 python -B -u /refactor/incoming/Deep_native_dot_r8747.py
