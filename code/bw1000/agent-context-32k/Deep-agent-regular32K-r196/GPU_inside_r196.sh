#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONDONTWRITEBYTECODE=1
unset USE_TORCH USE_TF USE_FLAX
export PYTHONPYCACHEPREFIX="/local/Deep-regular-r196-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/refactor/candidates/Deep-agent-regular32K-r196/python:$LM/python:/old/offline-deps:${PYTHONPATH:-}"
unset INFINILM_DEEP_LAYER_TRACE_R4026 NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS INFINILM_DEEP_QUALITY_PREFLIGHT_ONLY_R9101 INFINILM_DEEP_POSITION_PROOF_R8726
export OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=4
export INFINILM_DEEP_MOE_TAIL512_R7211=0 INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750=0
export DEEP_REPLICA_R9278=n01
python -B /refactor/incoming/deep_affinity_env_contract_r7166.py
timeout --kill-after=30s 15000 python -B -u /refactor/incoming/Deep-agent-regular32K-r196/bootstrap_r196.py
