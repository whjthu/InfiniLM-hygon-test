#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONDONTWRITEBYTECODE=1
unset USE_TORCH USE_TF USE_FLAX
export PYTHONPYCACHEPREFIX="/local/Deep-public-fixed-cohort-r465-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:/refactor/candidates/latest-family-native-r269/Deep/python:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/refactor/incoming/Deep-public-fixed-cohort-profile-r465/source-v1:/refactor/incoming/public-Deep-native-provider-r427/source-v3:/refactor/candidates/latest-family-native-r269/Deep/python:/old/offline-deps"
unset INFINILM_DEEP_LAYER_TRACE_R4026 NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS INFINILM_DEEP_QUALITY_PREFLIGHT_ONLY_R9101 INFINILM_DEEP_POSITION_PROOF_R8726
export OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=4
export INFINILM_DEEP_MOE_TAIL512_R7211=0 INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750=0
DEPLOY=/refactor/incoming/Deep-public-C4-remaining-quality/deployment-v2/shard5
ARGS=(--config "$DEPLOY/config.json" --config-sha "${DEEP_R465_COHORT_CONFIG_SHA:?}" --source-sha "${DEEP_R465_COHORT_SOURCE_SHA:?}" --run "${DEEP_R465_COHORT_RUN:?}" --actual-step "$SLURM_JOB_ID.$SLURM_STEP_ID" --gpu-start-epoch "${DEEP_R465_COHORT_GPUSTART:?}")
python3 -B -m public_deep_cohort "${ARGS[@]}" --experimental-profile deep-hc0-fixed-cohort-c4
LEFT=$(python3 -B -c 'import os,time;print(max(0,int(float(os.environ["DEEP_R465_COHORT_GPUSTART"])+14400-time.time())))')
test "$LEFT" -gt 120
timeout --kill-after=20s "$LEFT" python3 -B -u -m public_deep_cohort "${ARGS[@]}" --experimental-profile deep-hc0-fixed-cohort-c4 --serve
