#!/bin/bash
set -euo pipefail
export GLM_SHARED_PREFIX_VALIDATE_R5814=0
export GLM_V_ON_DEMAND_R403=1
source /opt/dtk/env.sh
export GLM_CROSSREQUEST_PREFILL_R7328=0
export GLM_PP_SIZE_R3603=2 GLM_CACHE_ROWS_R3627=1 GLM_CACHE_CAP_R3627=32768 USE_WEIGHT_CACHE_R3604=0
export GLM_STAGE_R5040="$SLURM_PROCID"
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
if [ "${GLM_R444_ENV_ONLY:-0}" = 1 ]; then
 export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
 export TORCH_LIB_DIR=/usr/local/lib/python3.10/dist-packages/torch/lib
else
 export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
fi
export LD_LIBRARY_PATH="/refactor/candidates/latest-family-native-r269/GLM/python:/refactor/candidates/glm-hybrid-decode-r5386:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/refactor/candidates/glm-grouped-native-model-r5117/helpers:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/refactor/incoming/GLM-W4-fewshot-quality-r444-startup-diagnostic-v4/provider:/refactor/candidates/latest-family-native-r269/GLM/python:/old/offline-deps"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
unset RDMAV_DRIVERS NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
test "${GLM_EXACT_BANKS_R4188:?}" = 0 -o "$GLM_EXACT_BANKS_R4188" = 1
case "${GLM_CHANNELS_R4363:?}" in
 0) unset NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS ;;
 8) export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 ;;
 *) exit 2 ;;
esac
unset NCCL_BUFFSIZE
test "${GLM_BOUNDED_ROPE_R4279:?}" = 1
test "${GLM_SHARED_CONTROL_R4279:?}" = 1
export GLM_PROJECTED_HOST_R4504=0
export GLM_ACTUAL_CAPTURE_R5164=0
export OMP_NUM_THREADS=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=ib NCCL_IB_DISABLE=0
if [ "$GLM_NETWORK_TRIAL" = shca ]; then
 PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
 export LD_LIBRARY_PATH="$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:${LD_LIBRARY_PATH:-}"
 export RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx"
 export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
else
 test "$GLM_NETWORK_TRIAL" = Socket
 export NCCL_NET=Socket
 unset NCCL_NET_PLUGIN UCX_MODULE_DIR UCX_TLS UCX_NET_DEVICES
fi
echo "GLM_NETWORK_R3573 actualrequested=$GLM_NETWORK_TRIAL interface=$NCCL_SOCKET_IFNAME backend=$NCCL_NET"

export LD_LIBRARY_PATH="/refactor/candidates/latest-family-native-r269/GLM/python:$LD_LIBRARY_PATH"
if [ "${GLM_R444_ENV_ONLY:-0}" = 1 ]; then
 printf 'GLM_R444_ENV_BEGIN\0'
 for key in "${!GLM_@}" "${!SLURM_@}" "${!NCCL_@}" LD_PRELOAD LD_LIBRARY_PATH INFINILM_OWNER_CPU_WIDTH_R2392 USE_WEIGHT_CACHE_R3604; do printf '%s\0%s\0' "$key" "${!key}"; done
 exit 0
fi
SOURCE=/refactor/incoming/GLM-W4-fewshot-quality-r444-startup-diagnostic-v4/provider
DEPLOY=/refactor/incoming/GLM-W4-fewshot-quality-r444-startup-diagnostic-v4/GPU-final-v1
ARGS=(--config "$DEPLOY/config-slot${GLM_R444_SLOT:?}.json" --config-sha "${GLM_R425_CONFIG_SHA:?}" --source-sha "${GLM_R425_SOURCE_SHA:?}" --stage "$SLURM_PROCID" --run "${GLM_R425_RUN:?}" --actual-step "$SLURM_JOB_ID.$SLURM_STEP_ID" --gpu-start-epoch "${GLM_R425_GPUSTART:?}")
python3 -B -m public_glm_native "${ARGS[@]}"
LEFT=$(python3 -B -c 'import os,time;print(max(0,int(float(os.environ["GLM_R425_GPUSTART"])+28800-time.time())))')
test "$LEFT" -gt 120
timeout --kill-after=20s "$LEFT" python3 -B -u -m public_glm_native "${ARGS[@]}" --serve
