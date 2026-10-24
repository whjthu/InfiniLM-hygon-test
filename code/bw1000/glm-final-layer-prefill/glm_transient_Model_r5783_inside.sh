#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export GLM_CROSSREQUEST_PREFILL_R7328=0
python3 /refactor/incoming/GLM_transient_preflight_r5783.py
export GLM_STAGE_R5040="$SLURM_PROCID"
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/candidates/glm-hybrid-decode-r5386:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/refactor/candidates/glm-grouped-native-model-r5117/helpers:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
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
export GLM_PROJECTED_HOST_R4504=1
export GLM_ACTUAL_CAPTURE_R5164=0
export OMP_NUM_THREADS=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=ib NCCL_IB_DISABLE=0
if [ "$GLM_NETWORK_TRIAL" = shca ]; then
 PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
 export LD_LIBRARY_PATH="$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:$LD_LIBRARY_PATH"
 export RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx"
 export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
else
 test "$GLM_NETWORK_TRIAL" = Socket
 export NCCL_NET=Socket
 unset NCCL_NET_PLUGIN UCX_MODULE_DIR UCX_TLS UCX_NET_DEVICES
fi
echo "GLM_NETWORK_R3573 actualrequested=$GLM_NETWORK_TRIAL interface=$NCCL_SOCKET_IFNAME backend=$NCCL_NET"

if [ "${GLM_PREFLIGHT_ONLY_R5194:-0}" = 1 ]; then
 timeout --kill-after=10s 600 python -u /refactor/incoming/GLM_transient_preengine_r5783.py
 exit 0
fi

if [ "$GLM_STAGE_R5040" = 0 ]; then python -u /refactor/incoming/build_glm_group_cases_r5394.py; fi
timeout --kill-after=20s 3450 python -u /refactor/incoming/modern_glm_crossrequest_r5783.py
