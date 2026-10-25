#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/candidates/glm-hybrid-decode-r5386:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/refactor/candidates/glm-grouped-native-model-r5117/helpers:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
unset RDMAV_DRIVERS NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_BUFFSIZE
export GLM_CROSSREQUEST_PREFILL_R7328=0 GLM_SHARED_PREFIX_VALIDATE_R5814=0 GLM_BOUNDED_ROPE_R4279=1 GLM_SHARED_CONTROL_R4279=1
export GLM_EXACT_BANKS_R4188=1 GLM_PROJECTED_HOST_R4504=1 GLM_GROUPED_ROUTED_R5123=1 GLM_DECODE_ROW_DEDUP_R5386=0
export GLM_BATCH_ALLSELECTED_R5096=1 GLM_ASYNC_PROJECTED_HOST_R5011=0 GLM_GROUP_PROJECTED_HOST_R5030=0 GLM_ACTUAL_CAPTURE_R5164=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=ib NCCL_IB_DISABLE=0
PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
export LD_LIBRARY_PATH="$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:$LD_LIBRARY_PATH"
export RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx"
export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
timeout --kill-after=15s 300 python -u /refactor/incoming/GLM_prefill_split_profile_r8996.py

