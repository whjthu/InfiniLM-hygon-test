#!/bin/bash
set -euo pipefail
ulimit -c 0
test "$SLURM_JOB_ID" = 910371
case "$SLURM_PROCID:$(hostname -s)" in 0:a02r3n03|1:a02r3n07) ;; *) exit 2;; esac
source /opt/dtk/env.sh
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR=/usr/local/lib/python3.10/dist-packages/torch/lib
test -d "$TORCH_LIB_DIR"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
test -f "$ROCBLAS_TENSILE_LIBPATH/TensileLibrary_lazy_gfx936.dat"
export PYTHONPATH="$COMMON/python:/old/offline-deps"
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/usr/local/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export PYTHONPYCACHEPREFIX="/local/Kimi-full-MoE-r437-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=8
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib
unset NCCL_DMABUF_ENABLE NCCL_NET_GDR_LEVEL NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
export LD_LIBRARY_PATH="$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:$LD_LIBRARY_PATH"
export RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx"
export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
export INFINILM_OWNER_CPU_WIDTH_R2392=4
export INFINILM_KIMI_I192_GROUPED_R8669Y=0 INFINILM_KIMI_I192_COMPACT_R8669ZC=1 INFINILM_KIMI_BATCH_I192_R9125=1
export INFINILM_KIMI_MOE_WORKSPACE_R4665=1 INFINILM_KIMI_MOE_EXECUTION_R4665=0 INFINILM_KIMI_SPEC_MATH_R3676=0
export INFINILM_MXFP4_TILE_TP_R2559=0 INFINILM_KIMI_METADATA_PACK_R5858=0 INFINILM_KIMI_BOUNDED_BF16_R6054=0 INFINILM_KIMI_STATIC_DIV_R6020=0
export INFINILM_KIMI_PACKED_W2_R6122=0 INFINILM_KIMI_PACKED_AUTO_R6136=0
unset INFINILM_KIMI_PROFILE_R9125 INFINILM_KIMI_MOE_AUX_M1_R4430 INFINILM_KIMI_ROUTER_M1_R4635 INFINILM_KIMI_W13_WAVE_ACT_R8660
LEFT=$(python3 -B -c 'import os,time;print(max(0,int(float(os.environ["KIMI_R437_GPU_START"])+570-time.time())))')
test "$LEFT" -gt 30
exec timeout --kill-after=15s "$LEFT" python3 -B -u /refactor/incoming/Kimi-full-MoE-lowrank-M1-r437/gpu-v8/GPU_run.py
