#!/bin/bash
set -euo pipefail
ulimit -c 0
test "$SLURM_JOB_ID" = 910371
if [ "$SLURM_PROCID" = 0 ]; then test "$(hostname -s)" = a02r3n03; else test "$SLURM_PROCID" = 1; test "$(hostname -s)" = a02r3n07; fi
source /opt/dtk/env.sh
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="/local/kimi391-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=8 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
unset NCCL_DMABUF_ENABLE NCCL_NET_GDR_LEVEL NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_BUFFSIZE RDMAV_DRIVERS
PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
export LD_LIBRARY_PATH="$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:$LD_LIBRARY_PATH"
export RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx"
export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
export INFINILM_KIMI_MOE_WORKSPACE_R4665=1 INFINILM_KIMI_MOE_EXECUTION_R4665=1 INFINILM_KIMI_I192_GROUPED_R8669Y=1 INFINILM_KIMI_I192_COMPACT_R8669ZC=1 INFINILM_KIMI_BATCH_I192_R9125=1 INFINILM_KIMI_SPEC_MATH_R3676=0 INFINILM_KIMI_PROFILE_R9125=0
export INFINILM_KIMI_W2_PAIR_R9280=0 INFINILM_KIMI_PACKED_W2_R6122=0 INFINILM_KIMI_PACKED_AUTO_R6136=0 INFINILM_MXFP4_TILE_TP_R2559=0 INFINILM_KIMI_METADATA_PACK_R5858=0 INFINILM_KIMI_BOUNDED_BF16_R6054=0 INFINILM_KIMI_STATIC_DIV_R6020=0
unset INFINILM_KIMI_COMPACT_CACHE_R4974 INFINILM_KIMI_TRACE_DIR INFINILM_KIMI_DIAG_DEVICE_SYNC INFINILM_KIMI_DIAG_STREAM_SYNC INFINILM_KIMI_ROUTER_M1_R4635 INFINILM_KIMI_MOE_AUX_M1_R4430
python -B -c 'import json,pathlib,sys;sys.path.insert(0,"/refactor/incoming/Kimi-MoE-prefill512-micro64-r391-gpu-v1");import controller;controller.verify(pathlib.Path("/refactor"),json.loads(pathlib.Path("/refactor/results/Kimi391-GPU-execution-GO-v1.json").read_bytes()))'
exec timeout --kill-after=15s 570 python -B -u /refactor/incoming/Kimi-MoE-prefill512-micro64-r391-v2/GPU_run.py /refactor/results/Kimi391-GPU-runtime-v1.json "/refactor/results/Kimi391-${SLURM_JOB_ID}.${SLURM_STEP_ID}-node${SLURM_PROCID}.json"
