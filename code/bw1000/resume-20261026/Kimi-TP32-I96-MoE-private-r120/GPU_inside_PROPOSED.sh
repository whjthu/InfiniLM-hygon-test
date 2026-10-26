#!/bin/bash
set -euo pipefail
ulimit -c 0
# No srun here: future root keeper supplies exactly4 owned nodes/8DCUs each.
test "${R120_COMPONENT_GO:-0}" = 1
test -n "${R120_RUNTIME_MANIFEST:-}"
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
export LD_LIBRARY_PATH="$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="/local/Kimi-TP32-r120-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib
unset NCCL_DMABUF_ENABLE NCCL_NET_GDR_LEVEL NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx"
export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=4
unset LD_PRELOAD INFINILM_KIMI_W13_WAVE_ACT_R8660 INFINILM_KIMI_PACKED_W2_R6122 INFINILM_KIMI_PACKED_AUTO_R6136
export INFINILM_KIMI_I192_GROUPED_R8669Y=0 INFINILM_KIMI_BATCH_I192_R9125=0 INFINILM_KIMI_PROFILE_R9125=0 INFINILM_KIMI_I192_COMPACT_R8669ZC=0
export INFINILM_KIMI_MOE_WORKSPACE_R4665=0 INFINILM_KIMI_MOE_EXECUTION_R4665=0 INFINILM_KIMI_SPEC_MATH_R3676=0 INFINILM_KIMI_W2_PAIR_R9280=0
export INFINILM_MXFP4_TILE_TP_R2559=0 INFINILM_KIMI_METADATA_PACK_R5858=0 INFINILM_KIMI_BOUNDED_BF16_R6054=0 INFINILM_KIMI_STATIC_DIV_R6020=0
output="/refactor/results/Kimi-TP32-I96-MoE-r120-${SLURM_JOB_ID}.${SLURM_STEP_ID}-node${SLURM_PROCID}.json"
timeout --kill-after=10s 720 python -B -u /refactor/incoming/Kimi-TP32-I96-MoE-r120/run_component_r120.py "$R120_RUNTIME_MANIFEST" "$output"
