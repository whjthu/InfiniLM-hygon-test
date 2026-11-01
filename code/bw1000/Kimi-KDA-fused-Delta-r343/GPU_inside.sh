#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONDONTWRITEBYTECODE=1
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:${PATH}
SOURCE=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
TORCH_LIB_DIR=$(python -c 'import pathlib,importlib.util;print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')
export PYTHONPATH="$SOURCE/python:/old/offline-deps:${PYTHONPATH:-}"
PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$SOURCE/python:$SOURCE/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD LD_AUDIT
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib
export NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca
export UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx" UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
export OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=8 PYTHONFAULTHANDLER=1
export KIMI_USE_RAMCACHE_R3679=0 KIMI_REDUCE_POLICY=0 INFINILM_OWNER_CPU_WIDTH_R2392=4
export INFINILM_DENSE_PREFILL_F32_R2475=1 INFINILM_MXFP4_TILE_TP_R2559=0
export INFINILM_KIMI_SPEC_R3632=1 INFINILM_KIMI_PAIR_TP16_R9006=1 INFINILM_KIMI_SPEC_MATH_R3676=0
export INFINILM_KIMI_ROWAR_R9043=0 INFINILM_KIMI_ROWAR_TRACE_R9043=0 INFINILM_KIMI_KDA_SERIAL_R9027=0 KIMI_STATE_CAPTURE_R9113=0
export INFINILM_KIMI_KDA_INPUT_M1_R4305=0 INFINILM_KIMI_KDA_GATE_M1_R4576=0 INFINILM_KIMI_KDA_GATE_M1_R4571=0
export INFINILM_KIMI_CCL512_R8603=0 INFINILM_KIMI_PREFILL512_R8562=0 INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562=0
unset INFINILM_KIMI_TRACE_DIR INFINILM_KIMI_KDA_TRACE_DIR INFINILM_KIMI_KDA_TRACE_LAYER INFINILM_KIMI_TARGET_TAPS_DIR_R9022
unset INFINILM_KIMI_DIAG_DEVICE_SYNC INFINILM_KIMI_DIAG_STREAM_SYNC NCCL_DMABUF_ENABLE NCCL_NET_GDR_LEVEL
exec timeout --kill-after=15s 540 python -B -u /refactor/incoming/Kimi-KDA-fused-Delta-r343/driver.py
