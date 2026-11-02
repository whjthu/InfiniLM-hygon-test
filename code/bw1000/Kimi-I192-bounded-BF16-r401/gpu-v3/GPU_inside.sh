#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
unset INFINILM_KIMI401_METADATA_PACK_DISABLED INFINILM_KIMI401_STATIC_DIV_DISABLED CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL LD_PRELOAD INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export PYTHONDONTWRITEBYTECODE=1 ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
export LD_LIBRARY_PATH="/refactor/install/hygon-kimi-5f91ad7-r3672/lib:$LM/python/infinilm/_lib:$LM/python:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec timeout --kill-after=15s 570 python3 -B -u /refactor/incoming/Kimi-I192-bounded-BF16-r401-gpu-v3/GPU_entry.py --manifest-sha "$KIMI401_GPU_SOURCE_SHA256" --deadline-unix "$KIMI401_DEADLINE_UNIX"
