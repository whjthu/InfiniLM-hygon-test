#!/bin/bash
# Exact successful CPU1080 path prefix, plus unchanged GPU-v1 math/preload policy.
source /opt/dtk/env.sh
export PYTHONDONTWRITEBYTECODE=1
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:${PATH}
SOURCE=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
TORCH_LIB_DIR=$(python -c 'import pathlib,importlib.util;print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')
export PYTHONPATH="$SOURCE/python:/old/offline-deps:${PYTHONPATH:-}"
PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$SOURCE/python:$SOURCE/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:${LD_LIBRARY_PATH:-}"
export INFINILM_DENSE_PREFILL_F32_R2475=1
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
