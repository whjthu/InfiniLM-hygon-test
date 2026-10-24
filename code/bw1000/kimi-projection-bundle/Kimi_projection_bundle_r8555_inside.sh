#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export PYTHONPYCACHEPREFIX="/local/qkv-r8545-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:${PATH}
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export INFINILM_DENSE_PREFILL_F32_R2475=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=8
python /refactor/incoming/kimi_projection_bundle_r8555.py \
 /refactor/candidates/Kimi-projection-bundle-r8552/operands.json \
 /refactor/candidates/Kimi-projection-component-r8553/libKimi_projection_component_r8553.so \
 /refactor/results/Kimi-projection-bundle-r8555-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json
