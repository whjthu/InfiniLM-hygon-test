#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export PYTHONPYCACHEPREFIX="/local/oproj-r8588-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:${PATH}
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export INFINILM_DENSE_PREFILL_F32_R2475=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=8
python /refactor/incoming/kimi_o_proj_runner_r8584.py \
 /refactor/candidates/Kimi-o-proj-component-r8581/operands.json \
 /refactor/candidates/Kimi-o-proj-component-build-r8583/libKimi_o_proj_component_r8583.so \
 /refactor/results/Kimi-o-proj-component-r8588-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json
