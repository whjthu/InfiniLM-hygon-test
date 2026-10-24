#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')"
export LD_LIBRARY_PATH="/refactor/candidates/glm-hybrid-decode-r5386:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/refactor/candidates/glm-grouped-native-model-r5117/helpers:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/refactor/incoming:$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export GLM_BOUNDED_ROPE_R4279=1 GLM_SHARED_CONTROL_R4279=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
echo "GLM_L0_R5759 numeric=${SLURM_JOB_ID}.${SLURM_STEP_ID} node=$(hostname)"
timeout --kill-after=10s 300 python -u /refactor/incoming/GLM_layer0_finite_r5760.py
