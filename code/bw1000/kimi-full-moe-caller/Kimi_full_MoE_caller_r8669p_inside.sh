#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="/local/Kimi-full-MoE-caller-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset LD_PRELOAD INFINILM_KIMI_W13_WAVE_ACT_R8660 INFINILM_KIMI_PACKED_W2_R6122 INFINILM_KIMI_PACKED_AUTO_R6136
export LD_PRELOAD="/refactor/candidates/Kimi-full-MoE-caller-r8669p/libKimi_HIP_caller_sample_r8669p.so"
export INFINILM_KIMI_MOE_WORKSPACE_R4665=1 INFINILM_KIMI_MOE_EXECUTION_R4665=1 INFINILM_KIMI_SPEC_MATH_R3676=0
timeout --kill-after=10s 540 python /refactor/incoming/Kimi_full_MoE_caller_runner_r8669p.py /refactor/incoming/Kimi-full-MoE-caller-manifest-r8669p.json /refactor/results/Kimi-full-MoE-caller-r8669p-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json
