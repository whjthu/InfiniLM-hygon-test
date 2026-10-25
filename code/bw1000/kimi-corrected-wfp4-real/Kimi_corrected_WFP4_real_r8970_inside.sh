#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="/local/Kimi-full-MoE-caller-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
unset LD_PRELOAD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
timeout --kill-after=10s 240 python /refactor/incoming/Kimi_corrected_WFP4_real_worker_r8970.py
