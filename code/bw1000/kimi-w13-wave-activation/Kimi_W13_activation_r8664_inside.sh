#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="/local/Kimi-W13-act-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset LD_PRELOAD INFINILM_KIMI_W13_WAVE_ACT_R8660 INFINILM_KIMI_PACKED_W2_R6122 INFINILM_KIMI_PACKED_AUTO_R6136
python /refactor/incoming/kimi_W13_activation_runner_r8662.py /refactor/incoming/Kimi-W13-activation-manifest-r8664.json /refactor/results/Kimi-W13-activation-r8664-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json
