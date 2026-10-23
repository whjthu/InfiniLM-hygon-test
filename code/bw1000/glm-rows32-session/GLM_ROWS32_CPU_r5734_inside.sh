#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPATH="/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python:/old/offline-deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
python -u /refactor/incoming/build_GLM_ROWS32_CPU_r5734.py
