#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= GPU_DEVICE_ORDINAL=
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/refactor/incoming/GLM-chunk32-arrival-r9258:/old/offline-deps:${PYTHONPATH:-}
python -B -u /refactor/incoming/GLM-chunk32-arrival-r9258/CPU_contract.py --torch-world
