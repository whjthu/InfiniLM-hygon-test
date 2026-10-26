#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONPYCACHEPREFIX=/local/PP4-map-r9154
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B -u /refactor/incoming/Kimi-PP4-crossalloc-r9154/map_probe.py
