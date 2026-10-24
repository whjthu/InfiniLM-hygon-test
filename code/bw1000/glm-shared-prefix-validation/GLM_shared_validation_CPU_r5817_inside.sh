#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
python3 /refactor/incoming/build_GLM_shared_validation_r5817.py
