#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
timeout --kill-after=5s 150 python -B -u /refactor/incoming/tiny_dflash2_forward_r9000.py
