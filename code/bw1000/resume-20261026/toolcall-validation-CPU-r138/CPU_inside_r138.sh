#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
unset LD_PRELOAD PYTHONPATH
export PYTHONPATH=/old/offline-deps
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
timeout --kill-after=5s 180 python -B -u /refactor/incoming/toolcall-validation-CPU-r138/CPU_static_r138.py
