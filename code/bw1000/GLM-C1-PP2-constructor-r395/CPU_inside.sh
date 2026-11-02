#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
unset LD_PRELOAD
exec timeout --kill-after=5s 270 python3 -B -u /refactor/incoming/GLM-C1-PP2-constructor-r395/CPU_main.py
