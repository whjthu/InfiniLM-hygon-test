#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset LD_PRELOAD
exec timeout --kill-after=5s 270 python3 -B -u /refactor/incoming/Kimi-I192-Model-pair-r389/CPU_main.py
