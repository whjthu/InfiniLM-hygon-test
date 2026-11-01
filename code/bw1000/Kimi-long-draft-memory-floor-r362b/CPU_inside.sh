#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec timeout --kill-after=5s 260 python3 -B -u /refactor/incoming/Kimi-long-draft-memory-floor-r362b/CPU_main.py
