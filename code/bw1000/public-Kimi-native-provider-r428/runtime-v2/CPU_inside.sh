#!/bin/bash
set -euo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/refactor/incoming/public-Kimi-native-provider-r428/runtime-v2:/old/offline-deps
exec timeout --kill-after=5s 270 python3 -B -u /refactor/incoming/public-Kimi-native-provider-r428/runtime-v2/CPU_main.py
