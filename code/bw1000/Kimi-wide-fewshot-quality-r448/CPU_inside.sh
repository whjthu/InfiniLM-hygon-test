#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
unset LD_PRELOAD GPU_DEVICE_ORDINAL INFINILM_OWNER_CPU_WIDTH_R2392 BASH_ENV ENV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/refactor/incoming/Kimi-public-incremental-text-r434:/refactor/incoming/public-Kimi-native-provider-r428/runtime-v2:/old/offline-deps
exec timeout --kill-after=10s 270 python3 -B -u /refactor/incoming/Kimi-wide-fewshot-quality-r448/CPU_encode.py
