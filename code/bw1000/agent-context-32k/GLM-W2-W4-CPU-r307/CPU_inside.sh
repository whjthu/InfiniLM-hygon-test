#!/bin/bash
set -euo pipefail
ulimit -c 0
unset LD_PRELOAD
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/refactor/incoming/GLM-prefill-W2-W4-owner-r307:/old/offline-deps
exec timeout --kill-after=5s 240 python -B /refactor/incoming/GLM-W2-W4-CPU-r307/CPU_main.py --source /refactor/incoming/GLM-prefill-W2-W4-owner-r307 --output /refactor/candidates/GLM-W2-W4-CPU-r307/CPU-inside.json
