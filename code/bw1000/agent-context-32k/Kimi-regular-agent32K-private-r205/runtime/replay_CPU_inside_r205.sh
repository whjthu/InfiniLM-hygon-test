#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/old/offline-deps
exec timeout --kill-after=5s 180 python -B -u /refactor/incoming/Kimi-agent-service-Model-r205/runtime/replay_saved477_r205.py --output /refactor/results/Kimi477-saved59-strict-replay-r205.json
