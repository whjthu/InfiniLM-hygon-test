#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1 USE_TORCH=0 USE_TF=0 USE_FLAX=0
unset LD_PRELOAD
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
timeout --kill-after=5s 240 python -B -u /refactor/incoming/Deep-offline-replay-r195b/replay_r195.py
