#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1 USE_TORCH=0 USE_TF=0 USE_FLAX=0
unset LD_PRELOAD
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
timeout --kill-after=5s 180 python -B -u /refactor/incoming/Deep-history-preserve-r205/CPU_saved_request_r205.py
