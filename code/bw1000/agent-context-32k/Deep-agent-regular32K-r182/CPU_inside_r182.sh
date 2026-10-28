#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1 USE_TORCH=0 USE_TF=0 USE_FLAX=0
unset LD_PRELOAD
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
timeout --kill-after=5s 300 python -B -u /refactor/incoming/Deep-agent-regular32K-r182/CPU_preflight_r182.py
