#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export USE_TORCH=0 USE_TF=0 PYTHONPATH=/refactor/incoming/GLM-PP4-capacity-r178:/refactor/incoming:/old/offline-deps:${PYTHONPATH:-}
timeout --kill-after=10s 180 python3 -B -u /refactor/incoming/GLM-PP4-capacity-r178/CPU_preflight_r178.py
