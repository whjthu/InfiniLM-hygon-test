#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export USE_TORCH=0 USE_TF=0
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
timeout --kill-after=10s 300 python3 -B -u /refactor/incoming/GLM-longcontext-audit-r177b/CPU_longtoken_preflight_r177b.py
