#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1 USE_TORCH=0 USE_TF=0 USE_FLAX=0
unset LD_PRELOAD
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
timeout --kill-after=5s 240 python -B -u /refactor/incoming/GLM-request-constraints-r203b/CPU_saved_requests_r203.py --source "${GLM_SOURCE_DIR_R203:-/refactor/incoming/GLM-regular-service-r192}" --run "${GLM_SOURCE_RUN_R203:?root must provide exact saved run}"
