#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 USE_TORCH=0 USE_TF=0
export PYTHONPATH=/refactor/incoming/GLM-regular-fixture-r178a:/old/offline-deps:${PYTHONPATH:-}
timeout --kill-after=10s 180 python3 -B -u /refactor/incoming/GLM-regular-fixture-r178a/CPU_prepare_profiles_r178a.py
