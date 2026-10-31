#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 USE_TORCH=0 USE_TF=0
timeout --kill-after=10s 220 python -B -u /refactor/incoming/GLM-owner-entry-r291/CPU_prepare_r291.py --manifest-sha ${GLM_CPU_MANIFEST_SHA:?}
