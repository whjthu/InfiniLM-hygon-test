#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
: "${RECOVERY_SOURCE_SHA_R151A:?root must pin recovery source manifest}"
timeout --kill-after=5s 240 python -B -u /refactor/incoming/Kimi-real-MoE-phase-recovery-r151a/CPU_validate_recovery_r151a.py --source-manifest-sha "$RECOVERY_SOURCE_SHA_R151A"
