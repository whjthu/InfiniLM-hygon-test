#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
: "${RECOVERED_BUILD_SHA_R151A:?root must pin actual new CPU receipt}"
timeout --kill-after=5s 120 python -B -u /refactor/incoming/Kimi-real-MoE-phase-recovery-r151a/prepare_runtime_recovered_r151a.py --CPU-build-sha "$RECOVERED_BUILD_SHA_R151A"
