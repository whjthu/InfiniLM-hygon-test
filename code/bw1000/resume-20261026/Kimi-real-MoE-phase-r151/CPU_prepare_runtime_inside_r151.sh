#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
: "${CPU_BUILD_SHA_R151:?root must provide actual CPU build receipt SHA}"
timeout --kill-after=5s 120 python -B -u /refactor/incoming/Kimi-real-MoE-phase-r151/prepare_runtime_r151.py --CPU-build-sha "$CPU_BUILD_SHA_R151"
