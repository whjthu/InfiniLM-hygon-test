#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
exec timeout --kill-after=5s 220 python3 -B -u /refactor/incoming/Deep-owner-cleanup-r289/CPU_prepare_r289.py --manifest-sha "$DEEP_CPU_MANIFEST_SHA"
