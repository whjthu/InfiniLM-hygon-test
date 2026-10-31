#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/refactor/candidates/Kimi-latest-service-r262/python:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python:/old/offline-deps"
MF=$(sha256sum /refactor/incoming/latest-family-entry-r272/SOURCE_MANIFEST.json | cut -d " " -f1)
exec timeout --kill-after=5s 220 python3 -B -u /refactor/incoming/latest-family-entry-r272/CPU_prepare_r272.py --manifest-sha "$MF"
