#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
MF=$(sha256sum /refactor/incoming/latest-family-native-r269/SOURCE_MANIFEST.json | cut -d " " -f1)
exec timeout --kill-after=5s 220 python3 -B -u /refactor/incoming/latest-family-native-r269/CPU_build_r269.py --manifest-sha "$MF"
