#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
MF=$(sha256sum /refactor/incoming/latest-family-service-r270/SOURCE_MANIFEST.json | cut -d " " -f1)
exec timeout --kill-after=5s 220 python3 -B -u /refactor/incoming/latest-family-service-r270/CPU_install_r270.py --manifest-sha "$MF"
