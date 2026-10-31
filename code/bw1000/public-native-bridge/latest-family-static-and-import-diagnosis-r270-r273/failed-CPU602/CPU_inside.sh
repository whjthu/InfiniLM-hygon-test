#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
MF=$(sha256sum /refactor/incoming/latest-family-service-r271/SOURCE_MANIFEST.json | cut -d " " -f1)
exec timeout --kill-after=5s 220 python3 -B -u /refactor/incoming/latest-family-service-r271/CPU_install_r271.py --manifest-sha "$MF"
