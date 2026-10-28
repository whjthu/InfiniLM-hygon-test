#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1
: "${1:?publication commit required}"
exec timeout --kill-after=10s 875s python3 -B -u /refactor/incoming/Kimi-I96-storage-launch-r168a/controller_r168a.py "$1"
