#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1
exec timeout --kill-after=20s 120 python -B /refactor/incoming/Deep-C1-decode-profile-r168a/CPU_static_r168a.py
