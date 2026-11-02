#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
timeout --signal=TERM --kill-after=5s 150 python -B /refactor/incoming/Kimi-draft-query-warm-cost-r379/CPU_main.py
