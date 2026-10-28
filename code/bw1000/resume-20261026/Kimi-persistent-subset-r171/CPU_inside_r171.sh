#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1
timeout --kill-after=5s 540 python3 -B -u /refactor/incoming/Kimi-persistent-subset-r171/CPU_build_r171.py
