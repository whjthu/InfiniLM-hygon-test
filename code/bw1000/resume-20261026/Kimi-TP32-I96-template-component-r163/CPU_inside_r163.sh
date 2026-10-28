#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1
timeout --kill-after=5s 900 python3 -B -u /refactor/incoming/Kimi-TP32-I96-template-component-r163/CPU_build_r163.py
