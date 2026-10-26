#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
cd /refactor/incoming/Kimi-resident-header-budget-r9208
timeout --kill-after=5s 160 python -B -u CPU_budget.py
