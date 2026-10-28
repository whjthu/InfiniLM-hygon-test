#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
timeout --kill-after=5s 180 python -B -u /refactor/incoming/GLM-arrival-defer-r153/CPU_prepare_r153.py
