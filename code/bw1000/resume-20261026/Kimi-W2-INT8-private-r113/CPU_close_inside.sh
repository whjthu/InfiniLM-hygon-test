#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONPYCACHEPREFIX=/local/Kimi-W2-INT8-CPUclose-r113
timeout --kill-after=5s 700 python -B -u /refactor/incoming/Kimi-W2-INT8-r113/CPU_close_r113.py
