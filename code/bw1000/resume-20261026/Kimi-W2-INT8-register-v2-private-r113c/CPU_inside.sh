#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONPYCACHEPREFIX=/local/Kimi-i8-register-v2-CPU-r113c
timeout --kill-after=5s 840 python -B -u /refactor/incoming/Kimi-W2-INT8-register-v2-r113c/CPU_build_r113v2.py
