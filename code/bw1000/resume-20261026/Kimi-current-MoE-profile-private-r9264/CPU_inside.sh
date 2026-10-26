#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONPYCACHEPREFIX=/local/Kimi-current-MoE-CPU-r9264
timeout --kill-after=5s 420 python -B -u /refactor/incoming/Kimi-current-MoE-profile-r9264/CPU_build_r9264.py
