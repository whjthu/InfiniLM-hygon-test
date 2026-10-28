#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONPYCACHEPREFIX=/local/Kimi-real-MoE-phase-CPU-r151
timeout --kill-after=5s 840 python -B -u /refactor/incoming/Kimi-real-MoE-phase-r151/CPU_build_r151.py
