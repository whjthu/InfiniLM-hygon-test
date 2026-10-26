#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONPYCACHEPREFIX=/local/Kimi-W2-pair-CPU-r9280
timeout --kill-after=5s 660 python -B -u /refactor/incoming/Kimi-W2-pair-r9280/CPU_build_r9280.py
