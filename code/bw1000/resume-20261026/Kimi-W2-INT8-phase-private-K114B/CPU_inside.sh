#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONPYCACHEPREFIX=/local/K114B-CPU
timeout --kill-after=5s 840 python -B -u /refactor/incoming/Kimi-W2-INT8-phase-K114B/CPU_build_K114B.py
