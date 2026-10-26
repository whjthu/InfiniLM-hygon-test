#!/bin/bash
set -euo pipefail
bash -n "$0"
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
python3 /refactor/incoming/Kimi-MoE-event-profile-r9220e/build_CPU_r9220e.py
