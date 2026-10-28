#!/usr/bin/env bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
cd /refactor/incoming/Kimi-W13-wavepartial-component-r155
exec timeout 900s python3 -B -u CPU_build_r155.py
