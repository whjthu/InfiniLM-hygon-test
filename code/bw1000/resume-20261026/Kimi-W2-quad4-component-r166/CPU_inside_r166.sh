#!/usr/bin/env bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
cd /refactor/incoming/Kimi-W2-quad4-component-r166
exec timeout 720s python3 -B -u CPU_build_r166.py
