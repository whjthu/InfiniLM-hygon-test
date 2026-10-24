#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
python -B /refactor/incoming/Deep_B64_M64_preflight_r8644.py
