#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
python -u /refactor/incoming/build_GLM_fused_CPU_r9028.py
