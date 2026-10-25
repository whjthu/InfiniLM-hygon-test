#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
python /refactor/incoming/build_Kimi_serial_CPU_r9020.py
