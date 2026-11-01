#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
python3 -B /refactor/incoming/Deep-Runtime-boundary-fusion-r344/CPU_build.py
