#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONPATH=/old/offline-deps:${PYTHONPATH:-}
python -B -u /refactor/incoming/Kimi-I192-batch-Model-r9129/CPU_build.py
