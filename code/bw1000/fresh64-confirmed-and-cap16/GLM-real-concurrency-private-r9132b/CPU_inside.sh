#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPATH=/refactor/incoming:/old/offline-deps:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
python -B -u /refactor/incoming/GLM-real-concurrency-r9132b/CPU_closure.py
