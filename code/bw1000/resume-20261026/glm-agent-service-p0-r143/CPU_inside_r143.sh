#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
timeout --kill-after=5s 240 python -B -u /refactor/incoming/GLM-agent-service-r143/prepare_overlay_r143.py
