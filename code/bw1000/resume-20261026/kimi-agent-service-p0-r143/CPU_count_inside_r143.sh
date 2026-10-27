#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPATH="/refactor/incoming/kimi-agent-service-p0-r143:/old/offline-deps:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1
timeout --kill-after=5s 120 python -B -u /refactor/incoming/kimi-agent-service-p0-r143/probe_context_r143.py --output /refactor/results/Kimi-agent-token-count-r143.json
