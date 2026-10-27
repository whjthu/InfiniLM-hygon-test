#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
timeout --kill-after=10s 1830 python -B -u /refactor/incoming/GLM-agent-P0-client-r145/client_driver_r145.py
