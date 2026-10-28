#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTHONPATH=/refactor/incoming/Kimi-agent-service-Model-r194/runtime:/refactor/candidates/Kimi-agent-service-r194/python:/old/offline-deps
exec timeout --kill-after=10s "${KIMI_CLIENT_LIMIT_S:?}" python -B -u /refactor/incoming/Kimi-agent-service-Model-r194/client/client_r194.py --base-url http://127.0.0.1:43844/v1 --model Kimi-K3 --output-dir "/refactor/results/Kimi-agent-P0-client-r194-${GLOBAL_RUN_R2354:?}" --identity "/refactor/results/Kimi-agent-service-identity-r194-${GLOBAL_RUN_R2354}.json" --timeout 3600
