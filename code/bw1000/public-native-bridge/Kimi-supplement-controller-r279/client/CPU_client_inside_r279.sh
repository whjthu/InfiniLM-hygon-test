#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export PYTHONPATH=/refactor/incoming/Kimi-latest-service-r275/runtime:/refactor/candidates/Kimi-latest-service-r262/python:/old/offline-deps
exec timeout --kill-after=10s "${KIMI_CLIENT_LIMIT_S:?}" python -B -u /refactor/incoming/Kimi-supplement-controller-r279/client/client_r278.py --base-url http://127.0.0.1:43844/v1 --model Kimi-K3 --output-dir "/refactor/results/Kimi-agent-P0-client-r275-${GLOBAL_RUN_R2354:?}" --identity "/refactor/results/Kimi-latest-service-identity-r275-${GLOBAL_RUN_R2354}.json" --service-report "/refactor/results/Kimi-latest-service-r275-${GLOBAL_RUN_R2354}-node0.json" --timeout 11200
