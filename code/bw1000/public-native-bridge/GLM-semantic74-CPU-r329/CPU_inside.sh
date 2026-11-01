#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
OUT=/refactor/candidates/GLM-semantic74-CPU-r329
mkdir "$OUT"
exec timeout --kill-after=5s 240 python3 -B -u /refactor/incoming/GLM-semantic74-CPU-r329/CPU_main.py --source /refactor/incoming/GLM-semantic74-owner-r328 --source-manifest-sha "$GLM_OWNER_SOURCE_SHA" --output "$OUT/CPU-result.json"
