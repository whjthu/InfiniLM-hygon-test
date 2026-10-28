#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export USE_TORCH=0 USE_TF=0 PYTHONPATH=/refactor/incoming/GLM-PP4-capacity-r186:/refactor/incoming:/old/offline-deps:${PYTHONPATH:-}
timeout --kill-after=10s 180 python3 -B -u /refactor/incoming/GLM-PP4-capacity-r186/CPU_preflight_r186.py --cpu-receipt-path /refactor/candidates/GLM-PP4-registry-repair-r184b/CPU-build.json --cpu-receipt-sha 6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529
