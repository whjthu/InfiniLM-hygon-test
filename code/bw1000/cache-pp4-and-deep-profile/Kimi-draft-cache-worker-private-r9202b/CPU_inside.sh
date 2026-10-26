#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /refactor/incoming/Kimi-draft-cache-worker-r9202b
export PYTHONPATH=/refactor/incoming:${PYTHONPATH:-}
timeout --kill-after=5s 160 python -B -u CPU_contract.py
