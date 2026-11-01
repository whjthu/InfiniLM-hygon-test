#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
exec timeout --kill-after=5s 180 python3 -B /refactor/incoming/Deep-prefill-chunk-pair-r348/controller/CPU_contract.py
